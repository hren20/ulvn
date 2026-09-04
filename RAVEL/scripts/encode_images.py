#!/usr/bin/env python3
"""
scripts/encode_images.py
----------------

Offline image encoder. Encodes a directory of landmark images into
L2-normalized feature vectors using a selected CNN or Transformer
backbone (e.g., ResNet, DINOv2, MegaLoc) and saves the results.

This script generates two files:
  - features.npy: (float32, shape [N, D]) L2-normalized feature vectors.
  - paths.txt:    (text file, N lines) The paths to the images that were
                  successfully encoded, matching the order in features.npy.

Usage:
  # Example using default (MegaLoc SOTA VPR model)
  # Note: MegaLoc is trained on 320x320.
  python scripts/encode_images.py --img_dir data/landmark/ \
                                  --out_features tmp/run/data_features.npy \
                                  --out_paths tmp/run/data_paths.txt

  # Example using DINOv2 (strong foundation model)
  # Note: DINOv2 performs best at 224x224 or larger.
  python scripts/encode_images.py --img_dir data/landmark \
                                  --arch dinov2_vits14 \
                                  --img_size 224 224 \
                                  --out_features tmp/dino_features.npy \
                                  --out_paths tmp/dino_paths.txt

  # Example using ResNet-18
  python scripts/encode_images.py --img_dir data/landmark \
                                  --arch resnet18 \
                                  --img_size 85 64 \
                                  --out_features tmp/resnet_features.npy \
                                  --out_paths tmp/resnet_paths.txt

"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple, Optional, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset, DataLoader

# --- Imports for torch.hub models ---
import torch.hub

# Attempt to import torchvision
try:
    import torchvision
    from torchvision import transforms
except ImportError:
    print(
        "ERROR: torchvision is required. Install with `pip install torchvision`.",
        file=sys.stderr
    )
    sys.exit(1)
# --- End Imports ---

# Attempt to import tqdm for progress bar
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


# ----------------------------------------------------------------------------
#                           MODEL DEFINITIONS
# ----------------------------------------------------------------------------

class ResNetEmbeddings(nn.Module):
    """ResNet backbone wrapper for feature extraction.

    Strips the final classification layer and adds L2 normalization
    to the output (avgpool layer).

    Attributes:
        arch (str): The ResNet architecture (e.g., "resnet18").
        backbone (nn.Sequential): The ResNet model trunk.
        out_dim (int): The output feature dimension (512 or 2048).
    """
    def __init__(self, arch: str = "resnet18", pretrained: bool = True):
        super().__init__()
        arch = arch.lower()
        if arch not in {"resnet18", "resnet50"}:
            raise ValueError(
                f"Unsupported arch: {arch}. Must be 'resnet18' or 'resnet50'."
            )

        self.arch = arch
        weights = "DEFAULT" if pretrained else None

        # Load weights robustly across torchvision versions
        if arch == "resnet18":
            try:
                from torchvision.models import resnet18, ResNet18_Weights
                weights_enum = ResNet18_Weights.DEFAULT if pretrained else None
                net = resnet18(weights=weights_enum)
            except Exception:
                print("Using torchvision pretrained=True model loader.")
                net = torchvision.models.resnet18(pretrained=pretrained)
            dim = 512
        else:  # resnet50
            try:
                from torchvision.models import resnet50, ResNet50_Weights
                weights_enum = ResNet50_Weights.DEFAULT if pretrained else None
                net = resnet50(weights=weights_enum)
            except Exception:
                print("Using torchvision pretrained=True model loader.")
                net = torchvision.models.resnet50(pretrained=pretrained)
            dim = 2048

        # Remove the final fully-connected (classifier) layer
        modules = list(net.children())[:-1]  # All layers except the last one (fc)
        self.backbone = nn.Sequential(*modules)
        self.out_dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the backbone."""
        feats = self.backbone(x)          # (N, C, 1, 1)
        feats = feats.view(feats.size(0), -1)  # (N, C)
        feats = F.normalize(feats, p=2, dim=1)  # L2 normalize
        return feats


class DINOv2Embeddings(nn.Module):
    """DINOv2 wrapper for feature extraction.

    Uses the CLS token as the global image descriptor and L2-normalizes it.

    Attributes:
        arch (str): The DINOv2 architecture (e.g., "dinov2_vits14").
        backbone (nn.Module): The DINOv2 model.
        out_dim (int): The output feature dimension.
    """
    def __init__(self, arch: str = "dinov2_vits14"):
        super().__init__()
        supported_archs = {
            "dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"
        }
        if arch not in supported_archs:
            raise ValueError(
                f"Unsupported DINOv2 arch: {arch}. Must be one of {supported_archs}"
            )

        self.arch = arch
        try:
            self.backbone = torch.hub.load('facebookresearch/dinov2', arch)
        except Exception as e:
            print(f"ERROR: Failed to load DINOv2 from torch.hub. Check network connection.")
            print(f"Error details: {e}")
            sys.exit(1)


        # DINOv2 output dimension map (CLS token)
        self.out_dim_map = {
            "dinov2_vits14": 384,
            "dinov2_vitb14": 768,
            "dinov2_vitl14": 1024,
            "dinov2_vitg14": 1536,
        }
        self.out_dim = self.out_dim_map[arch]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass, returns L2-normalized CLS token.

        DINOv2 backbone returns the CLS token feature by default.
        """
        feats = self.backbone(x)  # (N, D)
        feats = F.normalize(feats, p=2, dim=1)  # L2 normalize
        return feats


class MegaLocEmbeddings(nn.Module):
    """MegaLoc wrapper for feature extraction.

    Loads the SOTA VPR model from torch.hub and L2-normalizes its output.

    Attributes:
        arch (str): The model architecture (just "megaloc").
        backbone (nn.Module): The MegaLoc model.
        out_dim (int): The output feature dimension (fixed at 512).
    """
    def __init__(self, arch: str = "megaloc"):
        super().__init__()
        if arch != "megaloc":
            raise ValueError("This class only supports 'megaloc'.")

        self.arch = arch
        print("Loading 'MegaLoc' model from torch.hub 'gmberton/MegaLoc'...")
        try:
            self.backbone = torch.hub.load(
                'gmberton/MegaLoc', 'get_trained_model'
            )
        except Exception as e:
            print(f"ERROR: Failed to load MegaLoc from torch.hub. Check network connection.")
            print(f"Error details: {e}")
            sys.exit(1)
        
        # MegaLoc paper specifies a 512-dimensional output
        self.out_dim = 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass, returns L2-normalized descriptor."""
        feats = self.backbone(x)  # (N, D)
        feats = F.normalize(feats, p=2, dim=1)  # L2 normalize
        return feats


def build_encoder(arch: str = "megaloc", pretrained: bool = True) -> nn.Module:
    """
    Factory function to build the raw nn.Module encoder.

    Args:
        arch: The architecture name (e.g., "resnet18", "megaloc").
        pretrained: Whether to load pretrained weights.

    Returns:
        A torch.nn.Module instance (the encoder).
    """
    arch = arch.lower()

    if arch in ["resnet18", "resnet50"]:
        print(f"Building torchvision model: {arch}")
        return ResNetEmbeddings(arch=arch, pretrained=pretrained)

    elif arch.startswith("dinov2_"):
        print(f"Building DINOv2 model: {arch}")
        if not pretrained:
            print(
                "WARN: DINOv2 is always pretrained. '--no_pretrained' is ignored.",
                file=sys.stderr
            )
        return DINOv2Embeddings(arch=arch)

    elif arch == "megaloc":
        print(f"Building MegaLoc SOTA VPR model: {arch}")
        if not pretrained:
            print(
                "WARN: MegaLoc is always pretrained. '--no_pretrained' is ignored.",
                file=sys.stderr
            )
        return MegaLocEmbeddings(arch=arch)

    else:
        raise ValueError(f"Unknown architecture: {arch}")


def build_transform(
    img_w: int, img_h: int, arch: str = "megaloc"
) -> transforms.Compose:
    """Builds the image preprocessing pipeline based on the model arch.

    Args:
        img_w: Target image width.
        img_h: Target image height.
        arch: The model architecture (to select correct normalization).

    Returns:
        torchvision.transforms.Compose: The transformation pipeline.
    """
    arch = arch.lower()
    
    # All models listed (ResNet, DINOv2, MegaLoc) use standard ImageNet stats.
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    if arch.startswith("dinov2_"):
        print("Using ImageNet normalization constants for DINOv2.")
        if img_w < 224 or img_h < 224:
            print(
                f"WARN: DINOv2 performs best at 224x224 or larger, "
                f"but running at {img_w}x{img_h}.",
                file=sys.stderr
            )
    elif arch == "megaloc":
        print("Using ImageNet normalization constants for MegaLoc.")
        # MegaLoc paper suggests training at 320x320
        if (img_w, img_h) != (320, 320):
             print(
                f"INFO: MegaLoc was trained at 320x320, "
                f"but running at {img_w}x{img_h}.",
                file=sys.stderr
            )
    else:
        # Default (ResNet)
        print("Using ImageNet normalization constants for ResNet.")

    return transforms.Compose([
        transforms.Resize((img_h, img_w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


# ----------------------------------------------------------------------------
#                NEW VPR ENCODER API (for external import)
# ----------------------------------------------------------------------------

class VPRModel:
    """
    Abstract base interface for a VPR (Visual Place Recognition) model.

    Expected attributes:
      - model (nn.Module): The underlying torch model.
      - transform (transforms.Compose): The preprocessing transform.
      - device (torch.device): The device the model resides on.
      - out_dim (int): The output dimension of the feature vector.

    Expected methods:
      - encode(image: Image.Image) -> np.ndarray: Encodes a single PIL image.
    """
    def __init__(self):
        self.model: Optional[nn.Module] = None
        self.transform: Optional[transforms.Compose] = None
        self.device: Optional[torch.device] = None
        self.out_dim: int = 0

    def encode(self, image: Image.Image) -> np.ndarray:
        """Encodes a single PIL image to an L2-normalized 1D NumPy vector."""
        raise NotImplementedError("Must be implemented by a subclass.")


class GenericVPREncoder(VPRModel):
    """
    Concrete implementation of a VPR model.

    Wraps the model, transform, and device, and provides a simple .encode()
    method for processing single PIL images.
    """
    def __init__(
        self,
        model: nn.Module,
        transform: transforms.Compose,
        device: torch.device
    ):
        """
        Constructor.
        (Note: External users should use the `load_vpr_encoder()` factory
         function to create an instance.)
        """
        super().__init__()
        self.model = model.to(device).eval()
        self.transform = transform
        self.device = device

        # Automatically get output dim from the nn.Module wrapper
        if hasattr(model, 'out_dim'):
            self.out_dim = model.out_dim
        else:
            print(
                "WARN: Model does not have 'out_dim' attribute. "
                "Attempting to infer... This may be slow or fail.",
                file=sys.stderr
            )
            try:
                # Try to infer by passing a dummy tensor
                h, w = 224, 224 # Use a standard size
                if hasattr(transform, 'transforms'):
                    for t in transform.transforms:
                        if isinstance(t, transforms.Resize):
                            size = t.size
                            if isinstance(size, int):
                                h, w = size, size
                            else:
                                h, w = size[0], size[1] # (H, W)
                            break
                dummy_img = torch.randn(1, 3, h, w).to(device)
                self.out_dim = model(dummy_img).shape[1]
                print(f"Inferred out_dim: {self.out_dim}")
            except Exception as e:
                print(f"FATAL: Failed to infer out_dim: {e}", file=sys.stderr)
                self.out_dim = -1 # Indicate failure

    @torch.no_grad()
    def encode(self, image: Image.Image) -> np.ndarray:
        """
        Encodes a single PIL image to an L2-normalized 1D NumPy vector.

        Args:
            image: The input PIL.Image.Image object.

        Returns:
            np.ndarray: A 1D float32 feature vector (L2-normalized).
        """
        # 1. Preprocess (ensure RGB, transform)
        img_rgb = image.convert("RGB")
        tensor = self.transform(img_rgb)
        
        # 2. Add batch dim and move to device
        tensor = tensor.unsqueeze(0).to(self.device, non_blocking=True)
        
        # 3. Inference (our nn.Module wrappers all include L2 norm)
        feat = self.model(tensor)
        
        # 4. Remove batch dim, move to CPU, convert to numpy
        return feat.squeeze(0).cpu().numpy().astype("float32")


def load_vpr_encoder(
    arch: str = "megaloc",
    img_size: Tuple[int, int] = (320, 320),
    pretrained: bool = True,
    device_str: str = "auto"
) -> GenericVPREncoder:
    """
    High-level factory function: builds and loads a complete VPR encoder.
    
    **This is the main entry point for other scripts to import and call.**

    Example:
      >>> from encode_images import load_vpr_encoder
      >>> from PIL import Image
      >>>
      >>> encoder = load_vpr_encoder("megaloc", (320, 320))
      >>> img = Image.open("my_image.jpg")
      >>> feature_vector = encoder.encode(img)
      >>> print(feature_vector.shape)
      (512,)

    Args:
        arch: Architecture name (e.g. "resnet18", "dinov2_vits14", "megaloc").
        img_size: Target (W, H) image size tuple.
        pretrained: Whether to load pretrained weights.
        device_str: "auto", "cpu", or "cuda".

    Returns:
        GenericVPREncoder: The VPR encoder wrapper class with .encode() method.
    """
    # 1. Set device
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    
    # 2. Build nn.Module (raw model)
    model = build_encoder(arch=arch, pretrained=pretrained)
    
    # 3. Build Transform
    tfm = build_transform(img_size[0], img_size[1], arch)
    
    # 4. Create and return the wrapper
    print(f"Loading VPR Encoder '{arch}' to device: {device}")
    return GenericVPREncoder(model, tfm, device)


# ----------------------------------------------------------------------------
#           SCRIPT EXECUTION LOGIC (for batch processing)
# ----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for the encoding script.

    Returns:
        argparse.Namespace: The populated namespace with arguments.
    """
    ap = argparse.ArgumentParser(
        description="Offline Image Encoder - Encodes a directory of images."
    )

    # --- Input Configuration ---
    ap.add_argument(
        "--img_dir",
        type=str,
        default="data",
        help="Directory containing images (searched recursively). Default: ./data"
    )

    # --- Output Configuration ---
    default_tmp_dir = Path("./tmp")
    default_feat_name = "data_features.npy"
    default_paths_name = "data_paths.txt"

    ap.add_argument(
        "--out_features",
        type=str,
        default=str(default_tmp_dir / default_feat_name),
        help=f"Path to save feature file. Default: {default_tmp_dir / default_feat_name}"
    )
    ap.add_argument(
        "--out_paths",
        type=str,
        default=str(default_tmp_dir / default_paths_name),
        help=f"Path to save image paths file. Default: {default_tmp_dir / default_paths_name}"
    )

    # --- Model & Encoding Configuration ---
    ap.add_argument(
        "--arch",
        type=str,
        default="megaloc",  # <-- Default changed to megaloc
        choices=[
            "resnet18", "resnet50",
            "dinov2_vits14", "dinov2_vitb14",
            "megaloc"
        ],
        help="Backbone architecture. Default: megaloc"
    )
    ap.add_argument(
        "--no_pretrained",
        action="store_true",
        help="Use randomly initialized weights (if supported by the model)."
    )
    ap.add_argument(
        "--img_size",
        type=int,
        nargs=2,
        default=[320, 320],  # <-- Default changed to match megaloc
        metavar=("W", "H"),
        help="Resize each image to (width, height). Default: 320 320"
    )
    ap.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for encoding. Reduce if OOM (out of memory). Default: 128"
    )
    ap.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use. 'auto' prefers GPU if available. Default: auto"
    )
    ap.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of parallel workers for data loading. Default: 4"
    )

    args = ap.parse_args()

    # Ensure output directories exist before starting.
    Path(args.out_features).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_paths).parent.mkdir(parents=True, exist_ok=True)

    return args


def find_images(img_dir: str) -> List[str]:
    """Recursively finds all supported images in a directory.

    Args:
        img_dir: The root directory to search.

    Returns:
        A sorted list of string paths to found images.
    """
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    print(f"Searching for images in: {img_dir}...")
    
    root_path = Path(img_dir)
    if not root_path.is_dir():
        print(f"ERROR: Image directory not found: {img_dir}", file=sys.stderr)
        return []
        
    paths = []
    for ext in exts:
        paths.extend(root_path.rglob(f"*{ext}"))
        paths.extend(root_path.rglob(f"*{ext.upper()}"))

    # Convert to string, remove duplicates, and sort
    paths = sorted(list(set(str(p) for p in paths)))
    return paths


# --- NEW: DataLoader Components ---

class ImageFileDataset(Dataset):
    """
    A robust Dataset for loading images from a list of paths.
    Skips unreadable/corrupt images.
    """
    def __init__(self, paths: List[str], transform: Callable):
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[Optional[torch.Tensor], Optional[str]]:
        """Returns (tensor, path) or (None, None) if loading fails."""
        path = self.paths[idx]
        try:
            im = Image.open(path).convert("RGB")
            tensor = self.transform(im)
            return tensor, path
        except (IOError, UnidentifiedImageError, OSError, TimeoutError) as e:
            # Robustly skip unreadable or corrupt files
            print(
                f"[WARN] Skipping image: {path}. Error: {e}",
                file=sys.stderr
            )
            return None, None

def collate_filter_none(
    batch: List[Tuple[Optional[torch.Tensor], Optional[str]]]
) -> Optional[Tuple[torch.Tensor, List[str]]]:
    """
    Custom collate_fn that filters out (None, None) items
    from a batch that failed to load in the Dataset.
    """
    # Filter out None entries
    batch_filtered = [(t, p) for t, p in batch if t is not None]

    if not batch_filtered:
        return None  # This entire batch was corrupt

    # Unzip the batch
    tensors, paths = zip(*batch_filtered)
    
    # Stack tensors and return
    return torch.stack(tensors, dim=0), list(paths)

# --- END: DataLoader Components ---


def run_batch_encoding(
    paths: List[str],
    vpr_encoder: GenericVPREncoder,
    batch_size: int = 128,
    num_workers: int = 4
) -> Tuple[np.ndarray, List[str]]:
    """
    Encodes a list of image paths into feature vectors using an efficient
    DataLoader for parallel I/O.

    Args:
        paths: A list of string paths to images.
        vpr_encoder: The initialized GenericVPREncoder object.
        batch_size: Number of images to process at once.
        num_workers: Number of parallel processes for data loading.

    Returns:
        A tuple (features, successful_paths):
        - features (np.ndarray): A float32 array of shape [K, D] where K
          is the number of successfully encoded images.
        - successful_paths (List[str]): A list of K string paths
          corresponding to the rows in 'features'.
    """
    all_feats = []
    successful_paths = []
    
    # Use components from the VPR encoder object
    model = vpr_encoder.model
    device = vpr_encoder.device
    transform = vpr_encoder.transform

    # Set up Dataset and DataLoader
    dataset = ImageFileDataset(paths, transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_filter_none,
        pin_memory=True,
        drop_last=False
    )

    print(f"Encoding {len(paths)} images with {num_workers} workers...")

    # Set up progress bar if tqdm is available
    if TQDM_AVAILABLE:
        iterable = tqdm(
            loader,
            desc="[Encode]",
            unit="batch",
            total=len(loader)
        )
    else:
        iterable = loader
    
    # Run inference with @torch.no_grad()
    with torch.no_grad():
        for i, batch_data in enumerate(iterable):
            
            if batch_data is None:
                # Batch was empty or all images in it were corrupt
                continue

            tensors, paths_in_batch = batch_data
            
            x = tensors.to(device, non_blocking=True)
            z = model(x)

            # Move to CPU and store
            all_feats.append(z.cpu().numpy().astype("float32"))
            successful_paths.extend(paths_in_batch)

            if not TQDM_AVAILABLE and (i % 20 == 0):
                print(f"[encode] Processed batch {i}/{len(loader)}")


    if not all_feats:
        print(
            "ERROR: No features were produced. Check image directory and paths.",
            file=sys.stderr
        )
        return np.array([]), []

    # Combine all batch features into one large array
    features = np.concatenate(all_feats, axis=0)

    return features, successful_paths


def main():
    """Main execution function (for batch script)."""
    args = parse_args()

    # --- 1. Find Images ---
    paths = find_images(args.img_dir)
    if not paths:
        print(f"No images found in directory: {args.img_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(paths)} images to process.")

    # --- 2. Build Model & Transform (using the new high-level API) ---
    W, H = args.img_size
    print(
        f"Building encoder: {args.arch} (Pretrained: {not args.no_pretrained})"
    )
    
    # Use the high-level factory function to load the VPR encoder wrapper
    try:
        vpr_encoder = load_vpr_encoder(
            arch=args.arch,
            img_size=(W, H),
            pretrained=(not args.no_pretrained),
            device_str=args.device
        )
    except Exception as e:
        print(f"\n--- FATAL ERROR during model loading ---", file=sys.stderr)
        print(f"{e}", file=sys.stderr)
        sys.exit(1)


    print(f"Built model. Output dimension: {vpr_encoder.out_dim}")

    # --- 3. Encode Images (using the efficient batch function) ---
    print(f"Encoding images at size {W}x{H} with batch size {args.batch_size}...")
    t0 = time.time()

    # Pass the VPR object directly to the new batch encoding function
    features, successful_paths = run_batch_encoding(
        paths,
        vpr_encoder, # <-- Pass the whole object
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    dt = time.time() - t0
    
    # --- 4. Handle Results ---
    if len(successful_paths) == 0:
        print("\n--- Encoding FAILED ---")
        print("No images were successfully encoded. Check logs for errors.")
        sys.exit(1)

    num_skipped = len(paths) - len(successful_paths)

    print("\n--- Encoding Complete ---")
    print(
        f"Successfully encoded {features.shape[0]} images (skipped {num_skipped})"
    )
    print(f"Feature dimension: {features.shape[1]}")
    print(f"Total time: {dt:.2f}s")
    print(f"Avg speed: {(features.shape[0] / max(dt, 1e-3)):.1f} img/s")

    # --- 5. Save Results ---
    np.save(args.out_features, features)

    # Save the list of *successfully* encoded paths
    with open(args.out_paths, "w", encoding="utf-8") as f:
        for p in successful_paths:
            f.write(str(p) + "\n")

    print(f"\nSaved features to: {args.out_features}")
    print(f"Saved paths to:   {args.out_paths}")


if __name__ == "__main__":
    main()
