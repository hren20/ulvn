from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image as PILImage
from torchvision import transforms
import torchvision.transforms.functional as TF

from diffusers import DDIMScheduler, DDPMScheduler

from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from vint_train.models.nomad.nomad import DenseNetwork, NoMaD
from vint_train.models.nomad.nomad_vint import NoMaD_ViNT, replace_bn_with_gn
from vint_train.models.vint.vint import ViNT
from .common import to_numpy


IMAGE_ASPECT_RATIO = 4.0 / 3.0
ACTION_STATS = {
    "min": np.array([-2.5, -4.0], dtype=np.float32),
    "max": np.array([5.0, 4.0], dtype=np.float32),
}


def transform_images(
    pil_imgs: Union[PILImage.Image, List[PILImage.Image]],
    image_size: List[int],
    center_crop: bool = False,
) -> torch.Tensor:
    """Transform PIL images into the channel-concatenated ViNT/NoMaD tensor."""

    if not isinstance(pil_imgs, list):
        pil_imgs = [pil_imgs]
    transform_type = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensors = []
    for pil_img in pil_imgs:
        pil_img = pil_img.convert("RGB")
        w, h = pil_img.size
        if center_crop:
            if w > h:
                pil_img = TF.center_crop(pil_img, (h, int(h * IMAGE_ASPECT_RATIO)))
            else:
                pil_img = TF.center_crop(pil_img, (int(w / IMAGE_ASPECT_RATIO), w))
        pil_img = pil_img.resize(tuple(image_size))
        tensors.append(transform_type(pil_img).unsqueeze(0))
    return torch.cat(tensors, dim=1)


def _load_image(item: Union[str, PILImage.Image]) -> PILImage.Image:
    if isinstance(item, str):
        return PILImage.open(item).convert("RGB")
    return item.convert("RGB")


def _extract_state_dict(checkpoint: Any) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, torch.nn.Module):
        return checkpoint.state_dict()
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "ema_model", "ema"):
            value = checkpoint.get(key)
            if isinstance(value, torch.nn.Module):
                return value.state_dict()
            if isinstance(value, dict):
                if key == "ema" and "averaged_model" in value and isinstance(value["averaged_model"], dict):
                    return value["averaged_model"]
                return value
        return checkpoint
    raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)!r}")


def _strip_state_dict_prefixes(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned = {}
    for key, value in state_dict.items():
        for prefix in ("module.", "model.", "averaged_model."):
            if key.startswith(prefix):
                key = key[len(prefix):]
        cleaned[key] = value
    return cleaned


class BaseInferenceTrainer(ABC):
    def __init__(self, config: Dict[str, Any], checkpoint_path: str):
        self.config = dict(config)
        self.device = self.config.get("device") or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config["device"] = self.device
        self.model = self._create_model().to(self.device)
        self._load_checkpoint(checkpoint_path)
        self.model.eval()

    @abstractmethod
    def _create_model(self) -> torch.nn.Module:
        pass

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = _strip_state_dict_prefixes(_extract_state_dict(checkpoint))
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        self.checkpoint_missing_keys = list(missing)
        self.checkpoint_unexpected_keys = list(unexpected)
        if missing:
            preview = ", ".join(list(missing)[:8])
            print(f"[{self.__class__.__name__}] Missing checkpoint keys: {len(missing)} ({preview})")
        if unexpected:
            preview = ", ".join(list(unexpected)[:8])
            print(f"[{self.__class__.__name__}] Unexpected checkpoint keys: {len(unexpected)} ({preview})")
        print(f"[{self.__class__.__name__}] Loaded checkpoint: {checkpoint_path}")

    def prepare_inputs(self, image_context: List[Union[str, PILImage.Image]]) -> torch.Tensor:
        images = [_load_image(item) for item in image_context]
        tensor = transform_images(images, self.config["image_size"], center_crop=False)
        return tensor.to(self.device)

    @abstractmethod
    def action_inference(self, obs_images: torch.Tensor, goal_images: Optional[torch.Tensor] = None, num_samples: int = 8):
        pass


class InferenceViNTTrainer(BaseInferenceTrainer):
    """ViNT inference wrapper returning waypoint actions."""

    def _create_model(self) -> torch.nn.Module:
        required = [
            "image_size",
            "context_size",
            "len_traj_pred",
            "learn_angle",
            "obs_encoder",
            "obs_encoding_size",
            "late_fusion",
            "mha_num_attention_heads",
            "mha_num_attention_layers",
            "mha_ff_dim_factor",
        ]
        missing = [key for key in required if key not in self.config]
        if missing:
            raise KeyError(f"ViNT config missing required keys: {missing}")
        return ViNT(
            context_size=self.config["context_size"],
            len_traj_pred=self.config["len_traj_pred"],
            learn_angle=self.config["learn_angle"],
            obs_encoder=self.config["obs_encoder"],
            obs_encoding_size=self.config["obs_encoding_size"],
            late_fusion=self.config["late_fusion"],
            mha_num_attention_heads=self.config["mha_num_attention_heads"],
            mha_num_attention_layers=self.config["mha_num_attention_layers"],
            mha_ff_dim_factor=self.config["mha_ff_dim_factor"],
        )

    @torch.no_grad()
    def action_inference(self, obs_images: torch.Tensor, goal_images: Optional[torch.Tensor] = None, num_samples: int = 8):
        if goal_images is None:
            goal_images = obs_images[:, -3:, :, :]
        _distances, actions = self.model(obs_images, goal_images)
        actions = to_numpy(actions)
        if self.config.get("learn_angle", False):
            actions = actions[:, :, :2]
        return actions


class InferenceNoMaDTrainer(BaseInferenceTrainer):
    """NoMaD diffusion-policy inference wrapper returning waypoint samples."""

    def __init__(self, config: Dict[str, Any], checkpoint_path: str):
        self.noise_scheduler = None
        super().__init__(config, checkpoint_path)
        self.noise_scheduler = self._create_noise_scheduler()
        self.scheduler_type = self.config.get("scheduler_type", "ddpm")
        self.num_inference_steps = int(self.config.get("num_inference_steps", self.config.get("num_diffusion_iters", 10)))
        if self.scheduler_type == "ddim":
            self.noise_scheduler = DDIMScheduler.from_config(
                self.noise_scheduler.config,
                prediction_type="epsilon",
            )
        self.noise_scheduler.set_timesteps(self.num_inference_steps)

    def _create_model(self) -> torch.nn.Module:
        required = [
            "image_size",
            "len_traj_pred",
            "encoding_size",
            "context_size",
            "mha_num_attention_heads",
            "mha_num_attention_layers",
            "mha_ff_dim_factor",
            "down_dims",
            "cond_predict_scale",
        ]
        missing = [key for key in required if key not in self.config]
        if missing:
            raise KeyError(f"NoMaD config missing required keys: {missing}")

        vision_encoder_type = self.config.get("vision_encoder", "nomad_vint")
        if vision_encoder_type != "nomad_vint":
            raise ValueError("This release supports NoMaD vision_encoder='nomad_vint'.")

        vision_encoder = NoMaD_ViNT(
            obs_encoding_size=self.config["encoding_size"],
            context_size=self.config["context_size"],
            mha_num_attention_heads=self.config["mha_num_attention_heads"],
            mha_num_attention_layers=self.config["mha_num_attention_layers"],
            mha_ff_dim_factor=self.config["mha_ff_dim_factor"],
        )
        vision_encoder = replace_bn_with_gn(vision_encoder)
        noise_pred_net = ConditionalUnet1D(
            input_dim=2,
            global_cond_dim=self.config["encoding_size"],
            down_dims=self.config["down_dims"],
            cond_predict_scale=self.config["cond_predict_scale"],
        )
        dist_pred_net = DenseNetwork(embedding_dim=self.config["encoding_size"])
        return NoMaD(vision_encoder=vision_encoder, noise_pred_net=noise_pred_net, dist_pred_net=dist_pred_net)

    def _create_noise_scheduler(self):
        return DDPMScheduler(
            num_train_timesteps=int(self.config.get("num_diffusion_iters", 10)),
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )

    @staticmethod
    def _unnormalize_data(ndata: np.ndarray, stats: Dict[str, np.ndarray]) -> np.ndarray:
        data = (ndata + 1.0) / 2.0
        return data * (stats["max"] - stats["min"]) + stats["min"]

    def _diffusion_to_actions(self, diffusion_output: torch.Tensor) -> np.ndarray:
        ndeltas = to_numpy(diffusion_output.reshape(diffusion_output.shape[0], -1, 2))
        deltas = self._unnormalize_data(ndeltas, ACTION_STATS)
        return np.cumsum(deltas, axis=1)

    @torch.no_grad()
    def _sample_actions(self, cond: torch.Tensor, num_samples: int) -> torch.Tensor:
        cond = cond.repeat(num_samples, 1)
        noisy_actions = torch.randn(
            (cond.shape[0], int(self.config["len_traj_pred"]), 2),
            device=self.device,
        )
        for t in self.noise_scheduler.timesteps:
            noise_pred = self.model(
                "noise_pred_net",
                sample=noisy_actions,
                timestep=torch.full((noisy_actions.shape[0],), int(t), device=self.device, dtype=torch.long),
                global_cond=cond,
            )
            noisy_actions = self.noise_scheduler.step(noise_pred, t, noisy_actions).prev_sample
        return noisy_actions

    @torch.no_grad()
    def action_inference(self, obs_images: torch.Tensor, goal_images: Optional[torch.Tensor] = None, num_samples: int = 8):
        if goal_images is None:
            goal_images = obs_images[:, -3:, :, :]
        batch_size = obs_images.shape[0]
        goal_mask = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        obs_cond = self.model(
            "vision_encoder",
            obs_img=obs_images,
            goal_img=goal_images,
            input_goal_mask=goal_mask,
        )
        sampled = self._sample_actions(obs_cond, num_samples=max(1, int(num_samples)))
        return self._diffusion_to_actions(sampled)
