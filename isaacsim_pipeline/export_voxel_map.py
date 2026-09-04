import os
import argparse
import numpy as np

# --- 解析命令行参数 ---
parser = argparse.ArgumentParser(description='生成3D占用网格从USD场景')
parser.add_argument('--usd_path', type=str, required=True,
                    help='USD场景文件的路径')
parser.add_argument('--z_low', type=float, default=0.2,
                    help='扫描的最低高度 (默认: 0.2)')
parser.add_argument('--z_high', type=float, default=1.2,
                    help='扫描的最高高度 (默认: 1.2)')
parser.add_argument('--cell_size', type=float, default=0.05,
                    help='网格单元大小 (默认: 0.05米)')
parser.add_argument('--margin', type=float, default=25.0,
                    help='场景边界扩展边距 (默认: 25.0米)')
parser.add_argument('--output_name', type=str, default='',
                    help='输出文件名后缀 (例如: taoyuan，将生成 occupancy_3d_taoyuan.npy)')
parser.add_argument('--output_dir', type=str, default='outputs/occupancy_data',
                    help='输出目录 (默认: outputs/occupancy_data)')

args = parser.parse_args()

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True, "renderer": "RayTracedLighting"})

# --- 打开 USD 场景 ---
from isaacsim.core.utils.stage import open_stage, is_stage_loading

print(f"正在加载场景: {args.usd_path}")
open_stage(args.usd_path)

import omni
from isaacsim.core.utils.extensions import enable_extension

enable_extension("isaacsim.asset.gen.omap")

# --- 等待场景加载完成 ---
while is_stage_loading():
    simulation_app.update()

ctx = omni.usd.get_context()
stage = ctx.get_stage()

# --- 确保 DefaultPrim 存在 ---
from pxr import Usd, UsdGeom, Gf


def print_prim_hierarchy(prim, indent=""):
    print(f"{indent}{prim.GetPath()}")
    for child in prim.GetChildren():
        print_prim_hierarchy(child, indent + "  ")


if not stage.HasDefaultPrim():
    world = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(world)

root_prim = stage.GetDefaultPrim() if stage.HasDefaultPrim() else stage.GetPseudoRoot()
root_path = str(root_prim.GetPath())


# --- Create PhysicsScene before initializing the occupancy-map generator. ---
from omni.physx.scripts import physicsUtils

scene_path = f"{root_path}/physicsScene"
if stage.GetPrimAtPath(scene_path) is None:
    physicsUtils.add_physics_scene(stage, scene_path)

# Optional ground plane if a scene has no collision body.
# physicsUtils.set_or_add_ground_plane(stage, f"{root_path}/Ground", "Z", 0.0)

# --- Advance a few frames so PhysX can finish preparing collision data. ---
for _ in range(10):
    simulation_app.update()


def compute_valid_bbox(stage, root_prim):
    """计算场景的有效包围盒，过滤异常值"""
    valid_min = np.array([float("inf")] * 3)
    valid_max = np.array([float("-inf")] * 3)

    purposes = [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), purposes)

    prim_count = 0
    invalid_count = 0

    # 遍历所有可绘制的 Prim
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Gprim):  # Gprim = 几何体基类
            continue

        # 计算单个 Prim 的包围盒
        try:
            bbox = cache.ComputeWorldBound(prim)
            abox = bbox.ComputeAlignedRange()
            prim_min = np.array([abox.GetMin()[0], abox.GetMin()[1], abox.GetMin()[2]])
            prim_max = np.array([abox.GetMax()[0], abox.GetMax()[1], abox.GetMax()[2]])

            # 过滤异常值
            # 1. 检查是否有效（非 NaN/Inf）
            if not (np.isfinite(prim_min).all() and np.isfinite(prim_max).all()):
                print(f"⚠ 跳过无效包围盒: {prim.GetPath()}")
                invalid_count += 1
                continue

            # 2. 检查包围盒大小是否合理（单个物体不应超过 50m）
            size = prim_max - prim_min
            if np.any(size > 100.0):
                print(f"⚠ 跳过异常大的物体: {prim.GetPath()}, 尺寸: {size}")
                invalid_count += 1
                continue

            # 3. 检查是否在合理范围内（假设场景在 ±200m 内）
            if np.any(np.abs(prim_min) > 200) or np.any(np.abs(prim_max) > 200):
                print(f"⚠ 跳过远离原点的物体: {prim.GetPath()}")
                invalid_count += 1
                continue

            # 更新全局包围盒
            valid_min = np.minimum(valid_min, prim_min)
            valid_max = np.maximum(valid_max, prim_max)
            prim_count += 1

        except Exception as e:
            print(f"⚠ 计算包围盒出错: {prim.GetPath()}, 错误: {e}")
            invalid_count += 1
            continue

    print(f"\n✓ 有效几何体: {prim_count}, ✗ 无效几何体: {invalid_count}")

    # Return a bounded default when no valid geometry is available.
    if prim_count == 0 or not np.isfinite(valid_min).all():
        print("⚠ 未找到有效几何体，使用默认范围")
        return (0, 0, 0), (100, 100, 10)

    return tuple(valid_min), tuple(valid_max)


bb_min, bb_max = compute_valid_bbox(stage, root_prim)
print(f"过滤后的场景范围: {bb_min} 到 {bb_max}")


# --- Use a bounded default if the scene extent is invalid. ---
def _finite(v):
    return np.isfinite(v).all()


if (
    (not _finite(bb_min))
    or (not _finite(bb_max))
    or (bb_max[0] <= bb_min[0])
    or (bb_max[1] <= bb_min[1])
):
    extent = 15.0
    bb_min = (-extent, -extent, 0.0)
    bb_max = (extent, extent, 2.0)

# --- 配置 OMap 生成器 ---
from isaacsim.asset.gen.omap.bindings import _omap

physx = omni.physx.get_physx_interface()
physx.start_simulation()
stage_id = ctx.get_stage_id()

gen = _omap.Generator(physx, stage_id)
gen.update_settings(args.cell_size, 100, 0, -1)  # cell size (m)  # occupied  # free  # unknown

z_low, z_high = args.z_low, args.z_high
margin = args.margin
xmin = bb_min[0] - margin
ymin = bb_min[1] - margin
xmax = bb_max[0] + margin
ymax = bb_max[1] + margin
print(
    f"最终扫描范围: X[{xmin:.2f}, {xmax:.2f}], Y[{ymin:.2f}, {ymax:.2f}], Z[{z_low:.2f}, {z_high:.2f}]"
)


# Advance one frame so transforms and physics state are stable.
simulation_app.update()

# --- 生成体素并保存 ---
cell = args.cell_size

slices = []
# 计算层数；用 round 避免浮点误差
nz = int(np.ceil((z_high - z_low) / cell - 1e-9))

for k in range(nz):
    sl_low = z_low + k * cell
    sl_high = min(sl_low + cell, z_high)
    origin_z = 0.5 * (sl_low + sl_high)

    # Update the z window while keeping the computed XY scan extent.
    origin = ((xmin + xmax) * 0.5, (ymin + ymax) * 0.5, origin_z)
    gen.set_transform(origin, (xmin, ymin, sl_low), (xmax, ymax, sl_high))
    simulation_app.update()


    gen.generate2d()
    dims = gen.get_dimensions()
    if len(dims) >= 2:
        nx, ny = int(dims[0]), int(dims[1])
    else:
        raise RuntimeError(f"Unexpected dims: {dims}")

    buf2d = np.asarray(gen.get_buffer(), dtype=np.int8)

    # Accept both common buffer orderings returned by the generator.
    if buf2d.size == nx * ny:
        sl = buf2d.reshape(ny, nx)  # (y, x)
    elif buf2d.size == ny * nx:
        sl = buf2d.reshape(nx, ny).T  # 转置到 (y, x)
    else:
        raise RuntimeError(f"Buffer size {buf2d.size} doesn't match dims {nx}x{ny}")

    slices.append(sl)


buffer3d = np.stack(slices, axis=0)  # (z, y, x)

output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

# 构建文件名：occupancy_3d 或 occupancy_3d_<suffix>
if args.output_name:
    filename = f"occupancy_3d_{args.output_name}"
else:
    filename = "occupancy_3d"

metadata = {
    "xmin": xmin,
    "xmax": xmax,
    "ymin": ymin,
    "ymax": ymax,
    "zlow": z_low,
    "zhigh": z_high,
    "cell_size": cell,
    "shape": buffer3d.shape,
    "bb_min_original": bb_min,
    "bb_max_original": bb_max,
}

output_file = os.path.join(output_dir, f"{filename}.npy")
metadata_file = os.path.join(output_dir, f"{filename}_metadata.npy")

np.save(output_file, buffer3d)
np.save(metadata_file, metadata)

print(f"3-D 占用网格 shape (z,y,x): {buffer3d.shape}")
print(f"已保存到: {output_file}")
print(f"元数据已保存到: {metadata_file}")

simulation_app.close()
