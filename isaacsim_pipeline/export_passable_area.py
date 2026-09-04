import numpy as np
from pathlib import Path
import os
from PIL import Image
import argparse

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='从3D占用网格生成2D可通行区域地图')
    parser.add_argument('--npy_path', type=str, required=True,
                        help='3D占用网格文件的完整路径 (例如: /path/to/occupancy_3d_taoyuan1.npy)')
    parser.add_argument('--z_low', type=float, default=0.2,
                        help='检查高度范围的下界 (默认: 0.2)')
    parser.add_argument('--z_high', type=float, default=1.2,
                        help='检查高度范围的上界 (默认: 1.2)')
    parser.add_argument('--robot_radius', type=float, default=0.3,
                        help='机器人半径(米) (默认: 0.3)')
    parser.add_argument('--output_dir', type=str, default='outputs/passable_area',
                        help='输出目录 (默认: outputs/passable_area)')
    
    return parser.parse_args()


def load_occupancy_data(npy_path):
    """加载3D占用网格数据和元数据"""
    # 检查文件是否存在
    if not os.path.exists(npy_path):
        raise FileNotFoundError(f"找不到占用网格文件: {npy_path}")
    
    # 构造元数据文件路径
    meta_path = npy_path.replace('.npy', '_metadata.npy')
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"找不到元数据文件: {meta_path}")
    
    grid_3d = np.load(npy_path)
    metadata = np.load(meta_path, allow_pickle=True).item()

    # Flip the x axis and materialize a contiguous array for downstream libraries.
    grid_3d = grid_3d[..., ::-1].copy()

    print(f"✓ 加载3D网格: shape={grid_3d.shape}")
    print(f"✓ 网格范围: X[{metadata['xmin']:.2f}, {metadata['xmax']:.2f}], "
          f"Y[{metadata['ymin']:.2f}, {metadata['ymax']:.2f}], "
          f"Z[{metadata['zlow']:.2f}, {metadata['zhigh']:.2f}]")
    print(f"✓ Cell size: {metadata['cell_size']}m")
    
    return grid_3d, metadata


def apply_robot_radius_erosion(traversable_map, robot_radius, cell_size):
    """
    输入:
      traversable_map: 1 = 可通行(白), 0 = 障碍(黑)
    输出:
      eroded_map: 考虑半径后的可通行(1)/障碍(0)
    """
    import numpy as np
    import cv2

    # Convert robot radius from meters to pixels.
    kernel_radius_pixels = int(np.ceil(robot_radius / cell_size))
    kernel_size = 2 * kernel_radius_pixels + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # Treat traversable cells as foreground for the morphology operation.
    obstacle = (traversable_map == 1).astype(np.uint8)
    obstacle_dilated = cv2.erode(obstacle, kernel, iterations=1)

    # 膨胀后的障碍区=1 ⇒ 可通行区=0；取反回到 1=可通行, 0=障碍
    eroded_map = (obstacle_dilated == 0).astype(np.uint8)

    # 统计
    original_free = np.sum(traversable_map)
    eroded_free = np.sum(eroded_map)
    reduction = (original_free - eroded_free) / traversable_map.size * 100
    print(f"\n腐蚀/膨胀参数：半径={robot_radius}m, cell={cell_size}m, kernel={kernel_size}")
    print(f"  膨胀前可通行: {original_free / traversable_map.size * 100:.2f}%")
    print(f"  膨胀后可通行: {eroded_free / traversable_map.size * 100:.2f}%")
    print(f"  可通行减少: {reduction:.2f}%")

    return eroded_map


def compute_2d_traversable_map(
    grid_3d,
    metadata,
    method='any_in_range',
    z_min=0.20,
    z_max=2.00,
    treat_unknown_as_obstacle=False, # 是否将未知区域视为障碍
    robot_base_height_min=0.2,
    robot_base_height_max=0.5
):
    """
    从3D占用网格计算2D可通行区域图。
    Traversability is only computed inside the obstacle bounding box; cells
    outside the box are treated as obstacles.

    :param grid_3d: (nz, ny, nx) 形状的3D网格, 值: 100=occupied, 0=free, -1=unknown
    :return: (ny, nx) 形状的2D地图, 值: 1=可通行, 0=不可通行
    """
    nz, ny, nx = grid_3d.shape

    # Compute the global obstacle bounding box.
    # 将所有z层的障碍物投影到2D平面
    obstacle_mask_2d = np.any(grid_3d == 100, axis=0)
    
    # 获取2D平面上所有障碍物格子的坐标
    obstacle_y_indices, obstacle_x_indices = np.where(obstacle_mask_2d)

    # 边界情况：如果地图中没有任何障碍物
    if obstacle_y_indices.size == 0:
        print("未发现障碍物，整个地图被视为可通行区域。")
        return np.ones((ny, nx), dtype=np.uint8)

    # 计算能包围所有障碍物的最小边界框
    ymin, ymax = obstacle_y_indices.min(), obstacle_y_indices.max()
    xmin, xmax = obstacle_x_indices.min(), obstacle_x_indices.max()
    print(f"检测到障碍物边界框 (y,x): [{ymin}:{ymax+1}, {xmin}:{xmax+1}]")
    # --- 边界框计算结束 ---

    # Compute traversability within the selected height range.
    zlow      = float(metadata['zlow'])
    zhigh     = float(metadata['zhigh'])
    dz        = float(metadata.get('cell_size_z', metadata['cell_size']))
    z_min_clamped = max(z_min, zlow)
    z_max_clamped = min(z_max, zhigh)

    start_idx = int(np.clip(np.floor((z_min_clamped - zlow) / dz), 0, nz))
    end_idx   = int(np.clip(np.ceil((z_max_clamped - zlow) / dz), 0, nz))
    
    if end_idx <= start_idx:
        raise ValueError(f"无效的高度范围: [{z_min}, {z_max}] 不与 [{zlow}, {zhigh}] 相交")

    print(f"检查高度范围: {zlow + start_idx*dz:.2f}m - {zlow + (end_idx-1)*dz:.2f}m")
    print(f"对应层索引: [{start_idx}, {end_idx}) / 总共 {nz} 层")

    slab = grid_3d[start_idx:end_idx, :, :]

    # Compute the robot-base height slice.
    base_z_min_clamped = max(robot_base_height_min, zlow)
    base_z_max_clamped = min(robot_base_height_max, zhigh)
    
    base_start_idx = int(np.clip(np.floor((base_z_min_clamped - zlow) / dz), 0, nz))
    base_end_idx   = int(np.clip(np.ceil((base_z_max_clamped - zlow) / dz), 0, nz))
    
    print(f"机器人底盘高度范围: {base_z_min_clamped:.2f}m - {base_z_max_clamped:.2f}m")
    print(f"对应层索引: [{base_start_idx}, {base_end_idx}) / 总共 {nz} 层")
    
    # 提取机器人底盘高度范围的切片
    base_slab = grid_3d[base_start_idx:base_end_idx, :, :]
    # Compute potential traversability before bounding-box masking.
    if method == 'any_in_range':
        has_occ = np.any(slab == 100, axis=0)
        has_free = np.any(slab == 0, axis=0)

        # Check whether the robot-base height range contains obstacles.
        # 底盘高度范围内有任何障碍物，就不能通行
        base_has_obstacle = np.any(base_slab == 100, axis=0)
        print(f"底盘高度有障碍的位置占比: {np.sum(base_has_obstacle) / base_has_obstacle.size * 100:.2f}%")
        # is_blocked为True代表确定是障碍
        if treat_unknown_as_obstacle:
            has_unk = np.any(slab == -1, axis=0)
            is_blocked = has_occ | (has_unk & ~has_free)
        else:
            # 允许在桌子等高障碍物下通行
            is_blocked = has_occ & ~has_free
        
        # Combine selected-height obstacles with robot-base obstacles.
        is_blocked = is_blocked | base_has_obstacle

        potential_traversable_map = (~is_blocked).astype(np.uint8)

    elif method == 'majority_in_range':
        k = end_idx - start_idx
        occ_cnt = np.sum(slab == 100, axis=0)
        threshold = 0.5
        is_blocked = (occ_cnt > threshold * k)
        if treat_unknown_as_obstacle:
            has_unk = np.any(slab == -1, axis=0)
            is_blocked[has_unk] = True
            
        potential_traversable_map = (~is_blocked).astype(np.uint8)

    elif method == 'ground_level':
        ground_layer = grid_3d[0, :, :]
        is_blocked = (ground_layer == 100)
        if treat_unknown_as_obstacle:
            is_blocked[ground_layer == -1] = True
            
        potential_traversable_map = (~is_blocked).astype(np.uint8)

    else:
        raise ValueError(f"未知方法: {method}")

    # Apply bounding-box masking.
    # 1. 创建一个默认为0（障碍物）的最终地图
    final_traversable_map = np.zeros((ny, nx), dtype=np.uint8)
    
    # 2. 将边界框内的计算结果复制到最终地图中
    final_traversable_map[ymin:ymax+1, xmin:xmax+1] = potential_traversable_map[ymin:ymax+1, xmin:xmax+1]

    print(f"使用方法: {method}, unknown视为障碍: {treat_unknown_as_obstacle}")
    # 打印最终地图的可通行区域占比
    final_traversable_ratio = np.sum(final_traversable_map) / final_traversable_map.size * 100
    print(f"最终可通行区域占比: {final_traversable_ratio:.2f}%")

    return final_traversable_map


def visualize_and_save(traversable_map, metadata, output_path):
    """
    可视化并保存2D可通行区域图
    使用PIL直接保存，确保1像素 = 1体素
    
    traversable_map: (ny, nx) 数组，值为 1(可通行) 或 0(障碍物)
    输出图片尺寸: nx × ny (宽×高)
    """
    
    ny, nx = traversable_map.shape

    # ------------------------ revision ------------------------------
    flipped_map = np.flipud(traversable_map)

    # 创建图像：白色=障碍物(0), 黑色=可通行(1)
    # traversable_map中：1=可通行，0=障碍物
    # 我们要反转：白色(255)=障碍物，黑色(0)=可通行
    img_array = (1 - flipped_map) * 255  # 反转并缩放到0-255
    img_array = img_array.astype(np.uint8)
    
    # 使用PIL创建图像 (mode='L' 表示灰度图)
    img = Image.fromarray(img_array, mode='L')
    
    # 直接保存，确保像素精确对应
    img.save(output_path)
    
    print(f"✓ 图片已保存: {output_path}")
    print(f"  数组 shape: ({ny}, {nx})")
    print(f"  图片尺寸: {nx} × {ny} 像素 (宽×高)")
    print(f"  颜色: 白色=障碍物, 黑色=可通行")
    
    # 验证
    verify_img = Image.open(output_path)
    assert verify_img.size == (nx, ny), f"尺寸不匹配！期望{(nx, ny)}，实际{verify_img.size}"
    print(f"  ✓ 验证通过：1像素 = 1体素")


def extract_keyword_from_filename(npy_path):
    """
    从输入文件路径提取关键词
    例如: /path/to/occupancy_3d_taoyuan1.npy -> taoyuan1
    """
    filename = os.path.basename(npy_path)  # 获取文件名
    filename_without_ext = filename.replace('.npy', '')  # 去掉.npy
    
    # 移除 occupancy_3d_ 前缀
    if filename_without_ext.startswith('occupancy_3d_'):
        keyword = filename_without_ext[len('occupancy_3d_'):]
    else:
        # 如果没有这个前缀，就用整个文件名
        keyword = filename_without_ext
    
    return keyword


def main():
    # 解析命令行参数
    args = parse_args()
    
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract the scene key from the input file name.
    keyword = extract_keyword_from_filename(args.npy_path)
    print(f"从输入文件提取关键词: {keyword}")
    
    # 加载数据
    print("=" * 60)
    print("开始生成2D可通行区域图")
    print("=" * 60)
    print(f"输入文件: {args.npy_path}")
    print(f"高度范围: [{args.z_low}, {args.z_high}] m")
    print(f"机器人半径: {args.robot_radius} m")
    
    grid_3d, metadata = load_occupancy_data(args.npy_path)
    cell_size = metadata['cell_size']
    
    # 生成2D地图（使用 any_in_range 方法）
    method = 'any_in_range'
    print(f"\n{'='*60}")
    traversable_map = compute_2d_traversable_map(
        grid_3d, 
        metadata=metadata,
        method=method,
        z_min=args.z_low,
        z_max=args.z_high,
        robot_base_height_min=0.2,
        robot_base_height_max=0.5
    )

    # 应用机器人半径腐蚀
    eroded_map = apply_robot_radius_erosion(traversable_map, args.robot_radius, cell_size)

    # 生成输出文件名：keyword_max_passable_area.png
    output_filename = f"{keyword}_max_passable_area.png"
    output_path = os.path.join(output_dir, output_filename)
    
    # 保存可视化结果
    visualize_and_save(eroded_map, metadata, output_path)
    
    # 保存numpy数组（也按相同命名规则）
    npy_filename = f"{keyword}_max_passable_area.npy"
    npy_path = os.path.join(output_dir, npy_filename)
    np.save(npy_path, eroded_map)
    print(f"✓ 数组已保存: {npy_path}")
    
    print("\n" + "="*60)
    print("✓ 地图生成完成！")
    print(f"✓ 输出目录: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
