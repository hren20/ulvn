import numpy as np
import time
import argparse
import importlib
import os

plt = None
rcParams = None
Normalize = None
Image = None
label = None
distance_transform_edt = None
skeletonize = None
remove_small_objects = None
cdist = None


def load_runtime_dependencies():
    """Load heavy IsaacSim mapping dependencies after CLI parsing."""
    global plt, rcParams, Normalize, Image, label, distance_transform_edt
    global skeletonize, remove_small_objects, cdist

    import matplotlib.pyplot as _plt
    from matplotlib import rcParams as _rcParams
    from matplotlib.colors import Normalize as _Normalize
    from PIL import Image as _Image
    from scipy.ndimage import label as _label
    from scipy.ndimage import distance_transform_edt as _distance_transform_edt
    from skimage.morphology import skeletonize as _skeletonize, remove_small_objects as _remove_small_objects
    from scipy.spatial.distance import cdist as _cdist

    for module_name in ("extended_mapping.map_processing", "sdg_roadmap.skel_disk_graph_provider"):
        module = importlib.import_module(module_name)
        for name, value in vars(module).items():
            if not name.startswith("_"):
                globals()[name] = value

    plt = _plt
    rcParams = _rcParams
    Normalize = _Normalize
    Image = _Image
    label = _label
    distance_transform_edt = _distance_transform_edt
    skeletonize = _skeletonize
    remove_small_objects = _remove_small_objects
    cdist = _cdist


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='从2D可通行区域地图提取骨架并离散化采样')
    
    parser.add_argument('--map_path', type=str, required=True,
                        help='可通行区域地图的完整路径 (例如: /path/to/taoyuan1_max_passable_area.png)')
    
    parser.add_argument('--metadata_path', type=str, required=True,
                        help='元数据文件的完整路径 (例如: /path/to/occupancy_3d_taoyuan1_metadata.npy)')
    
    parser.add_argument('--spacing', type=float, default=0.5,
                        help='采样点间距(米) (默认: 0.5)')
    
    parser.add_argument('--angle_interval', type=int, choices=[45, 60, 90, 120, 180], default=60,
                        help='方向角度间隔(度)，可选: 45, 60, 90, 120, 180 (默认: 60)')
    parser.add_argument('--output_dir', type=str, default='outputs/skeleton_points',
                        help='输出目录 (默认: outputs/skeleton_points)')
    
    return parser.parse_args()


def extract_keyword_from_map_filename(map_path):
    """
    从地图文件路径提取关键词
    例如: /path/to/taoyuan1_max_passable_area.png -> taoyuan1
    """
    filename = os.path.basename(map_path)  # 获取文件名
    filename_without_ext = os.path.splitext(filename)[0]  # 去掉扩展名
    
    # 移除 _max_passable_area 后缀
    if filename_without_ext.endswith('_max_passable_area'):
        keyword = filename_without_ext[:-len('_max_passable_area')]
    else:
        # 如果没有这个后缀，就用整个文件名
        keyword = filename_without_ext
    
    return keyword


def distanceFigure(map_preproc_config, distance_map, cmap, contour_color=None, contour_lw=2, title=True):
    if title:
        plt.title("Signed Distance Field\nd_offset = {}".format(map_preproc_config['dist_field_offset']))
    distance_map.display(cmap=cmap)
    if contour_color is not None:
        X,Y = distance_map.getNpMeshgrid()
        plt.contour(X, Y, distance_map.data.T, levels=[0], colors=contour_color, linewidths=contour_lw)


def distanceGradFigure(map_preproc_config, distance_gradient_maps):
    # color inds : R:0, G:1, B:2
    x_color_ind = 1
    y_color_ind = 2 #B

    x_color = [.8,.2,.2]
    y_color = [.2,.8,.2]

    grad_x, grad_y = distance_gradient_maps[0].data, distance_gradient_maps[1].data
    grad_array = np.array([grad_x, grad_y])
    xnorm = Normalize(np.min(grad_x), np.max(grad_x))
    ynorm = Normalize(np.min(grad_y), np.max(grad_y))
    x_vals = plt.cm.binary(xnorm(grad_x))
    y_vals = plt.cm.binary(ynorm(grad_y))
    grad_colors = np.zeros(x_vals.shape)
    grad_colors[:,:,:] = 0
    grad_colors[:,:,0] += x_vals[:,:,0]*x_color[0]
    grad_colors[:,:,1] += x_vals[:,:,0]*x_color[1]
    grad_colors[:,:,2] += x_vals[:,:,0]*x_color[2]
    grad_colors[:,:,0] += y_vals[:,:,0]*y_color[0]
    grad_colors[:,:,1] += y_vals[:,:,0]*y_color[1]
    grad_colors[:,:,2] += y_vals[:,:,0]*y_color[2]

    no_grad_ind = np.array(np.where(dist_map.data == 0))
    grad_colors[no_grad_ind[0], no_grad_ind[1]] = 0
    grad_colors[:,:,3] = 1
    
    plt.title("SDF Gradient")
    distance_gradient_maps[0].display()
    plt.imshow(np.flip(grad_colors.transpose(1,0,2), axis=0), extent=distance_gradient_maps[0].getExtent(transpose=False))


def keep_largest_component(binary_map):
    """
    只保留最大的连通分量，删除所有孤立的小骨架
    
    Args:
        binary_map: 布尔类型的二值图像
    
    Returns:
        只包含最大连通分量的二值图像
    """
    # 标记所有连通分量（使用8邻域连通）
    labeled_array, num_features = label(binary_map, structure=np.ones((3, 3)))
    
    if num_features == 0:
        return binary_map
    
    # 计算每个连通分量的大小
    component_sizes = np.bincount(labeled_array.ravel())
    # 排除背景（标签0）
    component_sizes[0] = 0
    
    # 找到最大连通分量的标签
    largest_component_label = component_sizes.argmax()
    
    # 只保留最大的连通分量
    largest_component = (labeled_array == largest_component_label)
    
    print(f"Connected component analysis:")
    print(f"  Total components found: {num_features}")
    print(f"  Largest component size: {component_sizes[largest_component_label]} pixels")
    if num_features > 1:
        print(f"  Removed {num_features - 1} isolated components")
    
    return largest_component


def print_resolution_info(skeleton_map, xmin, xmax, ymin, ymax):
    """打印分辨率相关信息"""
    height, width = skeleton_map.data.shape
    resolution = skeleton_map.resolution
    
    print("\n" + "="*50)
    print("RESOLUTION INFORMATION")
    print("="*50)
    
    print(f"\n【图像信息】")
    print(f"  图像尺寸: {width} × {height} 像素")
    print(f"  分辨率: {resolution} 米/像素")
    print(f"  含义: 图像中1个像素 = 真实世界{resolution}米")
    
    print(f"\n【真实世界尺寸（从图像计算）】")
    real_width = width * resolution
    real_height = height * resolution
    print(f"  宽度: {width} 像素 × {resolution} = {real_width:.2f} 米")
    print(f"  高度: {height} 像素 × {resolution} = {real_height:.2f} 米")
    
    print(f"\n【坐标范围（从配置文件）】")
    x_range = xmax - xmin
    y_range = ymax - ymin
    print(f"  X范围: [{xmin:.2f}, {xmax:.2f}] = {x_range:.2f} 米")
    print(f"  Y范围: [{ymin:.2f}, {ymax:.2f}] = {y_range:.2f} 米")
    
    print(f"\n【匹配检查】")
    x_match = abs(real_width - x_range) < 1.0
    y_match = abs(real_height - y_range) < 1.0
    
    if x_match and y_match:
        print(f"  ✓ 图像尺寸与坐标范围匹配!")
    else:
        print(f"  ✗ 警告: 尺寸不匹配")
        print(f"     X差异: {abs(real_width - x_range):.2f} 米")
        print(f"     Y差异: {abs(real_height - y_range):.2f} 米")
    
    print(f"\n【采样示例】")
    for spacing in [0.5, 1.0, 2.0, 5.0]:
        pixels = spacing / resolution
        print(f"  {spacing}米间距 = {pixels:.1f} 个像素")
    
    print("="*50 + "\n")


def discretize_skeleton_improved(skeleton_map, xmin, ymin, resolution, spacing=0.5, 
                                  min_branch_length=10, use_pruning=True, output_path=None):
    """
    改进的骨架离散化:优先在主干上采样,避免细小分支
    
    Args:
        skeleton_map: 骨架地图对象
        xmin, ymin: 真实世界坐标的最小值
        resolution: 地图分辨率(米/像素)
        spacing: 采样间隔(米)
        min_branch_length: 要修剪的最小分支长度(像素)
        use_pruning: 是否使用分支修剪
        output_path: 保存路径
    
    Returns:
        discretized_points: numpy数组,形状为(N, 2)
    """
    print("\n" + "="*50)
    print("改进的骨架离散化处理")
    print("="*50)
    
    skeleton_data = skeleton_map.data.astype(bool)
    
    # 步骤1: 可选的分支修剪
    if use_pruning:
        print(f"\n【步骤1: 分支修剪】")
        print(f"  原始骨架像素数: {np.sum(skeleton_data)}")
        pruned_skeleton = prune_skeleton_branches(skeleton_data, min_branch_length)
        print(f"  修剪后骨架像素数: {np.sum(pruned_skeleton)}")
        print(f"  移除了 {np.sum(skeleton_data) - np.sum(pruned_skeleton)} 个末端像素")
        working_skeleton = pruned_skeleton
    else:
        working_skeleton = skeleton_data
    
    # 步骤2: 计算中心性得分
    print(f"\n【步骤2: 计算主干中心性】")
    centrality_scores = compute_skeleton_centrality(skeleton_data)
    
    # 获取骨架点及其中心性得分
    skeleton_pixels = np.argwhere(working_skeleton > 0)
    scores = centrality_scores[skeleton_pixels[:, 0], skeleton_pixels[:, 1]]
    
    print(f"  中心性得分范围: [{scores.min():.2f}, {scores.max():.2f}]")
    
    # 步骤3: 转换为真实世界坐标
    print(f"\n【步骤3: 坐标转换】")
    skeleton_coords = np.zeros((len(skeleton_pixels), 2))
    for idx, point in enumerate(skeleton_pixels):
        i, j = point
        skeleton_coords[idx, 0] = xmin + j * resolution  # X
        skeleton_coords[idx, 1] = ymin + i * resolution  # Y
    
    # 步骤4: 基于中心性的优先采样
    print(f"\n【步骤4: 优先采样】")
    pixel_spacing = spacing / resolution
    print(f"  采样间隔: {spacing}米 = {pixel_spacing:.1f}个像素")
    
    # 按中心性得分排序(从高到低)
    sorted_indices = np.argsort(scores)[::-1]
    sorted_coords = skeleton_coords[sorted_indices]
    sorted_scores = scores[sorted_indices]
    
    discretized_points = []
    used_mask = np.zeros(len(sorted_coords), dtype=bool)
    
    # 贪心选择:每次选择得分最高且距离已选点足够远的点
    for idx in range(len(sorted_coords)):
        if used_mask[idx]:
            continue
        
        candidate = sorted_coords[idx]
        
        # 检查是否与已选点距离足够
        if len(discretized_points) == 0:
            discretized_points.append(candidate)
            used_mask[idx] = True
        else:
            distances = np.linalg.norm(
                np.array(discretized_points) - candidate, axis=1
            )
            min_dist = distances.min()
            
            if min_dist >= spacing:
                discretized_points.append(candidate)
                used_mask[idx] = True
                
                # 标记附近的点为已使用
                all_distances = np.linalg.norm(
                    sorted_coords - candidate, axis=1
                )
                used_mask |= (all_distances < spacing * 0.7)
    
    discretized_points = np.array(discretized_points)
    
    # 步骤5: 统计信息
    print(f"\n【离散化结果】")
    print(f"  离散化后点数量: {len(discretized_points)}")
    print(f"  压缩比: {len(skeleton_pixels)}/{len(discretized_points)} = {len(skeleton_pixels)/len(discretized_points):.1f}:1")
    print(f"  X坐标范围: [{discretized_points[:,0].min():.2f}, {discretized_points[:,0].max():.2f}]米")
    print(f"  Y坐标范围: [{discretized_points[:,1].min():.2f}, {discretized_points[:,1].max():.2f}]米")
    
    # 计算实际间距统计
    if len(discretized_points) > 1:
        distances = cdist(discretized_points, discretized_points)
        np.fill_diagonal(distances, np.inf)
        min_distances = distances.min(axis=1)
        print(f"  实际平均间距: {min_distances.mean():.3f}米")
        print(f"  最小间距: {min_distances.min():.3f}米")
        print(f"  最大间距: {min_distances.max():.3f}米")
    
    # 步骤6: 保存
    if output_path is not None:
        np.save(output_path, discretized_points)
        print(f"\n✓ 坐标已保存到: {output_path}")
    
    print("="*50 + "\n")
    
    return discretized_points


def prune_skeleton_branches(skeleton_data, min_branch_length=10):
    """
    修剪骨架上的短分支
    
    Args:
        skeleton_data: 布尔类型的骨架数据
        min_branch_length: 最小分支长度(像素),短于此长度的末端分支将被移除
    
    Returns:
        修剪后的骨架
    """
    from scipy import ndimage
    
    # 复制数据
    pruned = skeleton_data.copy()
    
    # 定义8邻域结构
    struct = np.array([[1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1]])
    
    # 迭代修剪末端点
    iterations = 0
    max_iterations = min_branch_length
    
    for _ in range(max_iterations):
        # 计算每个点的邻居数量
        neighbor_count = ndimage.convolve(pruned.astype(int), struct, mode='constant') - pruned.astype(int)
        
        # 找到末端点(只有1个邻居)
        endpoints = (neighbor_count == 1) & pruned
        
        if not np.any(endpoints):
            break
        
        # 移除末端点
        pruned = pruned & ~endpoints
        iterations += 1
    
    print(f"  修剪了 {iterations} 层末端分支")
    return pruned


def save_pruned_skeleton(skeleton_map, xmin, xmax, ymin, ymax, 
                        min_branch_length=10, save_path=None, save_svg_path=None):
    """
    保存修剪后的骨架图(不显示)，并额外保存为矢量图（SVG格式）

    Args:
        skeleton_map: 原始骨架地图对象
        xmin, xmax, ymin, ymax: 坐标范围
        min_branch_length: 要修剪的最小分支长度(像素)
        save_path: 保存路径
        save_svg_path: 额外保存的SVG文件路径
    
    Returns:
        pruned_skeleton: 修剪后的骨架数据
    """
    print("\n" + "="*50)
    print("Saving Pruned Skeleton")
    print("="*50)
    
    # 执行分支修剪
    skeleton_data = skeleton_map.data.astype(bool)
    print(f"Original skeleton pixels: {np.sum(skeleton_data)}")
    
    pruned_skeleton = prune_skeleton_branches(skeleton_data, min_branch_length)
    print(f"Pruned skeleton pixels: {np.sum(pruned_skeleton)}")
    print(f"Removed: {np.sum(skeleton_data) - np.sum(pruned_skeleton)} pixels")
    
    # 创建可视化(不显示)
    plt.figure(figsize=(6.4, 4.8))
    
    custom_extent = [xmin, xmax, ymin, ymax]
    
    # 显示修剪后的骨架
    plt.imshow(pruned_skeleton,
               cmap=plt.cm.binary, 
               origin='lower', 
               extent=custom_extent)
    
    plt.xlabel('X (meters)', fontsize=12)
    plt.ylabel('Y (meters)', fontsize=12)
    plt.title('Pruned Skeleton', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    plt.axis('equal')
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Pruned skeleton saved to: {save_path}")
    
    # 如果需要保存为SVG图像
    if save_svg_path is not None:
        # 创建新的figure用于保存为SVG，设置画布大小为6.4x4.8英寸
        plt.figure(figsize=(6.4, 4.8))  
        plt.imshow(pruned_skeleton, cmap=plt.cm.binary, origin='lower', extent=custom_extent)

        # plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        plt.axis('equal')
        
        # 设置坐标轴数字的字体大小为14
        plt.tick_params(axis='both', which='major', labelsize=14)

        plt.tight_layout()

        # 保存为SVG
        plt.savefig(save_svg_path, format='svg', bbox_inches='tight')
        print(f"✓ Pruned skeleton saved as SVG to: {save_svg_path}")
        fig = plt.gcf()
        print(fig.get_size_inches()) 
        
        plt.close()  # 关闭图像,不显示

    if save_path is not None:
        # 生成纯骨架图的文件名
        base_path = os.path.splitext(save_path)[0]
        pure_skeleton_path = f"{base_path}_pure.png"
        
        # 将布尔数组转换为uint8图像 (True->255白色, False->0黑色)
        skeleton_image = ((~pruned_skeleton) * 255).astype(np.uint8)
        
        # 使用PIL保存，保持原始分辨率
        img = Image.fromarray(skeleton_image, mode='L')
        img.save(pure_skeleton_path)
        
        print(f"✓ Pure skeleton (no axes) saved to: {pure_skeleton_path}")
        print(f"  Pure skeleton resolution: {skeleton_image.shape[1]}x{skeleton_image.shape[0]}")
    
    print("="*50 + "\n")
    
    return pruned_skeleton



def compute_skeleton_centrality(skeleton_data):
    """
    计算骨架每个点的"中心性"得分,主干部分得分更高
    使用距离变换和局部密度来评估
    
    Args:
        skeleton_data: 布尔类型的骨架数据
    
    Returns:
        centrality_scores: 每个骨架点的中心性得分
    """
    # 方法1: 基于到骨架边界的距离
    # 骨架中心的点距离骨架边界更远
    inverted = ~skeleton_data
    dist_to_boundary = distance_transform_edt(skeleton_data)
    
    # 方法2: 基于局部密度(邻域内骨架点的数量)
    from scipy.ndimage import uniform_filter
    kernel_size = 15  # 可调整
    local_density = uniform_filter(skeleton_data.astype(float), size=kernel_size)
    
    # 综合得分
    centrality = dist_to_boundary * 0.5 + local_density * 100
    
    # 只保留骨架上的点的得分
    centrality = centrality * skeleton_data
    
    return centrality

def compute_directions_at_points(discretized_points, dist_map, grad_maps, 
                                 xmin, ymin, resolution, angle_interval=60,
                                 n_neighbors=20):
    """
    在每个骨架采样点上基于PCA计算切线方向，然后生成均匀分布的方向角度
    
    Args:
        discretized_points: numpy数组,形状为(N, 2),真实世界坐标(米)
        dist_map: 距离场地图对象
        grad_maps: 梯度地图列表 [grad_x_map, grad_y_map]
        xmin, ymin: 真实世界坐标的最小值
        resolution: 地图分辨率(米/像素)
        angle_interval: 角度间隔(度),可选: 45, 60, 90, 120, 180
        n_neighbors: 用于PCA拟合的邻近骨架点数量
    
    Returns:
        directions: numpy数组,形状为(N, num_directions),每个点的方向角度(弧度)
    """
    print("\n" + "="*50)
    print("计算骨架点的方向角度（使用PCA方法）")
    print("="*50)
    
    # 计算方向数量
    num_directions = 360 // angle_interval
    angle_step_rad = np.deg2rad(angle_interval)
    
    print(f"角度间隔: {angle_interval}度")
    print(f"方向数量: {num_directions}")
    print(f"方向类型: 切线方向(沿着走廊) - PCA方法")
    print(f"PCA邻近点数: {n_neighbors}")
    
    N = len(discretized_points)
    directions = np.zeros((N, num_directions))
    
    # 获取骨架地图的形状
    height, width = grad_maps[0].data.shape
    
    # 从距离场构建骨架二值图
    # 假设骨架点是距离场中值较大的点
    skeleton_map = dist_map.data > 0.2  # 使用阈值提取骨架
    
    # 获取所有骨架点 (i, j) - 像素坐标
    skeleton_i, skeleton_j = np.where(skeleton_map)
    skeleton_points_pixels = np.column_stack([skeleton_i, skeleton_j])  # shape: (M, 2)
    
    if len(skeleton_points_pixels) == 0:
        print("警告: 未找到骨架点，使用默认方向")
        for i in range(N):
            for dir_idx in range(num_directions):
                directions[i, dir_idx] = dir_idx * angle_step_rad
        return directions
    
    print(f"\n从距离场提取到 {len(skeleton_points_pixels)} 个骨架点")
    print(f"处理 {N} 个采样点...")
    
    valid_count = 0
    
    for idx, point in enumerate(discretized_points):
        # 1. 将真实世界坐标转换为像素坐标
        x_world, y_world = point[0], point[1]
        j_center = (x_world - xmin) / resolution  # 列索引（浮点数）
        i_center = (y_world - ymin) / resolution  # 行索引（浮点数）
        
        # 2. 边界检查
        if not (0 <= i_center < height and 0 <= j_center < width):
            print(f"  警告: 点 {idx} 超出地图范围")
            for dir_idx in range(num_directions):
                directions[idx, dir_idx] = dir_idx * angle_step_rad
            continue
        
        # 3. 计算当前点到所有骨架点的距离（像素坐标）
        distances = np.sqrt((skeleton_points_pixels[:, 0] - i_center)**2 + 
                          (skeleton_points_pixels[:, 1] - j_center)**2)
        
        # 4. 找到最近的n_neighbors个点
        n_actual = min(n_neighbors, len(distances))
        closest_indices = np.argpartition(distances, n_actual-1)[:n_actual]
        closest_points = skeleton_points_pixels[closest_indices]
        
        if len(closest_points) < 2:
            # 邻近点太少，使用默认方向
            for dir_idx in range(num_directions):
                directions[idx, dir_idx] = dir_idx * angle_step_rad
            continue
        
        # 5. PCA计算主方向
        # 提取i和j坐标
        i_coords = closest_points[:, 0].astype(float)
        j_coords = closest_points[:, 1].astype(float)
        
        # 中心化
        mean_i = np.mean(i_coords)
        mean_j = np.mean(j_coords)
        
        i_centered = i_coords - mean_i
        j_centered = j_coords - mean_j
        
        # 构建数据矩阵：每行一个点 [j, i]（对应世界坐标x, y）
        data = np.column_stack([j_centered, i_centered])
        
        # 计算协方差矩阵
        cov = np.cov(data.T)
        
        # 特征值分解
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        
        # 选择最大特征值对应的特征向量
        max_idx = np.argmax(eigenvalues)
        principal_dir = eigenvectors[:, max_idx]  # [dj, di] in pixel coords
        
        # 6. 转换到世界坐标系
        # 像素坐标：j向右(+x)，i向下(+y in image, -y in world)
        # 世界坐标：x向右，y向上
        dx = principal_dir[0]  # dj -> dx
        dy = -principal_dir[1]  # -di -> dy (因为i向下，world y向上)
        
        # 7. 计算切线角度
        main_angle = np.arctan2(dy, dx)
        
        valid_count += 1
        
        # 8. 根据角度间隔生成方向
        for dir_idx in range(num_directions):
            directions[idx, dir_idx] = main_angle + dir_idx * angle_step_rad
    
    # 9. 归一化角度到 [-π, π]
    directions = np.arctan2(np.sin(directions), np.cos(directions))
    
    print(f"\n【处理结果】")
    print(f"  有效点数: {valid_count}/{N}")
    print(f"  方向类型: 切线方向(沿着走廊) - PCA方法")
    print(f"  每个点的方向数: {num_directions}")
    print("="*50 + "\n")
    
    return directions

def compute_directions_at_points(discretized_points, dist_map, grad_maps, 
                                 xmin, ymin, resolution, angle_interval=60):
    """
    在每个骨架采样点上基于切线方向计算均匀分布的方向角度
    
    Args:
        discretized_points: numpy数组,形状为(N, 2),真实世界坐标(米)
        dist_map: 距离场地图对象
        grad_maps: 梯度地图列表 [grad_x_map, grad_y_map]
        xmin, ymin: 真实世界坐标的最小值
        resolution: 地图分辨率(米/像素)
        angle_interval: 角度间隔(度),可选: 45, 60, 90, 120
    
    Returns:
        directions: numpy数组,形状为(N, num_directions),每个点的方向角度(弧度)
    """
    print("\n" + "="*50)
    print("计算骨架点的方向角度")
    print("="*50)
    
    # 计算方向数量
    num_directions = 360 // angle_interval
    angle_step_rad = np.deg2rad(angle_interval)
    
    print(f"角度间隔: {angle_interval}度")
    print(f"方向数量: {num_directions}")
    print(f"方向类型: 切线方向(沿着走廊)")
    
    N = len(discretized_points)
    directions = np.zeros((N, num_directions))
    
    # 提取梯度数据
    grad_x = grad_maps[0].data  # shape: (height, width)
    grad_y = grad_maps[1].data
    
    print(f"\n处理 {N} 个采样点...")
    
    valid_count = 0
    for i, point in enumerate(discretized_points):
        # 1. 将真实世界坐标转换为像素坐标
        x_world, y_world = point[0], point[1]
        j = int((x_world - xmin) / resolution)  # 列索引
        i_pixel = int((y_world - ymin) / resolution)  # 行索引
        
        # 2. 边界检查
        if (0 <= i_pixel < grad_x.shape[0] and 0 <= j < grad_x.shape[1]):
            # 3. 获取该点的梯度
            gx = grad_x[i_pixel, j]
            gy = grad_y[i_pixel, j]
            
            # 4. 计算梯度大小
            grad_mag = np.sqrt(gx**2 + gy**2)
            
            if grad_mag > 1e-6:  # 避免除零
                valid_count += 1
                
                # 计算梯度方向(垂直于骨架,指向开放空间)
                gradient_angle = np.arctan2(gy, gx)
                
                # 骨架切线方向 = 梯度方向旋转90度
                # 沿着走廊的方向(平行于骨架)
                main_angle = gradient_angle + np.pi/2
                
                # 根据角度间隔生成方向
                for dir_idx in range(num_directions):
                    directions[i, dir_idx] = main_angle + dir_idx * angle_step_rad
            else:
                # 梯度太小,使用均匀分布的默认方向
                for dir_idx in range(num_directions):
                    directions[i, dir_idx] = dir_idx * angle_step_rad
        else:
            print(f"  警告: 点 {i} 超出地图范围")
            for dir_idx in range(num_directions):
                directions[i, dir_idx] = dir_idx * angle_step_rad
    
    # 5. 归一化角度到 [-π, π]
    directions = np.arctan2(np.sin(directions), np.cos(directions))
    
    print(f"\n【处理结果】")
    print(f"  有效点数: {valid_count}/{N}")
    print(f"  方向类型: 切线方向(沿着走廊)")
    print(f"  每个点的方向数: {num_directions}")
    print("="*50 + "\n")
    
    return directions


def save_points_with_directions(discretized_points, directions, output_path):
    """
    保存采样点及其方向信息到单个npy文件
    格式: 每行为 [x, y, angle1, angle2, ...]
    
    Args:
        discretized_points: 采样点坐标,形状(N, 2)
        directions: 方向角度(弧度),形状(N, num_directions)
        output_path: 保存路径
    """
    # 合并坐标和方向: [x, y, dir0, dir1, dir2, ...]
    data_combined = np.hstack([discretized_points, directions])
    
    np.save(output_path, data_combined)
    print(f"✓ 数据已保存到: {output_path}")
    print(f"  数据形状: {data_combined.shape}")
    print(f"  格式: [x, y, angle1, angle2, ...] (角度单位:弧度)")
    
    # 打印示例
    print(f"\n【数据示例】前3个点:")
    for i in range(min(3, len(data_combined))):
        print(f"  点 {i}:")
        print(f"    坐标: ({data_combined[i,0]:.3f}, {data_combined[i,1]:.3f}) 米")
        print(f"    方向(度): {np.rad2deg(data_combined[i,2:])}")
    
    return data_combined


def main():
    # 解析命令行参数
    args = parse_args()
    load_runtime_dependencies()
    
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # 固定分支修剪长度
    min_branch_length = 15
    
    # Extract the scene key from the input file name.
    keyword = extract_keyword_from_map_filename(args.map_path)
    print(f"从输入文件提取关键词: {keyword}")
    
    # 构建输出文件路径
    skeleton_img_path = os.path.join(output_dir, f"{keyword}_skeleton.png")
    skeleton_svg_img_path = os.path.join(output_dir, f"{keyword}_skeleton.svg")
    skeleton_points_path = os.path.join(output_dir, f"{keyword}_skeleton_points.npy")
    skeleton_with_dirs_path = os.path.join(output_dir, f"{keyword}_skeleton_points_with_directions.npy")
    
    print("\n" + "="*60)
    print("骨架提取和离散化处理")
    print("="*60)
    print(f"输入地图: {args.map_path}")
    print(f"输入元数据: {args.metadata_path}")
    print(f"采样间距: {args.spacing} 米")
    print(f"角度间隔: {args.angle_interval} 度")
    print(f"方向数量: {360 // args.angle_interval}")
    print(f"分支修剪长度: {min_branch_length} 像素")
    print(f"输出目录: {output_dir}")
    
    # 检查metadata文件是否存在
    if not os.path.exists(args.metadata_path):
        raise FileNotFoundError(f"找不到元数据文件: {args.metadata_path}")
    
    print(f"\n加载元数据: {args.metadata_path}")
    config_data = np.load(args.metadata_path, allow_pickle=True).item()
    
    # 提取坐标范围
    xmin = config_data['xmin']
    xmax = config_data['xmax']
    ymin = config_data['ymin']
    ymax = config_data['ymax']
    cell_size = config_data['cell_size']
    
    # Map file configuration
    map_file_config = {
        'map_path': args.map_path,
        'map_resolution': cell_size,
        'map_origin': np.array([0, 0])
    }
    
    ### Map initialization
    map_data = imageToArray(map_file_config['map_path'])
    env_map = EnvironmentMap(
        map_file_config['map_resolution'],
        map_data.shape,
        map_file_config['map_origin'],
        data=map_data
    )
    
    # Map preprocessing configuration
    map_preproc_config = {
        'subsampling_factor': 1.,
        'obst_thresh': 0.4,
        'dilation_erosion_dist': 0.6,
        'dist_field_offset': 0.2
    }
    
    ### Map preprocessing
    # Subsampling
    subsamp_factor = 1
    subsampled_map = subsampleMap(env_map, map_preproc_config['subsampling_factor'])
    # Obstacles extraction
    obst_map = subsampled_map.copy()
    obst_map.setData(subsampled_map.data > map_preproc_config['obst_thresh'])
    # Smoothing (dilation + erosion) of obstacles
    filtered_bin_map = mapDilateErode(obst_map, map_preproc_config['dilation_erosion_dist'])
    filtered_occ_map = subsampled_map.copy()
    filtered_occ_map.data[np.where(filtered_bin_map.data == 1)] = 1
    # Signed distance field and gradients
    dist_map, grad_maps = computeDistsScipy(filtered_bin_map, map_preproc_config['obst_thresh'], 
                                            obst_d_offset=map_preproc_config['dist_field_offset'], 
                                            compute_negative_dist=True)
    dist_grad_array = np.array([grad_maps[0].data, grad_maps[1].data])
    
    # Skeleton extraction
    skeleton_extraction_config = {
        'flux_threshold': -1e-2
    }
    
    ### Skeleton extraction
    # Compute gradient flux
    dist_grad_flux = gradFlux(dist_grad_array, include_diag_neighbors=True)
    grad_flux_map = dist_map.copy()
    grad_flux_map.setData(dist_grad_flux)
    
    thresh_flux_map = grad_flux_map.copy()
    thresh_flux_map.setData(thresh_flux_map.data < skeleton_extraction_config['flux_threshold'])
    
    # Threshold and thin the flux to obtain the skeleton
    thin_flux = fluxToSkeletonMap(dist_grad_flux, skeleton_extraction_config['flux_threshold'])
    skeleton_map = subsampled_map.copy()
    skeleton_map.setData(thin_flux.astype(int) * (dist_map.data > 0.2))
    
    skeleton_data = skeleton_map.data.astype(bool)
    # 只保留最大连通分量
    skeleton_data = keep_largest_component(skeleton_data)
    skeleton_map.setData(skeleton_data.astype(int))
    
    print_resolution_info(skeleton_map, xmin, xmax, ymin, ymax)
    
    # 保存修剪后的骨架图
    pruned_skel = save_pruned_skeleton(
        skeleton_map=skeleton_map,
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        min_branch_length=min_branch_length,
        save_path=skeleton_img_path,
        save_svg_path=skeleton_svg_img_path
    )
    
    # 离散化骨架
    discretized_points = discretize_skeleton_improved(
        skeleton_map=skeleton_map,
        xmin=xmin,
        ymin=ymin,
        resolution=map_file_config['map_resolution'],
        spacing=args.spacing,
        min_branch_length=min_branch_length,
        use_pruning=True,
        output_path=skeleton_points_path
    )
    
    # 计算方向（使用切线方向）
    directions = compute_directions_at_points(
        discretized_points=discretized_points,
        dist_map=dist_map,
        grad_maps=grad_maps,
        xmin=xmin,
        ymin=ymin,
        resolution=map_file_config['map_resolution'],
        angle_interval=args.angle_interval
    )
    
    # 保存带方向的数据
    data_combined = save_points_with_directions(
        discretized_points, 
        directions,
        skeleton_with_dirs_path
    )
    
    # 验证加载
    loaded_data = np.load(skeleton_with_dirs_path, allow_pickle=True)
    print(f"\n{'='*60}")
    print("验证加载的数据:")
    print(f"  数据形状: {loaded_data.shape}")
    print(f"  点数量: {loaded_data.shape[0]}")
    print(f"  每个点的列数: {loaded_data.shape[1]} (2个坐标 + {loaded_data.shape[1]-2}个角度)")
    print(f"\n  第一个点的数据:")
    print(f"    坐标: ({loaded_data[0,0]:.3f}, {loaded_data[0,1]:.3f})")
    print(f"    角度(弧度): {loaded_data[0,2:]}")
    print(f"    角度(度): {np.rad2deg(loaded_data[0,2:])}")
    print(f"{'='*60}\n")
    
    print("\n" + "="*60)
    print("✓ 所有处理完成！")
    print(f"✓ 输出文件:")
    print(f"  - 骨架图: {skeleton_img_path}")
    print(f"  - 采样点: {skeleton_points_path}")
    print(f"  - 带方向数据: {skeleton_with_dirs_path}")
    print("="*60)


if __name__ == "__main__":
    main()
