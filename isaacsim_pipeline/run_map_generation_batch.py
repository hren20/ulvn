#!/usr/bin/env python3
"""
自动化脚本：自动化批量处理USD场景文件
功能流程：
1. 遍历指定目录下的所有子文件夹，找到 start_result_navigation.usd 文件
2. 调用 export_voxel_map.py 生成3D占用网格
3. 调用 export_passable_area.py 提取可通行区域
4. 调用 export_skeleton_points.py 提取骨架点

"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
import json


class PipelineProcessor:
    def __init__(self, args):
        """初始化处理器"""
        self.args = args
        
        # USD文件搜索路径
        self.usd_base_dir = Path(args.usd_base_dir)
        
        self.output_voxel_dir = Path(args.output_voxel_dir)
        self.output_passable_dir = Path(args.output_passable_dir)
        self.output_skeleton_dir = Path(args.output_skeleton_dir)
        
        # 子脚本路径（与本批处理脚本在同一目录）
        self.script_dir = Path(__file__).parent.resolve()
        self.script1 = self.script_dir / "export_voxel_map.py"
        self.script2 = self.script_dir / "export_passable_area.py"
        self.script3 = self.script_dir / "export_skeleton_points.py"
        
        # 统计信息
        self.total_usd_files = 0
        self.successful_count = 0
        self.failed_records = []
        self.processing_log = []
        
    def find_usd_files(self):
        """查找所有USD文件并按字母顺序排序"""
        print("\n" + "="*70)
        print("步骤1: 搜索USD文件")
        print("="*70)
        print(f"搜索目录: {self.usd_base_dir}")
        
        if not self.usd_base_dir.exists():
            print(f"✗ 错误: 目录不存在: {self.usd_base_dir}")
            sys.exit(1)
        
        usd_files = []
        
        # 遍历所有子文件夹
        for subfolder in sorted(self.usd_base_dir.iterdir()):
            if subfolder.is_dir():
                usd_path = subfolder / "start_result_navigation.usd"
                if usd_path.exists():
                    usd_files.append(usd_path)
                    print(f"  ✓ 找到: {subfolder.name}/start_result_navigation.usd")
        
        self.total_usd_files = len(usd_files)
        print(f"\n总共找到 {self.total_usd_files} 个USD文件")
        
        if self.total_usd_files == 0:
            print("✗ 未找到任何USD文件，退出")
            sys.exit(1)
        
        return usd_files
    
    def verify_scripts_exist(self):
        """验证所有子脚本是否存在"""
        print("\n" + "="*70)
        print("验证子脚本")
        print("="*70)
        
        scripts = {
            "脚本1 (3D占用网格)": self.script1,
            "脚本2 (可通行区域)": self.script2,
            "脚本3 (骨架提取)": self.script3,
        }
        
        all_exist = True
        for name, script_path in scripts.items():
            if script_path.exists():
                print(f"  ✓ {name}: {script_path.name}")
            else:
                print(f"  ✗ {name}: 未找到 {script_path}")
                all_exist = False
        
        if not all_exist:
            print("\n✗ 错误: 部分脚本文件不存在，请检查")
            sys.exit(1)
        
        print("\n✓ 所有脚本验证通过")
    
    def run_script1_voxel(self, usd_path, index):
        """运行脚本1: 生成3D占用网格"""
        output_name = f"taoyuan{index}"
        
        cmd = [
            sys.executable,
            str(self.script1),
            "--usd_path", str(usd_path),
            "--z_low", str(self.args.z_low),
            "--z_high", str(self.args.z_high),
            "--cell_size", str(self.args.cell_size),
            "--margin", str(self.args.margin),
            "--output_name", output_name,
            "--output_dir", str(self.output_voxel_dir)
        ]
        
        print(f"\n  [步骤1/3] 生成3D占用网格...")
        print(f"  命令: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5分钟超时
            )
            
            if result.returncode == 0:
                print(f"  ✓ 成功生成: occupancy_3d_{output_name}.npy")
                return True, output_name
            else:
                print(f"  ✗ 失败 (返回码: {result.returncode})")
                print(f"  错误信息: {result.stderr[-500:]}")  # 只显示最后500字符
                return False, None
                
        except subprocess.TimeoutExpired:
            print(f"  ✗ 超时 (>5分钟)")
            return False, None
        except Exception as e:
            print(f"  ✗ 异常: {e}")
            return False, None
    
    def run_script2_passable(self, output_name):
        """运行脚本2: 提取可通行区域"""
        npy_path = self.output_voxel_dir / f"occupancy_3d_{output_name}.npy"
        
        if not npy_path.exists():
            print(f"  ✗ 输入文件不存在: {npy_path}")
            return False
        
        cmd = [
            sys.executable,
            str(self.script2),
            "--npy_path", str(npy_path),
            "--z_low", str(self.args.z_low),
            "--z_high", str(self.args.z_high),
            "--robot_radius", str(self.args.robot_radius),
            "--output_dir", str(self.output_passable_dir)
        ]
        
        print(f"\n  [步骤2/3] 提取可通行区域...")
        print(f"  命令: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=180  # 3分钟超时
            )
            
            if result.returncode == 0:
                print(f"  ✓ 成功生成: {output_name}_max_passable_area.png")
                return True
            else:
                print(f"  ✗ 失败 (返回码: {result.returncode})")
                print(f"  错误信息: {result.stderr[-500:]}")
                return False
                
        except subprocess.TimeoutExpired:
            print(f"  ✗ 超时 (>3分钟)")
            return False
        except Exception as e:
            print(f"  ✗ 异常: {e}")
            return False
    
    def run_script3_skeleton(self, output_name):
        """运行脚本3: 提取骨架点"""
        map_path = self.output_passable_dir / f"{output_name}_max_passable_area.png"
        metadata_path = self.output_voxel_dir / f"occupancy_3d_{output_name}_metadata.npy"
        
        if not map_path.exists():
            print(f"  ✗ 输入文件不存在: {map_path}")
            return False
        
        if not metadata_path.exists():
            print(f"  ✗ 元数据文件不存在: {metadata_path}")
            return False
        
        cmd = [
            sys.executable,
            str(self.script3),
            "--map_path", str(map_path),
            "--metadata_path", str(metadata_path),
            "--spacing", str(self.args.spacing),
            "--angle_interval", str(self.args.angle_interval),
            "--output_dir", str(self.output_skeleton_dir)
        ]
        
        print(f"\n  [步骤3/3] 提取骨架点...")
        print(f"  命令: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=180  # 3分钟超时
            )
            
            if result.returncode == 0:
                print(f"  ✓ 成功生成: {output_name}_skeleton_points_with_directions.npy")
                return True
            else:
                print(f"  ✗ 失败 (返回码: {result.returncode})")
                print(f"  错误信息: {result.stderr[-500:]}")
                return False
                
        except subprocess.TimeoutExpired:
            print(f"  ✗ 超时 (>3分钟)")
            return False
        except Exception as e:
            print(f"  ✗ 异常: {e}")
            return False
    
    def process_single_usd(self, usd_path, index):
        """处理单个USD文件"""
        print("\n" + "#"*70)
        print(f"处理第 {index} 个USD文件: {usd_path.parent.name}")
        print("#"*70)
        
        record = {
            "index": index,
            "usd_path": str(usd_path),
            "folder_name": usd_path.parent.name,
            "status": "unknown",
            "failed_step": None,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # 步骤1: 生成3D占用网格
        success1, output_name = self.run_script1_voxel(usd_path, index)
        if not success1:
            record["status"] = "failed"
            record["failed_step"] = "step1_voxel"
            self.failed_records.append(record)
            print(f"\n✗ 第 {index} 个USD处理失败于步骤1")
            return False
        
        # 步骤2: 提取可通行区域
        success2 = self.run_script2_passable(output_name)
        if not success2:
            record["status"] = "failed"
            record["failed_step"] = "step2_passable"
            self.failed_records.append(record)
            print(f"\n✗ 第 {index} 个USD处理失败于步骤2")
            return False
        
        # 步骤3: 提取骨架点
        success3 = self.run_script3_skeleton(output_name)
        if not success3:
            record["status"] = "failed"
            record["failed_step"] = "step3_skeleton"
            self.failed_records.append(record)
            print(f"\n✗ 第 {index} 个USD处理失败于步骤3")
            return False
        
        # 全部成功
        record["status"] = "success"
        record["output_name"] = output_name
        self.processing_log.append(record)
        print(f"\n✓ 第 {index} 个USD处理完成: {output_name}")
        return True
    
    def process_all(self):
        """处理所有USD文件"""
        # 查找USD文件
        usd_files = self.find_usd_files()
        
        # 验证脚本
        self.verify_scripts_exist()
        
        # 显示参数
        self.show_parameters()
        
        # 确认开始
        print("\n" + "="*70)
        confirm = input(f"准备处理 {self.total_usd_files} 个USD文件，是否继续? (y/n): ").strip().lower()
        if confirm != 'y':
            print("已取消")
            sys.exit(0)
        
        # 开始处理
        print("\n" + "="*70)
        print("开始批量处理")
        print("="*70)
        start_time = datetime.now()
        
        for idx, usd_path in enumerate(usd_files, start=1):
            success = self.process_single_usd(usd_path, idx)
            if success:
                self.successful_count += 1
            
            # 显示进度
            print(f"\n进度: {idx}/{self.total_usd_files} | 成功: {self.successful_count} | 失败: {len(self.failed_records)}")
        
        # 处理完成
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        self.print_summary(duration)
        self.save_report()
    
    def show_parameters(self):
        """显示处理参数"""
        print("\n" + "="*70)
        print("处理参数")
        print("="*70)
        print(f"[步骤1] 3D占用网格:")
        print(f"  - z_low: {self.args.z_low} m")
        print(f"  - z_high: {self.args.z_high} m")
        print(f"  - cell_size: {self.args.cell_size} m")
        print(f"  - margin: {self.args.margin} m")
        print(f"\n[步骤2] 可通行区域:")
        print(f"  - robot_radius: {self.args.robot_radius} m")
        print(f"\n[步骤3] 骨架提取:")
        print(f"  - spacing: {self.args.spacing} m")
        print(f"  - angle_interval: {self.args.angle_interval}°")
        print("="*70)
    
    def print_summary(self, duration):
        """打印处理摘要"""
        print("\n" + "#"*70)
        print("批量处理完成")
        print("#"*70)
        print(f"\n总处理文件数: {self.total_usd_files}")
        print(f"  ✓ 成功: {self.successful_count}")
        print(f"  ✗ 失败: {len(self.failed_records)}")
        
        if self.total_usd_files > 0:
            success_rate = self.successful_count / self.total_usd_files * 100
            print(f"\n成功率: {success_rate:.1f}%")
        
        print(f"总耗时: {duration:.1f} 秒 ({duration/60:.1f} 分钟)")
        
        if self.failed_records:
            print(f"\n失败的文件:")
            for record in self.failed_records:
                print(f"  - 第 {record['index']} 个: {record['folder_name']} (失败于: {record['failed_step']})")
        
        print(f"\n输出目录:")
        print(f"  - 3D占用网格: {self.output_voxel_dir}")
        print(f"  - 可通行区域: {self.output_passable_dir}")
        print(f"  - 骨架点: {self.output_skeleton_dir}")
        print("#"*70 + "\n")
    
    def save_report(self):
        """保存处理报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.script_dir / f"pipeline_report_{timestamp}.json"
        
        report = {
            "timestamp": timestamp,
            "parameters": vars(self.args),
            "statistics": {
                "total_files": self.total_usd_files,
                "successful": self.successful_count,
                "failed": len(self.failed_records)
            },
            "successful_records": self.processing_log,
            "failed_records": self.failed_records
        }
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"✓ 处理报告已保存: {report_file}")


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='批量处理USD场景文件的主控脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 使用默认参数
  python run_map_generation_batch.py
  
  # 自定义所有参数
  python run_map_generation_batch.py --z_low 0.3 --z_high 1.5 --cell_size 0.1 \\
                           --margin 30.0 --robot_radius 0.4 \\
                           --spacing 1.0 --angle_interval 90

参数说明:
  [步骤1] 3D占用网格生成参数
  [步骤2] 可通行区域提取参数
  [步骤3] 骨架点提取参数
        """
    )
    parser.add_argument('--usd_base_dir', type=str,
                        default='assets/GRScenes-100/home_scenes/scenes',
                        help='包含场景子目录的USD根目录')
    parser.add_argument('--output_voxel_dir', type=str,
                        default='outputs/occupancy_data',
                        help='3D占用栅格输出目录')
    parser.add_argument('--output_passable_dir', type=str,
                        default='outputs/passable_area',
                        help='2D可通行区域输出目录')
    parser.add_argument('--output_skeleton_dir', type=str,
                        default='outputs/skeleton_points',
                        help='骨架点和方向输出目录')
    
    # 步骤1: 3D占用网格参数
    parser.add_argument('--z_low', type=float, default=0.2,
                        help='扫描最低高度 (米, 默认: 0.2)')
    parser.add_argument('--z_high', type=float, default=1.2,
                        help='扫描最高高度 (米, 默认: 1.2)')
    parser.add_argument('--cell_size', type=float, default=0.05,
                        help='网格单元大小 (米, 默认: 0.05)')
    parser.add_argument('--margin', type=float, default=25.0,
                        help='场景边界扩展边距 (米, 默认: 25.0)')
    
    # 步骤2: 可通行区域参数
    parser.add_argument('--robot_radius', type=float, default=0.3,
                        help='机器人半径 (米, 默认: 0.3)')
    
    # 步骤3: 骨架提取参数
    parser.add_argument('--spacing', type=float, default=0.5,
                        help='采样点间距 (米, 默认: 0.5)')
    parser.add_argument('--angle_interval', type=int, choices=[45, 60, 90, 120], default=60,
                        help='方向角度间隔 (度, 可选: 45/60/90/120, 默认: 60)')
    
    return parser.parse_args()


def main():
    """主函数"""
    print("\n" + "="*70)
    print("USD场景批量处理主控脚本")
    print("="*70)
    print("功能: 自动化处理多个USD场景文件")
    print("流程: USD → 3D占用网格 → 可通行区域 → 骨架点")
    print("="*70)
    
    # 解析参数
    args = parse_arguments()
    
    # 创建处理器并执行
    try:
        processor = PipelineProcessor(args)
        processor.process_all()
        print("\n✓ 所有处理完成!")
        
    except KeyboardInterrupt:
        print("\n\n✗ 用户中断处理")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
