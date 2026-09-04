#!/usr/bin/env python3
"""
基于NPY坐标文件的批量图像捕获工具（支持悬浮模式和碰撞检测）

功能：
1. 从NPY文件读取坐标列表
2. 可选使用"悬浮模式"或"直接设置"模式进行定位
   - 悬浮模式: 更稳定，机器人不会掉落或翻转
   - 直接设置: 速度可能更快，但依赖于物理引擎的稳定性
3. 碰撞检测与前进重试 - 碰撞时沿朝向前进重试，直到找到安全位置或达到最大尝试次数
4. 碰撞后稳定等待 - 在碰撞并移动到新位置后，等待机器人完全稳定再拍照
5. 捕获并保存图像
6. 生成详细的执行报告

改进：
- 碰撞时不再直接跳过，而是沿当前朝向前进小距离重试
- 可配置前进步长和最大重试次数
- 碰撞后移动到新位置会等待稳定时间，确保拍照质量
"""

try:
    import rospy
    from geometry_msgs.msg import Pose
    from sensor_msgs.msg import Image
    from nav_msgs.msg import Odometry
    from std_msgs.msg import Bool, String
    from geometry_msgs.msg import Vector3Stamped
    from cv_bridge import CvBridge
    ROSInterruptException = rospy.ROSInterruptException
except ImportError:
    rospy = None
    Pose = Image = Odometry = Bool = String = Vector3Stamped = CvBridge = None
    ROSInterruptException = KeyboardInterrupt

try:
    import cv2
except ImportError:
    cv2 = None
import numpy as np
import math
import time
import os
import sys
from datetime import datetime
import json
import argparse


def _require_capture_runtime_dependencies():
    missing = []
    if rospy is None or Pose is None or Image is None or Odometry is None:
        missing.append("ROS Python packages: rospy, geometry_msgs, sensor_msgs, nav_msgs, std_msgs")
    if CvBridge is None:
        missing.append("cv_bridge")
    if cv2 is None:
        missing.append("opencv-python / cv2")
    if missing:
        raise RuntimeError(
            "capture_photos.py requires additional runtime dependencies for image capture: "
            + "; ".join(missing)
            + ". Install/provide them in the ROS/IsaacSim environment before running capture."
        )


class NPYCoordinateCapturer:
    def __init__(
        self,
        save_dir,
        use_hover=False,
        advance_distance=0.1,
        max_advance_attempts=5,
        stabilization_time=2.0,
    ):
        """
        初始化NPY坐标捕获器

        Args:
            save_dir: 保存图像的目录
            use_hover: 是否使用悬浮模式进行定位
            advance_distance: 碰撞后每次前进的距离（米）
            max_advance_attempts: 最大前进尝试次数
            stabilization_time: 碰撞后移动到新位置的稳定等待时间（秒）
        """
        rospy.init_node("npy_coordinate_capturer", anonymous=True)

        self.use_hover = use_hover
        self.save_dir = save_dir
        self.advance_distance = advance_distance
        self.max_advance_attempts = max_advance_attempts
        self.stabilization_time = stabilization_time
        
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            rospy.loginfo(f"[Capturer] Created directory: {self.save_dir}")

        self.bridge = CvBridge()

        # --- 发布器 ---
        if self.use_hover:
            rospy.loginfo("[Capturer] Using HOVER mode for positioning.")
            self.command_pub = rospy.Publisher("/isaac_node/command", String, queue_size=1)
        else:
            rospy.loginfo("[Capturer] Using SET_POSE_SIMPLE mode for positioning.")
            self.pose_pub = rospy.Publisher(
                "/isaac_node/set_pose_simple", Pose, queue_size=1
            )

        # --- 订阅器 ---
        self.image_sub = rospy.Subscriber(
            "/isaac_node/camera0/image_raw", Image, self.image_callback
        )
        self.odom_sub = rospy.Subscriber(
            "/isaac_node/odom", Odometry, self.odom_callback
        )
        self.collision_sub = rospy.Subscriber(
            "/isaac_node/collision_detected", Bool, self.collision_callback
        )
        self.collision_force_sub = rospy.Subscriber(
            "/isaac_node/collision_force", Vector3Stamped, self.collision_force_callback
        )
        if self.use_hover:
            self.status_sub = rospy.Subscriber(
                "/isaac_node/command_status", String, self.status_callback
            )

        # --- 状态变量 ---
        self.current_pose = None
        self.latest_image = None
        self.collision_detected = False
        self.collision_force = None
        self.last_collision_check_time = None
        self.command_status = None

        # --- 统计信息 ---
        self.total_coordinates = 0
        self.successful_captures = 0
        self.skipped_collisions = 0
        self.failed_captures = 0
        self.positioning_failures = 0
        self.advance_successes = 0  # 前进后成功的次数
        self.advance_failures = 0   # 前进仍失败的次数

        self.capture_records = []
        rospy.loginfo(f"[Capturer] Collision handling: advance {advance_distance}m, max {max_advance_attempts} attempts")
        rospy.loginfo(f"[Capturer] Stabilization time after collision: {stabilization_time}s")
        rospy.loginfo("[Capturer] ✓ Initialized successfully")

    def status_callback(self, msg):
        """接收命令状态回调"""
        try:
            self.command_status = json.loads(msg.data)
        except Exception as e:
            rospy.logwarn(f"[Capturer] Failed to parse status JSON: {e}")

    def image_callback(self, msg: Image):
        """接收图像回调"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.latest_image = cv_image
        except Exception as e:
            rospy.logerr(f"[Capturer] Failed to convert image: {e}")

    def odom_callback(self, msg: Odometry):
        """接收里程计回调"""
        self.current_pose = msg.pose.pose

    def collision_callback(self, msg: Bool):
        """碰撞检测回调"""
        self.collision_detected = msg.data
        self.last_collision_check_time = rospy.Time.now()

    def collision_force_callback(self, msg: Vector3Stamped):
        """碰撞力回调"""
        self.collision_force = msg

    def load_coordinates(self, npy_file, default_z=0.4):
        """
        从NPY文件加载坐标
        格式: [x, y, ang]
        
        Args:
            npy_file: NPY文件路径
            default_z: 默认高度
        
        Returns:
            coordinates: numpy数组,格式为 [x, y, z, yaw]
        """
        try:
            data = np.load(npy_file)
            rospy.loginfo(f"[Capturer] Loaded data from: {npy_file}")
            rospy.loginfo(f"[Capturer] Shape: {data.shape}")
            
            if data.shape[1] != 3:
                rospy.logerr(f"[Capturer] Expected 3 columns [x, y, ang], got {data.shape[1]}")
                return None
            
            rospy.loginfo(f"[Capturer] Format: [x, y, ang]")
            
            # 创建坐标数组: [x, y, z, yaw]
            coordinates = np.zeros((len(data), 4))
            coordinates[:, 0] = data[:, 0]  # x
            coordinates[:, 1] = data[:, 1]  # y
            coordinates[:, 2] = default_z    # z
            coordinates[:, 3] = data[:, 2]   # yaw (angle)
            
            # 打印统计信息
            rospy.loginfo(f"[Capturer] Total capture positions: {len(coordinates)}")
            rospy.loginfo(f"[Capturer] Using z height: {default_z} m")
            
            if len(coordinates) > 0:
                x_min, x_max = coordinates[:, 0].min(), coordinates[:, 0].max()
                y_min, y_max = coordinates[:, 1].min(), coordinates[:, 1].max()
                rospy.loginfo(f"[Capturer] X range: [{x_min:.2f}, {x_max:.2f}]")
                rospy.loginfo(f"[Capturer] Y range: [{y_min:.2f}, {y_max:.2f}]")
            
            return coordinates
            
        except Exception as e:
            rospy.logerr(f"[Capturer] Failed to load NPY file: {e}")
            return None

    def check_collision_status(self, wait_time=0.5):
        """检查碰撞状态"""
        rospy.sleep(wait_time)
        if self.last_collision_check_time is None:
            rospy.logwarn("[Capturer] No collision data received yet.")
            return False
        age = (rospy.Time.now() - self.last_collision_check_time).to_sec()
        if age > 2.0:
            rospy.logwarn(f"[Capturer] Collision data is old ({age:.1f}s)")
        return self.collision_detected

    def send_command(self, command_dict, timeout=5):
        """发送命令并等待响应"""
        self.command_status = None
        command_msg = String()
        command_msg.data = json.dumps(command_dict)
        self.command_pub.publish(command_msg)
        
        start_time = time.time()
        while self.command_status is None and (time.time() - start_time) < timeout:
            rospy.sleep(0.1)
            if rospy.is_shutdown(): return False
        
        if self.command_status:
            if self.command_status.get("success", False):
                rospy.loginfo(f"[Capturer] Command successful: {self.command_status.get('message', '')}")
                return True
            else:
                rospy.logerr(f"[Capturer] Command failed: {self.command_status.get('message', 'No message')}")
                return False
        
        rospy.logerr("[Capturer] Command timed out. No response from robot.")
        return False

    def set_robot_pose(self, x, y, z=0.4, yaw=0.0):
        """
        设置机器人位置，根据初始化模式选择悬浮或直接设置。
        返回: (success, has_collision)
        """
        if self.use_hover:
            command = {
                "command": "hover",
                "enable": True,
                "position": {"x": x, "y": y, "z": z},
                "orientation": {
                    "qx": 0.0,
                    "qy": 0.0,
                    "qz": math.sin(yaw / 2.0),
                    "qw": math.cos(yaw / 2.0),
                },
            }
            positioning_success = self.send_command(command)
        else:
            pose_msg = Pose()
            pose_msg.position.x = x
            pose_msg.position.y = y
            pose_msg.position.z = z
            pose_msg.orientation.z = math.sin(yaw / 2.0)
            pose_msg.orientation.w = math.cos(yaw / 2.0)
            self.pose_pub.publish(pose_msg)
            positioning_success = True  # Assume success for simple pose setting

        if not positioning_success:
            return False, False # 定位失败，也无所谓碰撞

        # 等待位置稳定并检查碰撞
        rospy.sleep(2.0)
        has_collision = self.check_collision_status(wait_time=0.5)
        
        return positioning_success, not has_collision

    def calculate_advance_position(self, x, y, yaw, advance_distance):
        """
        计算沿着当前朝向前进指定距离后的位置
        
        Args:
            x, y: 当前位置
            yaw: 当前朝向角度（弧度）
            advance_distance: 前进距离（米）
        
        Returns:
            new_x, new_y: 前进后的新位置
        """
        # 沿着朝向的正方向移动（前进）
        # yaw是机器人的朝向角，前进意味着沿着yaw的方向移动
        new_x = x + advance_distance * math.cos(yaw)
        new_y = y + advance_distance * math.sin(yaw)
        return new_x, new_y

    def try_position_with_advance(self, x, y, z, yaw, index):
        """
        尝试定位，如果碰撞则前进重试
        
        Returns:
            success: 是否成功找到安全位置
            final_x, final_y: 最终安全位置（如果成功）
            advance_count: 前进次数
        """
        current_x, current_y = x, y
        had_collision = False  # 标记是否曾经发生过碰撞
        
        for attempt in range(self.max_advance_attempts + 1):
            print(f"  Attempt {attempt + 1}/{self.max_advance_attempts + 1}: "
                  f"x={current_x:.2f}, y={current_y:.2f}, z={z:.2f}, yaw={yaw:.3f}rad")
            
            # 尝试定位
            positioning_success, is_safe = self.set_robot_pose(current_x, current_y, z, yaw)
            
            if not positioning_success:
                print(f"  ✗ Positioning failed at attempt {attempt + 1}")
                return False, None, None, attempt
            
            if is_safe:
                # Wait for stabilization after moving away from a collision.
                if had_collision and attempt > 0:
                    print(f"  ⏱ Waiting {self.stabilization_time}s for robot to stabilize after collision...")
                    rospy.sleep(self.stabilization_time)
                    print(f"  ✓ Robot stabilized")
                
                if attempt == 0:
                    print(f"  ✓ No collision at original position")
                else:
                    print(f"  ✓ Found safe position after {attempt} advance(s)")
                return True, current_x, current_y, attempt
            
            # 有碰撞
            had_collision = True
            if self.collision_force is not None:
                fx, fy, fz = self.collision_force.vector.x, self.collision_force.vector.y, self.collision_force.vector.z
                force_mag = math.sqrt(fx**2 + fy**2 + fz**2)
                print(f"  ✗ Collision detected (force: {force_mag:.2f} N)")
            else:
                print(f"  ✗ Collision detected")
            
            # 如果还没达到最大尝试次数，计算前进位置
            if attempt < self.max_advance_attempts:
                current_x, current_y = self.calculate_advance_position(
                    current_x, current_y, yaw, self.advance_distance
                )
                print(f"  → Advancing {self.advance_distance}m along orientation...")
            else:
                print(f"  ✗ Max advance attempts ({self.max_advance_attempts}) reached")
        
        return False, None, None, self.max_advance_attempts

    def disable_hover_mode(self):
        """禁用悬浮模式"""
        if not self.use_hover:
            return True
        rospy.loginfo("[Capturer] Disabling hover mode...")
        command = {"command": "hover", "enable": False}
        success = self.send_command(command, timeout=3)
        if success:
            rospy.loginfo("[Capturer] ✓ Hover mode disabled.")
        else:
            rospy.logwarn("[Capturer] ✗ Failed to disable hover mode.")
        return success

    def capture_image(self, x, y, z, yaw, index, advance_count=0):
        """捕获图像并保存"""
        time.sleep(1)
        timeout = 3.0
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.latest_image is not None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                # 如果有前进，在文件名中标注
                advance_suffix = f"_advance{advance_count}" if advance_count > 0 else ""
                filename = f"coord_{index:04d}_x{x:.2f}_y{y:.2f}_z{z:.2f}_yaw{yaw:.3f}rad{advance_suffix}_{timestamp}.png"
                filepath = os.path.join(self.save_dir, filename)
                cv2.imwrite(filepath, self.latest_image)
                return True, filepath
            rospy.sleep(0.1)
        return False, None

    def process_coordinates(self, coordinates):
        """处理坐标列表，批量捕获图像"""
        self.total_coordinates = len(coordinates)

        print("\n" + "=" * 70)
        print("Starting Batch Capture")
        print("=" * 70)
        print(f"Positioning Mode: {'HOVER' if self.use_hover else 'SET_POSE_SIMPLE'}")
        print(f"Total coordinates: {self.total_coordinates}")
        print(f"Advance distance: {self.advance_distance} m")
        print(f"Max advance attempts: {self.max_advance_attempts}")
        print(f"Stabilization time (after collision): {self.stabilization_time} s")
        print(f"Save directory: {os.path.abspath(self.save_dir)}")
        print("=" * 70 + "\n")

        for idx, coord in enumerate(coordinates):
            if rospy.is_shutdown():
                print("\nROS shutdown requested. Stopping capture process.")
                break
            
            print(f"\n{'='*70}")
            print(f"Processing coordinate {idx + 1}/{self.total_coordinates}")
            print(f"{'='*70}")

            x, y, z, yaw = float(coord[0]), float(coord[1]), float(coord[2]), float(coord[3])
            print(f"Target position: x={x:.2f}, y={y:.2f}, z={z:.2f}, yaw={yaw:.3f}rad ({math.degrees(yaw):.1f}°)")

            start_time = time.time()
            status = "unknown"
            filename = None
            advance_count = 0

            print("Step 1: Positioning with collision detection and advance...")
            success, final_x, final_y, advance_count = self.try_position_with_advance(x, y, z, yaw, idx)

            if not success:
                print("✗ FAILED - Could not find safe position after all attempts")
                self.skipped_collisions += 1
                if advance_count > 0:
                    self.advance_failures += 1
                status = "collision_all_advances_failed"
            else:
                # 成功找到安全位置
                if advance_count > 0:
                    print(f"✓ Safe position found after {advance_count} advance(s)")
                    print(f"  Final position: x={final_x:.2f}, y={final_y:.2f}")
                    self.advance_successes += 1
                
                print("Step 2: Capturing image...")
                capture_success, filename = self.capture_image(final_x, final_y, z, yaw, idx, advance_count)
                
                if capture_success:
                    print(f"✓ Image saved: {os.path.basename(filename)}")
                    self.successful_captures += 1
                    status = "success" if advance_count == 0 else "success_after_advance"
                else:
                    print("✗ Failed to capture image (timeout)")
                    self.failed_captures += 1
                    status = "capture_failed"

            # 记录
            record = {
                "index": idx,
                "original_x": x,
                "original_y": y,
                "final_x": final_x if success else None,
                "final_y": final_y if success else None,
                "z": z,
                "yaw": yaw,
                "advance_count": advance_count,
                "status": status,
                "filename": os.path.basename(filename) if filename else None,
                "time": time.time() - start_time,
            }
            self.capture_records.append(record)

            # 显示进度
            progress_str = (
                f"\nProgress: {idx + 1}/{self.total_coordinates} | "
                f"Success: {self.successful_captures} | "
                f"Advance Success: {self.advance_successes} | "
                f"Failed: {self.skipped_collisions + self.failed_captures + self.positioning_failures}"
            )
            print(progress_str)

        self.print_summary()
        self.save_report()

    def print_summary(self):
        """打印执行摘要"""
        print("\n" + "#" * 70)
        print("BATCH CAPTURE COMPLETED")
        print("#" * 70)
        print(f"\nTotal coordinates attempted: {self.total_coordinates}")
        print(f"  - Successful captures (original position): {self.successful_captures - self.advance_successes}")
        print(f"  - Successful captures (after advance): {self.advance_successes}")
        print(f"  - Total successful captures: {self.successful_captures}")
        print(f"  - Failed (collision, all advances failed): {self.advance_failures}")
        print(f"  - Failed (other collisions): {self.skipped_collisions - self.advance_failures}")
        print(f"  - Failed (positioning): {self.positioning_failures}")
        print(f"  - Failed (image capture): {self.failed_captures}")
        
        total_failed = (self.skipped_collisions + self.positioning_failures + 
                       self.failed_captures)
        
        if self.total_coordinates > 0:
            success_rate = self.successful_captures / self.total_coordinates * 100
            advance_success_rate = self.advance_successes / self.total_coordinates * 100
            print(f"\nSuccess rate: {success_rate:.1f}%")
            print(f"  - Success via advance: {advance_success_rate:.1f}%")
        
        print(f"\nAdvance statistics:")
        print(f"  - Advance distance: {self.advance_distance} m")
        print(f"  - Max attempts per position: {self.max_advance_attempts + 1}")
        print(f"  - Positions saved by advance: {self.advance_successes}")
        
        if total_failed > 0:
            print(f"\n⚠ Failed positions: {total_failed}")
            print(f"  → Check 'failed_positions_*.txt' for details and retry coordinates")
        
        print(f"\nSave directory: {os.path.abspath(self.save_dir)}")
        print("#" * 70 + "\n")

    def save_report(self):
        """保存详细报告"""
        if not self.capture_records:
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(self.save_dir, f"capture_report_{timestamp}.json")
        stats = {
            "total_coordinates": self.total_coordinates,
            "successful_captures": self.successful_captures,
            "advance_successes": self.advance_successes,
            "advance_failures": self.advance_failures,
            "skipped_collisions": self.skipped_collisions,
            "positioning_failures": self.positioning_failures,
            "failed_captures": self.failed_captures,
            "advance_distance": self.advance_distance,
            "max_advance_attempts": self.max_advance_attempts,
            "stabilization_time": self.stabilization_time,
        }
        report = {"timestamp": timestamp, "statistics": stats, "records": self.capture_records}
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)
        print(f"✓ JSON report saved: {report_file}")

        # CSV 报告
        csv_file = os.path.join(self.save_dir, f"capture_report_{timestamp}.csv")
        with open(csv_file, "w") as f:
            f.write("index,original_x,original_y,final_x,final_y,z,yaw,advance_count,status,filename,time\n")
            for r in self.capture_records:
                final_x = f"{r['final_x']:.2f}" if r['final_x'] is not None else "N/A"
                final_y = f"{r['final_y']:.2f}" if r['final_y'] is not None else "N/A"
                f.write(
                    f"{r['index']},{r['original_x']:.2f},{r['original_y']:.2f},"
                    f"{final_x},{final_y},{r['z']:.2f},"
                    f"{r['yaw']:.3f},{r['advance_count']},{r['status']},{r['filename']},{r['time']:.2f}\n"
                )
        print(f"✓ CSV report saved: {csv_file}")
        
        # 失败位置记录文件（TXT格式）
        failed_records = [r for r in self.capture_records if not r['status'].startswith('success')]
        if failed_records:
            failed_file = os.path.join(self.save_dir, f"failed_positions_{timestamp}.txt")
            with open(failed_file, "w") as f:
                f.write("=" * 80 + "\n")
                f.write("FAILED CAPTURE POSITIONS\n")
                f.write("=" * 80 + "\n")
                f.write(f"Generated: {timestamp}\n")
                f.write(f"Total failed positions: {len(failed_records)}\n")
                f.write("=" * 80 + "\n\n")
                
                f.write("Format: Index | X | Y | Z | Yaw(rad) | Yaw(deg) | Status\n")
                f.write("-" * 80 + "\n\n")
                
                for r in failed_records:
                    yaw_deg = math.degrees(r['yaw'])
                    f.write(f"Index: {r['index']}\n")
                    f.write(f"  Position: x={r['original_x']:.3f}, y={r['original_y']:.3f}, z={r['z']:.3f}\n")
                    f.write(f"  Orientation: yaw={r['yaw']:.4f} rad ({yaw_deg:.2f}°)\n")
                    f.write(f"  Status: {r['status']}\n")
                    f.write(f"  Advance attempts: {r['advance_count']}\n")
                    f.write("-" * 80 + "\n")
                
                # 添加可直接复制的坐标列表（NPY格式）
                f.write("\n" + "=" * 80 + "\n")
                f.write("COORDINATES FOR RETRY (NPY format: x, y, ang)\n")
                f.write("=" * 80 + "\n")
                f.write("# You can copy these coordinates to a new NPY file for retry\n")
                f.write("# Format: x, y, yaw(rad)\n\n")
                
                for r in failed_records:
                    f.write(f"{r['original_x']:.6f}, {r['original_y']:.6f}, {r['yaw']:.6f}  # Index {r['index']}\n")
                
            print(f"✓ Failed positions saved: {failed_file}")
            print(f"  → {len(failed_records)} failed position(s) recorded")
        else:
            print("✓ No failed positions - all captures successful!")

    def wait_for_initial_data(self, timeout=10.0):
        """等待初始数据"""
        print("Waiting for initial data from ROS topics...")
        start_time = time.time()
        while not rospy.is_shutdown() and (time.time() - start_time < timeout):
            cam_ok = self.latest_image is not None
            odom_ok = self.current_pose is not None
            if cam_ok and odom_ok:
                print("✓ Initial data received\n")
                return True
            rospy.sleep(0.2)
        
        if self.latest_image is None: rospy.logerr("[Capturer] ✗ Failed to receive camera image.")
        if self.current_pose is None: rospy.logerr("[Capturer] ✗ Failed to receive odometry.")
        return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='NPY Coordinate Batch Capturer with Collision Advance',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python %(prog)s --npy_path coords.npy --output_dir ./photos
  python %(prog)s --npy_path coords.npy --use_hover --height 0.5 --advance_distance 0.15
  python %(prog)s --npy_path coords.npy --output_dir ./output --max_advance_attempts 8 --stabilization_time 3.0
        """
    )
    
    parser.add_argument(
        '--npy_path',
        type=str,
        required=True,
        help='NPY文件路径，格式: [x, y, ang]'
    )
    
    parser.add_argument(
        '--use_hover',
        action='store_true',
        default=False,
        help='使用悬浮模式进行定位 (默认: 不使用)'
    )
    
    parser.add_argument(
        '--height',
        type=float,
        default=0.4,
        help='机器人高度z (单位: 米, 默认: 0.4)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='输出目录路径'
    )
    
    parser.add_argument(
        '--advance_distance',
        type=float,
        default=0.1,
        help='碰撞后每次前进的距离 (单位: 米, 默认: 0.1)'
    )
    
    parser.add_argument(
        '--max_advance_attempts',
        type=int,
        default=5,
        help='最大前进尝试次数 (默认: 5)'
    )
    
    parser.add_argument(
        '--stabilization_time',
        type=float,
        default=2.0,
        help='碰撞后移动到新位置的稳定等待时间 (单位: 秒, 默认: 2.0)'
    )

    args = parser.parse_args()

    # 检查NPY文件是否存在
    if not os.path.exists(args.npy_path):
        print(f"✗ Error: NPY file not found: {args.npy_path}")
        return

    _require_capture_runtime_dependencies()

    print("\n" + "=" * 70)
    print("NPY Coordinate Batch Capturer (with Collision Advance)")
    print("=" * 70)
    print(f"NPY file: {args.npy_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Positioning mode: {'HOVER' if args.use_hover else 'SET_POSE_SIMPLE'}")
    print(f"Robot height: {args.height} m")
    print(f"Advance distance: {args.advance_distance} m")
    print(f"Max advance attempts: {args.max_advance_attempts}")
    print(f"Stabilization time: {args.stabilization_time} s")
    print("=" * 70 + "\n")

    capturer = None
    try:
        # 初始化捕获器
        capturer = NPYCoordinateCapturer(
            save_dir=args.output_dir,
            use_hover=args.use_hover,
            advance_distance=args.advance_distance,
            max_advance_attempts=args.max_advance_attempts,
            stabilization_time=args.stabilization_time,
        )

        if not capturer.wait_for_initial_data():
            return

        # 加载坐标
        coordinates = capturer.load_coordinates(args.npy_path, default_z=args.height)
        if coordinates is None or len(coordinates) == 0:
            print("✗ Error: Failed to load coordinates or the file is empty.")
            return

        print("\nCoordinate Preview (first 5):")
        for i, coord in enumerate(coordinates[:5]):
            print(f"  {i}: x={coord[0]:.2f}, y={coord[1]:.2f}, z={coord[2]:.2f}, yaw={coord[3]:.3f}rad ({math.degrees(coord[3]):.1f}°)")
        if len(coordinates) > 5:
            print(f"  ... and {len(coordinates) - 5} more")
        print()

        print(f"\nReady to process {len(coordinates)} capture positions")
        print(f"Collision handling: advance {args.advance_distance}m up to {args.max_advance_attempts} times")
        print(f"Stabilization after collision: {args.stabilization_time}s")
        confirm = input("Start processing? (y/n): ").strip().lower()

        if confirm != "y":
            print("Cancelled by user.")
            return

        capturer.process_coordinates(coordinates)
        print("\n✓ All done!")

    except (ROSInterruptException, KeyboardInterrupt):
        print("\n✗ Process interrupted by user.")
    except Exception as e:
        print(f"\n✗ An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if capturer and capturer.use_hover:
            capturer.disable_hover_mode()


if __name__ == "__main__":
    main()
