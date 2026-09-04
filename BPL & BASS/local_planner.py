import os
import argparse
import numpy as np
from PIL import Image as PILImage
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import yaml
except ImportError:
    yaml = None

try:
    import rospy
    from sensor_msgs.msg import Image
    from std_msgs.msg import Float32MultiArray, String
    from visualization_msgs.msg import Marker
except ImportError:
    rospy = None
    Image = None
    Float32MultiArray = None
    String = None
    Marker = None

try:
    from inference_utils import MODEL_REGISTRY
    from inference_utils.common import load_config, inference_config_init, msg_to_pil, create_marker_from_points
except ImportError:
    MODEL_REGISTRY = None
    load_config = None
    inference_config_init = None
    msg_to_pil = None
    create_marker_from_points = None


def _require_local_runtime_dependencies():
    missing = []
    if rospy is None or Image is None or Float32MultiArray is None or String is None or Marker is None:
        missing.append("ROS Python packages: rospy, sensor_msgs, std_msgs, visualization_msgs")
    if MODEL_REGISTRY is None or load_config is None or inference_config_init is None:
        missing.append("inference_utils with MODEL_REGISTRY and config helpers")
    if msg_to_pil is None or create_marker_from_points is None:
        missing.append("inference_utils.common image/marker helpers")
    if yaml is None:
        missing.append("PyYAML")
    if missing:
        raise RuntimeError(
            "local_planner.py requires additional runtime dependencies: "
            + "; ".join(missing)
            + ". Provide these modules before launching the ROS local planner."
        )


def load_robot_config(path: str):
    if yaml is None:
        raise RuntimeError("PyYAML is required to load robot configuration files.")
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Robot config not found: {config_path}. Copy config/robot.example.yaml "
            "to config/robot.yaml or pass --robot-config."
        )
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    required = ["max_v", "max_w", "frame_rate"]
    missing = [key for key in required if key not in config]
    if missing:
        raise KeyError(f"Robot config missing required keys: {missing}")
    return config

context_queue = []
context_size = None
goal_path = None
robo_pos = None
robo_orientation = None
rela_pos = None
closest_node = 0

def build_topomap_paths(topomap_root: str):
    root = Path(topomap_root)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"topomap-root not found or not a directory: {topomap_root}")

    png_files = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() == ".png"]
    png_files.sort(key=lambda p: p.name)

    if not png_files:
        raise RuntimeError(f"No .png files found under {topomap_root}")

    paths = [str(p.resolve()) for p in png_files]
    return paths

def image_callback(msg):
    if msg_to_pil is None:
        raise RuntimeError("Missing inference_utils.common.msg_to_pil; cannot convert ROS images.")
    img = msg_to_pil(msg)

    if context_size is not None:
        if len(context_queue) >= context_size + 1:
            context_queue.pop(0)
        context_queue.append(img)

def goal_path_callback(msg):
    global goal_path
    goal_path = msg.data
    rospy.loginfo("Received goal path: %s", goal_path)

def pos_callback(msg):
    global robo_pos, robo_orientation
    robo_pos = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
    robo_orientation = np.array([
        msg.pose.orientation.x,
        msg.pose.orientation.y,
        msg.pose.orientation.z,
        msg.pose.orientation.w
    ])

def main(args):
    global context_size, rela_pos, closest_node, goal_path

    _require_local_runtime_dependencies()

    robot_config = load_robot_config(args.robot_config)
    max_v = float(robot_config["max_v"])
    frame_rate = float(robot_config["frame_rate"])

    config, ckpt_path = load_config(args.model, args.config)
    config = inference_config_init(config, args)

    config["save_images"] = args.save_images
    config["save_path"] = args.save_path

    context_size = config["context_size"]

    TrainerCls = MODEL_REGISTRY[config["model_type"]]
    trainer = TrainerCls(config=config, checkpoint_path=ckpt_path)

    rospy.init_node("navigate_node", anonymous=False)
    rospy.Subscriber(args.image_topic, Image, image_callback, queue_size=1)

    rospy.Subscriber(args.goal_path_topic, String, goal_path_callback, queue_size=1)

    goal_img_pub = rospy.Publisher("/goal_image1", Image, queue_size=1)

    waypoint_pub = rospy.Publisher(args.waypoint_topic, Float32MultiArray, queue_size=1)

    subgoal_marker_pub = rospy.Publisher(args.subgoal_marker_topic, Marker, queue_size=1)
    goal_marker_pub = rospy.Publisher(args.goal_marker_topic, Marker, queue_size=1)

    marker_pub = rospy.Publisher(args.sampled_marker_topic, Marker, queue_size=10)
    sampled_actions_pub = rospy.Publisher(args.sampled_actions_topic, Float32MultiArray, queue_size=1)


    rate = rospy.Rate(config.get("frame_rate", frame_rate))

    scale = 10
    scale_factor = scale * max_v / frame_rate

    while not rospy.is_shutdown():
        if goal_path is None:
            try:
                rospy.loginfo_throttle(1.0, "Waiting for goal_path from topic '%s'..." % args.goal_path_topic)
            except Exception:
                rospy.loginfo("Waiting for goal_path from topic '%s'..." % args.goal_path_topic)
            rate.sleep()
            continue


        if len(context_queue) < context_size + 1:
            rate.sleep()
            continue

        obs_tensor = trainer.prepare_inputs(context_queue)

        goal_tensor = trainer.prepare_inputs([goal_path])

        if not os.path.exists(goal_path):
            rospy.logwarn(f"[goal_image] path not found: {goal_path}")
        else:
            try:
                rgb = np.asarray(PILImage.open(goal_path).convert("RGB"), dtype=np.uint8)  # HxWx3
                rgb = np.ascontiguousarray(rgb)
                H, W, _ = rgb.shape

                msg = Image()
                msg.header.stamp = rospy.Time.now()
                msg.header.frame_id = "map"
                msg.height = H
                msg.width  = W
                msg.encoding = "rgb8"
                msg.is_bigendian = 0
                msg.step = W * 3
                msg.data = rgb.tobytes(order="C")

                goal_img_pub.publish(msg)
            except Exception as e:
                rospy.logerr(f"[goal_image] publish error: {e}")



        # Inference: predict actions and waypoint distance.
        actions = trainer.action_inference(
            obs_tensor,
            goal_images=goal_tensor,
            num_samples=args.num_samples,
        )
        # print(actions.shape)
        chosen_waypoint = actions[0][args.waypoint]
        
        if config.get("normalize", False):
            chosen_waypoint[:2] *= (scale_factor / scale)

        msg = Float32MultiArray()
        msg.data = chosen_waypoint
        waypoint_pub.publish(msg)

        sampled_actions_msg = Float32MultiArray()
        flat_action = actions[0].flatten()
        sampled_actions_msg.data = np.concatenate(([0], flat_action))

        traj_pts = flat_action[:16].reshape(-1, 2) * scale_factor
        marker = create_marker_from_points(
            traj_pts,
            color=(1.0, 0.0, 0.0),
            scale=0.08,
            frame_id="base_link",
            z_value=0.0,
            marker_id=0,
            namespace="sampled_action_traj"
        )
        marker_pub.publish(marker)
        sampled_actions_pub.publish(sampled_actions_msg)

        rate.sleep()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="nomad")
    parser.add_argument("--config", type=str, default="../config/models.yaml")
    parser.add_argument("--robot-config", type=str, default="../config/robot.yaml")
    parser.add_argument("--topomap-root", type=str, default="../topomaps/images")
    parser.add_argument("--dir", type=str, default="collision_forward")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--waypoint", type=int, default=3)
    parser.add_argument("--pos-goal", action="store_true")
    parser.add_argument("--radius", type=int, default=3)
    parser.add_argument("--init-node", type=int, default=0)
    parser.add_argument("--goal-node", type=int, default=40)
    parser.add_argument("--image-topic", type=str, default="/isaac_node/camera0/image_raw")
    parser.add_argument("--goal-path-topic", type=str, default="/goal_path")
    parser.add_argument("--pos-topic", type=str, default="/model_position")
    parser.add_argument("--waypoint-topic", type=str, default="/waypoint")
    parser.add_argument("--subgoal-marker-topic", type=str, default="/goal")
    parser.add_argument("--goal-marker-topic", type=str, default="/topoplan/goal_marker")
    parser.add_argument("--close-threshold","-t",default=3,type=int,
                        help="""temporal distance within the next node in the topomap before localizing to it (default: 3)""",)
    parser.add_argument("--sampled-marker-topic", type=str,
                        default="/path")
    parser.add_argument("--sampled-actions-topic", type=str,
                        default="/sampled_actions")
    
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument("--save-path", type=str, default="outputs/local_planner_images", help="Image save path")
    
    args = parser.parse_args()
    main(args)
