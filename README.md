# Kinova Gen3 VLM-Based Grasping

This project implements a vision-language-guided grasping pipeline for a Kinova Gen3 robot.

The system uses:

- Kinova Gen3 robot
- Robotiq 2F-85 gripper
- Kinova Vision RGB-D camera
- ROS Noetic
- MoveIt
- Qwen3-VL-2B-Instruct

The Vision-Language Model detects the requested object and suggests a grasp strategy.  
RGB-D geometry is then used to calculate and validate the final 3-D grasp point.

---

## System Overview

```text
RGB image
    ↓
Qwen3-VL target detection
    ↓
Object bounding box and grasp strategy
    ↓
Depth-based grasp candidate generation
    ↓
Edge, cavity, depth, and gripper-width validation
    ↓
3-D grasp point
    ↓
Robot motion, gripper closing, and object lifting

Robot Initial Setup

Start the robot, camera, MoveIt, home motion, and gripper initialization:

```bash
roslaunch vlm_gen3_basic initial_setup.launch
```

This launch file:

Starts the Kinova driver
Starts the Kinova Vision camera
Moves the robot to the saved home pose
Opens the gripper

Warning: This command may move the real robot.

Detection-Only Test

Always test the perception pipeline before enabling robot motion:

```bash
roslaunch vlm_gen3_basic auto_grasp_and_lift.launch \
  execute:=false \
  allow_motion:=false \
  target:="blue cup"
```
This runs:

RGB image capture
Qwen3-VL target detection
Depth-based grasp-point calculation
Grasp validation

The robot does not move.
