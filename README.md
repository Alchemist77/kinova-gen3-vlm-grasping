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
