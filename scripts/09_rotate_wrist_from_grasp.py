#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Rotate only Kinova Gen3 joint_7 by an angle computed from result.json.

This script does NOT use MoveIt.

It:
1. Reads grasp_pixel and bbox from result.json.
2. Computes the continuous grasp angle.
3. Reads the current 7 joint positions from /joint_states.
4. Keeps joints 1..6 unchanged.
5. Adds the computed relative angle only to joint_7.
6. Sends one FollowJointTrajectory goal.

Required ROS parameters:
  _result_json:=/path/result.json
  _action_name:=/YOUR_TRAJECTORY_CONTROLLER/follow_joint_trajectory

Safety:
  Default execute=false.
"""

import json
import math
import os
import sys

import actionlib
import rospy
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint


JOINT_NAMES = ["joint_{}".format(i) for i in range(1, 8)]


def normalize_axis_angle_deg(angle_deg):
    # Gripper closing direction is an undirected axis:
    # 0 deg and 180 deg are equivalent.
    while angle_deg > 90.0:
        angle_deg -= 180.0
    while angle_deg < -90.0:
        angle_deg += 180.0
    return angle_deg


def compute_relative_angle_deg(data, angle_sign, angle_offset_deg):
    grasp = data.get("grasp_pixel")
    bbox = data.get("bbox")

    if not isinstance(grasp, (list, tuple)) or len(grasp) != 2:
        raise RuntimeError("Missing/invalid grasp_pixel in result.json")

    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise RuntimeError("Missing/invalid bbox in result.json")

    u, v = float(grasp[0]), float(grasp[1])
    x1, y1, x2, y2 = map(float, bbox)

    center_u = 0.5 * (x1 + x2)
    center_v = 0.5 * (y1 + y2)

    dx = u - center_u
    dy = v - center_v

    if math.hypot(dx, dy) < 1.0:
        raise RuntimeError("Grasp point is too close to bbox center")

    radial_angle_deg = math.degrees(math.atan2(dy, dx))
    axis_angle_deg = normalize_axis_angle_deg(radial_angle_deg)

    command_deg = normalize_axis_angle_deg(
        angle_sign * axis_angle_deg + angle_offset_deg
    )

    return command_deg, radial_angle_deg, axis_angle_deg, dx, dy



def choose_joint7_target(current_rad, requested_deg, lower_rad, upper_rad):
    """
    The parallel gripper closing axis repeats every 180 degrees.
    Choose an equivalent relative rotation that keeps joint_7 within limits
    and requires the smallest motion.
    """
    candidates = []

    for k in range(-4, 5):
        candidate_deg = requested_deg + 180.0 * k
        candidate_rad = current_rad + math.radians(candidate_deg)

        if lower_rad <= candidate_rad <= upper_rad:
            candidates.append((abs(candidate_deg), candidate_deg, candidate_rad))

    if not candidates:
        raise RuntimeError(
            "No equivalent joint_7 target is inside limits "
            "[{:.3f}, {:.3f}] rad. current={:.3f} rad, requested={:.1f} deg".format(
                lower_rad, upper_rad, current_rad, requested_deg
            )
        )

    candidates.sort(key=lambda item: item[0])
    _magnitude, selected_deg, selected_rad = candidates[0]
    return selected_deg, selected_rad

def read_current_joint_positions(topic, timeout):
    msg = rospy.wait_for_message(topic, JointState, timeout=timeout)
    lookup = dict(zip(msg.name, msg.position))

    missing = [name for name in JOINT_NAMES if name not in lookup]
    if missing:
        raise RuntimeError(
            "Missing joints in {}: {}".format(topic, ", ".join(missing))
        )

    return [float(lookup[name]) for name in JOINT_NAMES]



def discover_trajectory_actions():
    """
    Discover FollowJointTrajectory action bases from published ROS topics.

    An action server publishes topics such as:
      <base>/status
      <base>/feedback
      <base>/result

    We derive <base> from those topic names.
    """
    candidates = set()

    for topic_name, _topic_type in rospy.get_published_topics():
        for suffix in ("/status", "/feedback", "/result"):
            if (
                topic_name.endswith(suffix)
                and "follow_joint_trajectory" in topic_name
            ):
                candidates.add(topic_name[:-len(suffix)])

    return sorted(candidates)


def connect_trajectory_action(requested_action_name, timeout_sec):
    """
    Try the requested action first, then auto-discover any available
    FollowJointTrajectory action server.
    """
    names_to_try = []

    if requested_action_name and requested_action_name.lower() != "auto":
        names_to_try.append(requested_action_name.rstrip("/"))

    for discovered in discover_trajectory_actions():
        if discovered not in names_to_try:
            names_to_try.append(discovered)

    common_names = [
        "/my_gen3/gen3_joint_trajectory_controller/follow_joint_trajectory",
        "/my_gen3/joint_trajectory_controller/follow_joint_trajectory",
        "/gen3_joint_trajectory_controller/follow_joint_trajectory",
        "/joint_trajectory_controller/follow_joint_trajectory",
    ]
    for name in common_names:
        if name not in names_to_try:
            names_to_try.append(name)

    rospy.loginfo(
        "FollowJointTrajectory candidates: %s",
        names_to_try if names_to_try else "none",
    )

    per_candidate_timeout = max(
        1.0,
        float(timeout_sec) / max(1, len(names_to_try)),
    )

    for name in names_to_try:
        rospy.loginfo("Trying trajectory action: %s", name)
        client = actionlib.SimpleActionClient(
            name,
            FollowJointTrajectoryAction,
        )
        if client.wait_for_server(rospy.Duration(per_candidate_timeout)):
            rospy.loginfo("Connected trajectory action: %s", name)
            return client, name

    raise RuntimeError(
        "No FollowJointTrajectory action server found. "
        "Run: rostopic list | grep follow_joint_trajectory"
    )

def main():
    rospy.init_node("rotate_joint7_from_grasp", anonymous=False)

    result_json = str(rospy.get_param(
        "~result_json",
        "/home/abr-lab/catkin_ws/src/vlm_gen3_basic/vlm/results/result.json",
    ))
    joint_state_topic = str(rospy.get_param("~joint_state_topic", "/joint_states"))
    action_name = str(rospy.get_param("~action_name", "auto"))
    action_wait_timeout_sec = float(
        rospy.get_param("~action_wait_timeout_sec", 12.0)
    )

    angle_sign = float(rospy.get_param("~angle_sign", -1.0))
    angle_offset_deg = float(rospy.get_param("~angle_offset_deg", 0.0))
    min_rotation_deg = float(rospy.get_param("~min_rotation_deg", 3.0))
    max_rotation_deg = float(rospy.get_param("~max_rotation_deg", 90.0))
    duration_sec = float(rospy.get_param("~duration_sec", 2.5))
    joint7_lower_limit_rad = float(
        rospy.get_param("~joint7_lower_limit_rad", -3.14159)
    )
    joint7_upper_limit_rad = float(
        rospy.get_param("~joint7_upper_limit_rad", 3.14159)
    )
    execute = bool(rospy.get_param("~execute", False))
    allow_motion = bool(rospy.get_param("~allow_motion", False))

    if angle_sign not in (-1.0, 1.0):
        raise RuntimeError("angle_sign must be +1.0 or -1.0")

    if not os.path.isfile(result_json):
        raise RuntimeError("result.json not found: {}".format(result_json))

    with open(result_json, "r", encoding="utf-8") as file:
        data = json.load(file)

    angle_deg, radial_deg, axis_deg, dx, dy = compute_relative_angle_deg(
        data,
        angle_sign,
        angle_offset_deg,
    )

    angle_deg = max(-max_rotation_deg, min(max_rotation_deg, angle_deg))

    rospy.logwarn(
        "Grasp geometry: dx=%.1f px dy=%.1f px radial=%.1f deg "
        "axis=%.1f deg -> joint_7 relative command=%.1f deg",
        dx, dy, radial_deg, axis_deg, angle_deg,
    )

    data["joint7_rotation_decision"] = {
        "dx_px": dx,
        "dy_px": dy,
        "radial_angle_deg": radial_deg,
        "axis_angle_deg": axis_deg,
        "command_deg": angle_deg,
        "angle_sign": angle_sign,
        "angle_offset_deg": angle_offset_deg,
    }
    with open(result_json, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)

    if abs(angle_deg) < min_rotation_deg:
        rospy.logwarn("Rotation skipped: |%.1f| < %.1f deg", angle_deg, min_rotation_deg)
        return 0

    current = read_current_joint_positions(joint_state_topic, timeout=5.0)

    selected_angle_deg, selected_target_rad = choose_joint7_target(
        current_rad=current[6],
        requested_deg=angle_deg,
        lower_rad=joint7_lower_limit_rad,
        upper_rad=joint7_upper_limit_rad,
    )

    if abs(selected_angle_deg - angle_deg) > 1e-6:
        rospy.logwarn(
            "Requested %.1f deg would exceed the configured joint_7 limits; "
            "using equivalent gripper-axis rotation %.1f deg.",
            angle_deg,
            selected_angle_deg,
        )

    angle_deg = selected_angle_deg
    target = list(current)
    target[6] = selected_target_rad

    rospy.logwarn(
        "joint_7: current=%.4f rad, target=%.4f rad, delta=%.4f rad (%.1f deg)",
        current[6],
        target[6],
        target[6] - current[6],
        angle_deg,
    )

    if not (execute and allow_motion):
        rospy.logwarn(
            "DRY RUN: no motion. Use _execute:=true _allow_motion:=true."
        )
        return 0

    client, connected_action_name = connect_trajectory_action(
        action_name,
        action_wait_timeout_sec,
    )

    goal = FollowJointTrajectoryGoal()
    goal.trajectory.joint_names = JOINT_NAMES

    # Give the controller an explicit start state and an end state.
    # Some Kinova trajectory controllers abort a one-point trajectory.
    start_point = JointTrajectoryPoint()
    start_point.positions = list(current)
    start_point.time_from_start = rospy.Duration(0.25)

    target_point = JointTrajectoryPoint()
    target_point.positions = target
    target_point.time_from_start = rospy.Duration(max(0.75, duration_sec))

    goal.trajectory.points = [start_point, target_point]
    goal.trajectory.header.stamp = rospy.Time.now() + rospy.Duration(0.25)

    rospy.logwarn(
        "Sending trajectory through %s: joints 1..6 fixed, only joint_7 changed.",
        connected_action_name,
    )
    client.send_goal(goal)

    if not client.wait_for_result(rospy.Duration(duration_sec + 5.0)):
        client.cancel_goal()
        raise RuntimeError("joint_7 trajectory timed out")

    result = client.get_result()
    state = client.get_state()
    status_text = client.get_goal_status_text()

    rospy.loginfo(
        "Trajectory action state=%s status_text=%r result=%s",
        state,
        status_text,
        result,
    )

    if state != actionlib.GoalStatus.SUCCEEDED:
        result_code = getattr(result, "error_code", None)
        result_string = getattr(result, "error_string", "")
        raise RuntimeError(
            "joint_7 trajectory failed: action_state={}, status_text={!r}, "
            "result_error_code={}, result_error_string={!r}".format(
                state,
                status_text,
                result_code,
                result_string,
            )
        )

    rospy.loginfo("joint_7 rotation completed.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except rospy.ROSInterruptException:
        sys.exit(130)
    except Exception as exc:
        rospy.logerr("JOINT_7 ROTATION FAILED: %s", exc)
        sys.exit(1)
