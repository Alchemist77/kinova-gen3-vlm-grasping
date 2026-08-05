#!/usr/bin/env python3
"""Move the Kinova Gen3 arm to the saved tutorial home pose.

This node controls only the seven arm joints. It does not command the gripper.
"""

import sys
from typing import Any, List

import moveit_commander
import rospy
from sensor_msgs.msg import JointState


ARM_JOINT_NAMES = [f"joint_{index}" for index in range(1, 8)]


def extract_plan(plan_result: Any):
    """Support both old and new MoveIt Python plan() return formats."""
    if isinstance(plan_result, tuple):
        if len(plan_result) < 2:
            return False, None
        success = bool(plan_result[0])
        trajectory = plan_result[1]
        return success, trajectory

    trajectory = plan_result
    points = getattr(
        getattr(trajectory, "joint_trajectory", None), "points", []
    )
    return bool(points), trajectory


def read_current_arm_positions(topic: str, timeout: float) -> List[float]:
    msg = rospy.wait_for_message(topic, JointState, timeout=timeout)
    lookup = dict(zip(msg.name, msg.position))
    missing = [name for name in ARM_JOINT_NAMES if name not in lookup]
    if missing:
        raise RuntimeError("Missing joints in JointState: " + ", ".join(missing))
    return [float(lookup[name]) for name in ARM_JOINT_NAMES]


def main() -> int:
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("vlm_gen3_move_home", anonymous=False)

    group_name = rospy.get_param("~move_group_name", "arm")
    joint_state_topic = rospy.get_param("~joint_state_topic", "/joint_states")
    home = rospy.get_param("~home_joint_positions", [])
    tolerance = float(rospy.get_param("~home_tolerance_rad", 0.02))
    velocity = float(rospy.get_param("~velocity_scale", 0.08))
    acceleration = float(rospy.get_param("~acceleration_scale", 0.08))
    start_delay = float(rospy.get_param("~start_delay_sec", 8.0))

    if len(home) != 7:
        rospy.logerr("home_joint_positions must contain exactly 7 values; got %d", len(home))
        return 2

    rospy.logwarn("Automatic home movement is enabled.")
    rospy.logwarn("The arm will move after %.1f seconds. Keep the emergency stop ready.", start_delay)
    rospy.sleep(start_delay)

    try:
        current = read_current_arm_positions(joint_state_topic, timeout=15.0)
    except Exception as exc:
        rospy.logerr("Could not read current arm state from %s: %s", joint_state_topic, exc)
        return 3

    errors = [abs(a - b) for a, b in zip(current, home)]
    rospy.loginfo("Maximum home-joint error: %.6f rad", max(errors))
    if max(errors) <= tolerance:
        rospy.loginfo("Robot is already at the saved home pose. No movement required.")
        return 0

    try:
        group = moveit_commander.MoveGroupCommander(group_name)
    except Exception as exc:
        rospy.logerr("Could not create MoveIt group '%s': %s", group_name, exc)
        return 4

    active_joints = list(group.get_active_joints())
    rospy.loginfo("MoveIt group '%s' active joints: %s", group_name, active_joints)
    if len(active_joints) != 7:
        rospy.logerr(
            "MoveIt group '%s' must have 7 active joints, but has %d. "
            "Set move_group_name to the correct arm group.",
            group_name,
            len(active_joints),
        )
        return 5

    group.set_max_velocity_scaling_factor(max(0.01, min(velocity, 1.0)))
    group.set_max_acceleration_scaling_factor(max(0.01, min(acceleration, 1.0)))
    group.set_goal_joint_tolerance(tolerance)
    group.set_start_state_to_current_state()
    group.set_joint_value_target([float(value) for value in home])

    rospy.loginfo("Planning movement to the saved home pose...")
    success, trajectory = extract_plan(group.plan())
    if not success or trajectory is None:
        rospy.logerr("MoveIt could not find a valid trajectory to the home pose.")
        group.clear_pose_targets()
        return 6

    point_count = len(trajectory.joint_trajectory.points)
    rospy.loginfo("Plan ready with %d trajectory points. Executing at low speed...", point_count)
    executed = bool(group.execute(trajectory, wait=True))
    group.stop()
    group.clear_pose_targets()

    if not executed:
        rospy.logerr("MoveIt reported that home execution failed.")
        return 7

    try:
        final = read_current_arm_positions(joint_state_topic, timeout=5.0)
        final_errors = [abs(a - b) for a, b in zip(final, home)]
        rospy.loginfo("Final maximum home-joint error: %.6f rad", max(final_errors))
        if max(final_errors) > tolerance:
            rospy.logwarn("Movement completed, but the final pose is outside the configured tolerance.")
            return 8
    except Exception as exc:
        rospy.logwarn("Movement completed, but final verification failed: %s", exc)

    rospy.loginfo("Kinova Gen3 is at the saved tutorial home pose.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except rospy.ROSInterruptException:
        sys.exit(1)
