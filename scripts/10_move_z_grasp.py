#!/usr/bin/python3
import copy
import math
import sys

import moveit_commander
import rospy


def main():
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("vlm_move_to_grasp_z", anonymous=False)

    pregrasp_offset = float(rospy.get_param("~pregrasp_offset_m", 0.10))
    grasp_clearance = float(rospy.get_param("~grasp_clearance_m", 0.04))
    execute = bool(rospy.get_param("~execute", False))
    allow_motion = bool(rospy.get_param("~allow_motion", False))
    max_z_motion = float(rospy.get_param("~max_z_motion_m", 0.15))
    eef_step = float(rospy.get_param("~eef_step", 0.003))
    min_fraction = float(rospy.get_param("~min_fraction", 0.98))

    descent = pregrasp_offset - grasp_clearance

    if descent <= 0.0:
        raise RuntimeError(
            "pregrasp_offset_m must be larger than grasp_clearance_m"
        )

    if descent > max_z_motion:
        raise RuntimeError(
            f"Final grasp descent {descent:.4f} m exceeds limit {max_z_motion:.4f} m"
        )

    group = moveit_commander.MoveGroupCommander("arm")
    current = group.get_current_pose().pose
    target_z = current.position.z - descent

    rospy.loginfo("Current EE Z: %.4f m", current.position.z)
    rospy.logwarn(
        "Relative final grasp descent: %.4f m "
        "(pregrasp %.4f - clearance %.4f)",
        descent,
        pregrasp_offset,
        grasp_clearance,
    )
    rospy.logwarn("Target EE Z: %.4f m", target_z)

    count = max(2, int(math.ceil(descent / max(eef_step, 1e-4))) + 1)
    waypoints = []

    for i in range(1, count + 1):
        pose = copy.deepcopy(current)
        pose.position.z = current.position.z - descent * (i / count)
        waypoints.append(pose)

    plan, fraction = group.compute_cartesian_path(
        waypoints,
        eef_step,
        avoid_collisions=True,
    )

    rospy.loginfo("Cartesian final approach fraction: %.3f", fraction)

    if fraction < min_fraction:
        raise RuntimeError(
            f"Final grasp approach incomplete: fraction={fraction:.3f}"
        )

    if not execute or not allow_motion:
        rospy.logwarn(
            "DRY RUN ONLY: final relative grasp approach planned, not executed."
        )
        return

    rospy.logwarn("Executing final relative grasp approach...")
    ok = group.execute(plan, wait=True)
    group.stop()
    group.clear_pose_targets()

    if not ok:
        raise RuntimeError("MoveIt failed to execute final grasp approach")

    rospy.loginfo("Reached final grasp pose.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        rospy.logerr("%s", exc)
        sys.exit(1)
