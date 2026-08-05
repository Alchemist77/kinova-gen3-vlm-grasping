#!/usr/bin/python3
"""
Stage 09: move downward to a relative pre-grasp position.

Moves the Kinova Gen3 end effector only along Z with Cartesian interpolation.

Keeps unchanged:
- X
- Y
- end-effector orientation
- gripper state

Convention:
- negative relative_z_m moves downward
- positive relative_z_m moves upward
"""

import json
import math
import os
import sys
from copy import deepcopy

import geometry_msgs.msg
import moveit_commander
import rospy
import rospkg
import tf2_geometry_msgs  # noqa: F401
import tf2_ros


def interpolate_z(start_pose, target_z, step_m):
    dz = target_z - start_pose.position.z
    distance = abs(dz)
    steps = max(1, int(math.ceil(distance / step_m)))

    waypoints = []
    for i in range(1, steps + 1):
        alpha = float(i) / float(steps)
        pose = deepcopy(start_pose)
        pose.position.z = start_pose.position.z + alpha * dz
        waypoints.append(pose)

    return waypoints, distance


def main():
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("vlm_gen3_move_z_pregrasp", anonymous=False)

    move_group_name = rospy.get_param("~move_group", "arm")
    relative_z = float(rospy.get_param("~relative_z_m", -0.01))
    use_detected_grasp_z = bool(rospy.get_param("~use_detected_grasp_z", False))
    result_json = rospy.get_param(
        "~result_json",
        os.path.join(rospkg.RosPack().get_path("vlm_gen3_basic"), "vlm", "results", "result.json"),
    )
    camera_frame_override = str(rospy.get_param(
        "~camera_frame_override", "camera_color_optical_frame"
    )).strip()
    planning_frame = str(rospy.get_param("~planning_frame", "")).strip()
    transform_timeout = float(rospy.get_param("~transform_timeout_sec", 5.0))
    pregrasp_offset = float(rospy.get_param("~pregrasp_offset_m", 0.10))
    max_z_motion = float(rospy.get_param("~max_z_motion_m", 0.05))
    minimum_target_z = float(rospy.get_param("~minimum_target_z_m", 0.10))

    eef_step = float(rospy.get_param("~eef_step_m", 0.003))
    min_fraction = float(rospy.get_param("~min_path_fraction", 0.98))
    velocity_scale = float(rospy.get_param("~velocity_scale", 0.03))
    acceleration_scale = float(rospy.get_param("~acceleration_scale", 0.03))
    safety_delay = max(0.0, float(rospy.get_param("~safety_delay_sec", 3.0)))

    execute = bool(rospy.get_param("~execute", False))
    allow_motion = bool(rospy.get_param("~allow_motion", False))

    if eef_step <= 0.0:
        raise RuntimeError("eef_step_m must be greater than zero")

    group = moveit_commander.MoveGroupCommander(move_group_name)
    group.set_max_velocity_scaling_factor(max(0.01, min(velocity_scale, 1.0)))
    group.set_max_acceleration_scaling_factor(max(0.01, min(acceleration_scale, 1.0)))

    current_pose = group.get_current_pose().pose

    if use_detected_grasp_z:
        if not os.path.isfile(result_json):
            raise RuntimeError("Result JSON not found: {}".format(result_json))
        with open(result_json, "r") as f:
            result = json.load(f)
        if not planning_frame:
            planning_frame = group.get_planning_frame()

        frozen = result.get("point_base_m")
        if isinstance(frozen, dict) and all(k in frozen for k in ("x", "y", "z")):
            grasp_z_base = float(frozen["z"])
            rospy.loginfo(
                "Using frozen grasp target from result JSON in %s. "
                "Camera movement will not change the target.",
                result.get("planning_frame", planning_frame),
            )
        else:
            point = result.get("point_camera_m")
            if not isinstance(point, dict):
                raise RuntimeError("Missing point_camera_m and point_base_m in result JSON")
            source_frame = camera_frame_override or str(result.get("camera_frame", "")).strip()
            if not source_frame:
                raise RuntimeError("No camera frame available")

            point_camera = geometry_msgs.msg.PointStamped()
            point_camera.header.stamp = rospy.Time(0)
            point_camera.header.frame_id = source_frame
            point_camera.point.x = float(point["x"])
            point_camera.point.y = float(point["y"])
            point_camera.point.z = float(point["z"])

            tf_buffer = tf2_ros.Buffer(rospy.Duration(20.0))
            tf_listener = tf2_ros.TransformListener(tf_buffer)
            transform = tf_buffer.lookup_transform(
                planning_frame, source_frame, rospy.Time(0), rospy.Duration(transform_timeout)
            )
            point_base = tf2_geometry_msgs.do_transform_point(point_camera, transform)
            grasp_z_base = float(point_base.point.z)
            rospy.logwarn(
                "No frozen point_base_m found; transformed the camera point now."
            )

        target_z = grasp_z_base + pregrasp_offset
        relative_z = target_z - current_pose.position.z
        rospy.loginfo(
            "Detected grasp Z in %s: %.4f m; pre-grasp offset: %.4f m",
            planning_frame, grasp_z_base, pregrasp_offset,
        )
    else:
        target_z = current_pose.position.z + relative_z

    if abs(relative_z) < 1e-6:
        rospy.logwarn("Already at requested pre-grasp Z; no Z movement required.")
        return 0

    if abs(relative_z) > max_z_motion:
        raise RuntimeError(
            "Requested Z motion {:.4f} m exceeds max_z_motion_m {:.4f} m"
            .format(abs(relative_z), max_z_motion)
        )

    rospy.loginfo(
        "Current EE: x=%.4f y=%.4f z=%.4f",
        current_pose.position.x,
        current_pose.position.y,
        current_pose.position.z,
    )
    rospy.logwarn(
        "Pre-grasp target: x=%.4f y=%.4f z=%.4f",
        current_pose.position.x,
        current_pose.position.y,
        target_z,
    )
    rospy.logwarn("Requested relative Z displacement: dz=%.4f m", relative_z)

    if target_z < minimum_target_z:
        raise RuntimeError(
            "Target Z {:.4f} m is below minimum_target_z_m {:.4f} m"
            .format(target_z, minimum_target_z)
        )

    waypoints, distance = interpolate_z(current_pose, target_z, eef_step)
    rospy.loginfo(
        "Generated %d Cartesian Z waypoints over %.4f m",
        len(waypoints),
        distance,
    )

    plan, fraction = group.compute_cartesian_path(
        waypoints,
        eef_step,
        avoid_collisions=True,
    )

    rospy.loginfo("Cartesian path fraction: %.3f", fraction)

    if fraction < min_fraction:
        raise RuntimeError(
            "Cartesian path fraction {:.3f} is below required {:.3f}"
            .format(fraction, min_fraction)
        )

    rospy.logwarn("Preview successful. X, Y, orientation, and gripper remain unchanged.")

    if not execute:
        rospy.logwarn("DRY RUN ONLY: execute=false. No movement commanded.")
        return 0

    if not allow_motion:
        raise RuntimeError(
            "execute=true but allow_motion=false. Both must be true."
        )

    if safety_delay > 0.0:
        rospy.logwarn("ROBOT WILL MOVE TO PRE-GRASP AFTER %.2f SECONDS.", safety_delay)
        rospy.logwarn("Keep the emergency stop ready.")
    else:
        rospy.logwarn("ROBOT WILL MOVE TO PRE-GRASP NOW. No safety delay is configured.")

    success = group.execute(plan, wait=True)
    group.stop()
    group.clear_pose_targets()

    if not success:
        raise RuntimeError("MoveIt Cartesian execution failed")

    final_pose = group.get_current_pose().pose
    rospy.loginfo(
        "Execution complete. Final EE: x=%.4f y=%.4f z=%.4f",
        final_pose.position.x,
        final_pose.position.y,
        final_pose.position.z,
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        rospy.logerr("%s", exc)
        sys.exit(1)
    finally:
        moveit_commander.roscpp_shutdown()
