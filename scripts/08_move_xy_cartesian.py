#!/usr/bin/python3
import copy
import json
import math
import sys

import geometry_msgs.msg
import moveit_commander
import rospy
import tf2_geometry_msgs  # noqa: F401
import tf2_ros


def get_camera_point(data):
    for p in (
        data.get("point_camera_m"),
        data.get("grasp_point_camera_m"),
        data.get("semantic_grasp", {}).get("point_camera_m"),
    ):
        if isinstance(p, dict) and all(k in p for k in ("x", "y", "z")):
            return float(p["x"]), float(p["y"]), float(p["z"])
    raise KeyError("No camera-frame 3-D grasp point found in result JSON")


def main():
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("vlm_move_xy_cartesian", anonymous=False)

    result_json = rospy.get_param("~result_json")
    camera_frame_override = rospy.get_param(
        "~camera_frame_override",
        "",
    )
    execute = bool(rospy.get_param("~execute", False))
    allow_motion = bool(rospy.get_param("~allow_motion", False))
    max_xy_motion = float(rospy.get_param("~max_xy_motion_m", 0.15))
    eef_step = float(rospy.get_param("~eef_step", 0.005))
    min_fraction = float(rospy.get_param("~min_fraction", 0.98))
    velocity_scale = float(rospy.get_param("~velocity_scale", 0.15))
    acceleration_scale = float(rospy.get_param("~acceleration_scale", 0.10))
    minimum_detected_base_z_m = float(
        rospy.get_param("~minimum_detected_base_z_m", 0.05)
    )
    thin_minimum_detected_base_z_m = float(
        rospy.get_param("~thin_minimum_detected_base_z_m", 0.02)
    )

    with open(result_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    thin_object_mode = bool(data.get("thin_object_mode", False))
    plate_edge_override_applied = bool(
        data.get("plate_edge_override_applied", False)
    )
    category_lower = str(
        data.get("category", "")
    ).strip().lower()
    planner_mode = str(
        data.get("planner_mode", "")
    ).strip().lower()

    ridge = data.get("stable_surface_ridge") or {}
    validated_boundary_grasp = bool(
        planner_mode in {"rim_pinch", "edge_pinch"}
        and ridge.get("edge_gate_passed") is True
        and ridge.get("radial_gate_passed") is True
        and ridge.get("cavity_depth_gate_passed") is True
    )

    # Use the lower base-Z floor for any geometrically validated boundary
    # grasp, regardless of semantic object category. This avoids arbitrary
    # plate/cup/bowl special cases while still requiring the strict edge,
    # radial, and cavity/surface validation gates to pass.
    low_profile_edge_mode = bool(
        thin_object_mode
        or plate_edge_override_applied
        or validated_boundary_grasp
    )

    active_minimum_base_z_m = (
        thin_minimum_detected_base_z_m
        if low_profile_edge_mode
        else minimum_detected_base_z_m
    )

    rospy.logwarn(
        "Base-Z safety mode=%s minimum=%.4f m "
        "(category=%s planner=%s thin=%s plate_override=%s "
        "validated_boundary=%s)",
        (
            "low_profile_edge"
            if low_profile_edge_mode
            else "normal_object"
        ),
        active_minimum_base_z_m,
        category_lower,
        planner_mode,
        thin_object_mode,
        plate_edge_override_applied,
        validated_boundary_grasp,
    )

    result_frame = str(data.get("camera_frame", "")).strip()
    if result_frame:
        camera_frame = result_frame
        if (
            camera_frame_override
            and camera_frame_override != result_frame
        ):
            rospy.logwarn(
                "Ignoring camera_frame_override=%s because result JSON "
                "point is explicitly expressed in %s",
                camera_frame_override,
                result_frame,
            )
    elif camera_frame_override:
        camera_frame = camera_frame_override
    else:
        raise RuntimeError(
            "Neither result JSON camera_frame nor override is available"
        )

    cx, cy, cz = get_camera_point(data)

    group = moveit_commander.MoveGroupCommander("arm")
    group.set_max_velocity_scaling_factor(velocity_scale)
    group.set_max_acceleration_scaling_factor(acceleration_scale)
    planning_frame = group.get_planning_frame()

    tf_buffer = tf2_ros.Buffer(rospy.Duration(10.0))
    listener = tf2_ros.TransformListener(tf_buffer)

    point = geometry_msgs.msg.PointStamped()
    point.header.frame_id = camera_frame
    point.header.stamp = rospy.Time(0)
    point.point.x, point.point.y, point.point.z = cx, cy, cz

    rospy.loginfo("Waiting for TF: %s -> %s", camera_frame, planning_frame)
    transformed = tf_buffer.transform(point, planning_frame, rospy.Duration(5.0))

    # Freeze the detected target in the robot base/planning frame BEFORE
    # the wrist-mounted camera moves. Later stages must reuse this fixed point.
    data["point_base_m"] = {
        "x": float(transformed.point.x),
        "y": float(transformed.point.y),
        "z": float(transformed.point.z),
    }
    data["planning_frame"] = planning_frame
    with open(result_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    rospy.loginfo(
        "Frozen target in %s: x=%.4f y=%.4f z=%.4f",
        planning_frame,
        transformed.point.x,
        transformed.point.y,
        transformed.point.z,
    )

    # Stop before any robot motion when the transformed surface point is at
    # or below the table/base safety floor. Do not silently clamp it.
    if transformed.point.z < active_minimum_base_z_m:
        raise RuntimeError(
            "Unsafe detected grasp Z in {}: {:.4f} m is below "
            "active minimum {:.4f} m for {} mode. "
            "Depth/rim candidate rejected before XY motion.".format(
                planning_frame,
                transformed.point.z,
                active_minimum_base_z_m,
                (
                    "low_profile_edge"
                    if low_profile_edge_mode
                    else "normal_object"
                ),
            )
        )

    current = group.get_current_pose().pose
    dx = transformed.point.x - current.position.x
    dy = transformed.point.y - current.position.y
    distance = math.hypot(dx, dy)

    rospy.logwarn(
        "XY-only displacement: dx=%.4f dy=%.4f distance=%.4f m",
        dx, dy, distance,
    )

    if distance > max_xy_motion:
        raise RuntimeError(
            f"Requested XY motion {distance:.4f} m exceeds limit {max_xy_motion:.4f} m"
        )

    count = max(2, int(math.ceil(distance / max(eef_step, 1e-4))) + 1)
    waypoints = []

    for i in range(1, count + 1):
        t = i / count
        pose = copy.deepcopy(current)
        pose.position.x = current.position.x + dx * t
        pose.position.y = current.position.y + dy * t
        pose.position.z = current.position.z
        waypoints.append(pose)

    plan, fraction = group.compute_cartesian_path(
        waypoints,
        eef_step,
        avoid_collisions=True,
    )

    rospy.loginfo("Cartesian XY path fraction: %.3f", fraction)

    if fraction < min_fraction:
        raise RuntimeError(f"XY path incomplete: fraction={fraction:.3f}")

    if not execute or not allow_motion:
        rospy.logwarn("DRY RUN ONLY: XY movement planned, not executed.")
        return

    rospy.logwarn("Executing XY-only movement...")
    ok = group.execute(plan, wait=True)
    group.stop()
    group.clear_pose_targets()

    if not ok:
        raise RuntimeError("XY execution failed")

    rospy.loginfo("XY movement complete.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        rospy.logerr("%s", exc)
        sys.exit(1)
