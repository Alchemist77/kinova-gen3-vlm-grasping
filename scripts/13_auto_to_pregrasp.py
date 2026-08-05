#!/usr/bin/env python3
"""Run capture -> VLM detection -> depth grasp -> XY alignment -> absolute pre-grasp.

This pipeline stops at the pre-grasp pose. It does not perform the final grasp
approach, close the gripper, or lift the object.
"""

import shlex
import subprocess
import sys
from pathlib import Path

import rospy
import rospkg


def run_stage(label, command):
    rospy.loginfo("========== %s ==========", label)
    rospy.loginfo("Running: %s", " ".join(shlex.quote(str(x)) for x in command))
    result = subprocess.run(command)
    if result.returncode != 0:
        raise RuntimeError("{} failed with exit code {}".format(label, result.returncode))


def bool_text(value):
    return "true" if bool(value) else "false"


def main():
    rospy.init_node("vlm_gen3_auto_to_pregrasp", anonymous=False)

    package_root = Path(rospkg.RosPack().get_path("vlm_gen3_basic"))
    image_path = package_root / "vlm" / "data" / "captured_rgb.jpg"
    object_json = package_root / "vlm" / "results" / "object_analysis.json"
    result_json = package_root / "vlm" / "results" / "result.json"
    output_image = package_root / "vlm" / "results" / "validated_grasp.jpg"
    clean_output_image = (
        package_root / "vlm" / "results" / "validated_grasp_clean.jpg"
    )
    debug_output_image = (
        package_root / "vlm" / "results" / "validated_grasp_debug.jpg"
    )
    info_output_text = (
        package_root / "vlm" / "results" / "validated_grasp_info.txt"
    )

    target = str(rospy.get_param("~target", "yellow bowl")).strip()
    if not target:
        raise RuntimeError("target cannot be empty. Example: target:=\"yellow bowl\"")

    model_path = str(rospy.get_param(
        "~model_path", "/home/abr-lab/vlm/Qwen3-VL-2B-Instruct"
    ))
    python_path = str(rospy.get_param(
        "~vlm_python", "/home/abr-lab/vlm/qwen_env/bin/python"
    ))

    image_topic = str(rospy.get_param("~image_topic", "/camera/color/image_raw"))
    depth_topic = str(rospy.get_param(
        "~depth_topic", "/camera/depth/image_rect_raw"
    ))
    camera_info_topic = str(rospy.get_param(
        "~camera_info_topic", "/camera/color/camera_info"
    ))
    camera_frame = str(rospy.get_param(
        "~camera_frame_override", "camera_color_optical_frame"
    ))

    execute = bool(rospy.get_param("~execute", False))
    allow_motion = bool(rospy.get_param("~allow_motion", False))
    pregrasp_offset = float(rospy.get_param("~pregrasp_offset_m", 0.10))
    safety_delay = float(rospy.get_param("~safety_delay_sec", 0.0))

    max_xy_motion = float(rospy.get_param("~max_xy_motion_m", 0.40))
    max_z_motion = float(rospy.get_param("~max_z_motion_m", 0.30))
    minimum_target_z = float(rospy.get_param("~minimum_target_z_m", 0.05))
    skip_motion_stages = bool(rospy.get_param("~skip_motion_stages", False))

    # Bowl/open-top rim grasp parameters.
    rim_grasp_enabled = bool(rospy.get_param("~rim_grasp_enabled", True))
    rim_inner_ratio = float(rospy.get_param("~rim_inner_ratio", 0.80))
    rim_outer_ratio = float(rospy.get_param("~rim_outer_ratio", 0.96))
    rim_inset_px = int(rospy.get_param("~rim_inset_px", 5))
    surface_standoff_m = float(rospy.get_param("~surface_standoff_m", 0.0))
    depth_frame_count = max(
        1, int(rospy.get_param("~depth_frame_count", 7))
    )
    depth_frame_interval_sec = max(
        0.0, float(rospy.get_param("~depth_frame_interval_sec", 0.06))
    )
    cavity_min_frame_support = max(
        1, int(rospy.get_param("~cavity_min_frame_support", 2))
    )

    # Visualization outputs.
    save_clean_grasp_image = bool(
        rospy.get_param("~save_clean_grasp_image", True)
    )
    save_debug_grasp_image = bool(
        rospy.get_param("~save_debug_grasp_image", True)
    )
    save_grasp_info_text = bool(
        rospy.get_param("~save_grasp_info_text", True)
    )

    if not (0.0 <= rim_inner_ratio < rim_outer_ratio <= 1.0):
        raise RuntimeError(
            "Require 0 <= rim_inner_ratio < rim_outer_ratio <= 1"
        )

    if surface_standoff_m < 0.0:
        raise RuntimeError("surface_standoff_m cannot be negative")

    if camera_frame != "camera_color_optical_frame":
        rospy.logwarn(
            "Expected camera_color_optical_frame for pixel-to-3D output, "
            "but received '%s'.",
            camera_frame,
        )

    rospy.logwarn("Target object supplied to Qwen: '%s'", target)
    rospy.logwarn("Pipeline stops %.3f m above the detected grasp point.", pregrasp_offset)
    rospy.loginfo(
        "Rim grasp: enabled=%s, radial band=%.2f..%.2f, inset=%d px",
        rim_grasp_enabled, rim_inner_ratio, rim_outer_ratio, rim_inset_px
    )
    rospy.loginfo("RGB topic: %s", image_topic)
    rospy.loginfo("Depth topic: %s", depth_topic)
    rospy.loginfo("Camera info topic: %s", camera_info_topic)
    rospy.loginfo("3D camera frame: %s", camera_frame)
    rospy.loginfo(
        "Surface standoff subtracted from measured depth: %.3f m",
        surface_standoff_m,
    )
    if execute and allow_motion:
        rospy.logwarn("AUTOMATIC ROBOT MOTION IS ENABLED. Keep the emergency stop ready.")
    else:
        rospy.logwarn("Preview mode: robot motion is disabled.")

    run_stage("1/5 Capture RGB", [
        "rosrun", "vlm_gen3_basic", "05_capture_rgb.py",
        "_image_topic:={}".format(image_topic),
        "_output_path:={}".format(image_path),
        "_timeout_sec:=15.0",
    ])

    run_stage("2/5 Qwen target detection", [
        python_path,
        str(package_root / "vlm" / "06_vlm_detect.py"),
        "--image", str(image_path),
        "--target", target,
        "--model", model_path,
        "--output-dir", str(object_json.parent),
    ])

    run_stage("3/5 Semantic depth grasp point", [
        "rosrun", "vlm_gen3_basic", "07_semantic_depth_grasp.py",
        "_object_json:={}".format(object_json),
        "_result_json:={}".format(result_json),
        "_output_image:={}".format(output_image),
        "_clean_output_image:={}".format(clean_output_image),
        "_debug_output_image:={}".format(debug_output_image),
        "_info_output_text:={}".format(info_output_text),
        "_depth_topic:={}".format(depth_topic),
        "_camera_info_topic:={}".format(camera_info_topic),
        "_timeout_sec:=15.0",
        "_rim_grasp_enabled:={}".format(bool_text(rim_grasp_enabled)),
        "_rim_inner_ratio:={}".format(rim_inner_ratio),
        "_rim_outer_ratio:={}".format(rim_outer_ratio),
        "_rim_inset_px:={}".format(rim_inset_px),
        "_surface_standoff_m:={}".format(surface_standoff_m),
        "_depth_frame_count:={}".format(depth_frame_count),
        "_depth_frame_interval_sec:={}".format(depth_frame_interval_sec),
        "_cavity_min_frame_support:={}".format(cavity_min_frame_support),
        "_save_clean_grasp_image:={}".format(
            bool_text(save_clean_grasp_image)
        ),
        "_save_debug_grasp_image:={}".format(
            bool_text(save_debug_grasp_image)
        ),
        "_save_grasp_info_text:={}".format(
            bool_text(save_grasp_info_text)
        ),
    ])

    if skip_motion_stages:
        rospy.loginfo("DETECTION-ONLY COMPLETE. Motion stages were intentionally skipped.")
        return 0

    run_stage("4/5 Move X/Y", [
        "rosrun", "vlm_gen3_basic", "08_move_xy_cartesian.py",
        "_result_json:={}".format(result_json),
        "_camera_frame_override:={}".format(camera_frame),
        "_execute:={}".format(bool_text(execute)),
        "_allow_motion:={}".format(bool_text(allow_motion)),
        "_max_xy_motion_m:={}".format(max_xy_motion),
        "_safety_delay_sec:={}".format(safety_delay),
    ])

    run_stage("5/5 Move to absolute pre-grasp Z", [
        "rosrun", "vlm_gen3_basic", "09_move_z_pregrasp.py",
        "_use_detected_grasp_z:=true",
        "_result_json:={}".format(result_json),
        "_camera_frame_override:={}".format(camera_frame),
        "_pregrasp_offset_m:={}".format(pregrasp_offset),
        "_execute:={}".format(bool_text(execute)),
        "_allow_motion:={}".format(bool_text(allow_motion)),
        "_max_z_motion_m:={}".format(max_z_motion),
        "_minimum_target_z_m:={}".format(minimum_target_z),
        "_safety_delay_sec:={}".format(safety_delay),
    ])

    rospy.loginfo("AUTO-TO-PREGRASP COMPLETE.")
    rospy.loginfo("Robot stopped at pre-grasp. Gripper was not closed.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as exc:
        rospy.logerr("AUTO-TO-PREGRASP FAILED: %s", exc)
        sys.exit(1)
