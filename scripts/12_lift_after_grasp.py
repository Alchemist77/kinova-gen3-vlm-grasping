#!/usr/bin/python3

import copy
import math
import sys
import threading

import moveit_commander
import rospy

from kortex_driver.srv import SendGripperCommand, SendGripperCommandRequest
from kortex_driver.msg import Finger, GripperMode


def make_gripper_request(close_value):
    request = SendGripperCommandRequest()
    request.input.mode = GripperMode.GRIPPER_POSITION

    finger = Finger()
    finger.finger_identifier = 0
    finger.value = close_value
    request.input.gripper.finger.append(finger)
    return request


def gripper_keep_closing_worker(stop_event, close_value, repeat_hz):
    """Keep re-sending the close target while the arm is lifting."""
    service_name = "/base/send_gripper_command"

    try:
        rospy.wait_for_service(service_name, timeout=10.0)
        send_command = rospy.ServiceProxy(
            service_name,
            SendGripperCommand,
            persistent=False,
        )
        request = make_gripper_request(close_value)
        period = 1.0 / max(0.5, repeat_hz)

        rospy.logwarn(
            "Gripper close keep-alive started: target=%.3f, rate=%.2f Hz",
            close_value,
            repeat_hz,
        )

        while not stop_event.is_set() and not rospy.is_shutdown():
            try:
                send_command(request)
            except rospy.ServiceException as exc:
                # A single missed service call must not immediately abort the arm lift.
                rospy.logwarn_throttle(
                    1.0,
                    "Gripper keep-alive service call failed: %s",
                    exc,
                )
            stop_event.wait(period)

        rospy.loginfo("Gripper close keep-alive stopped.")

    except rospy.ROSException as exc:
        rospy.logerr("Gripper service unavailable during lift: %s", exc)
    except Exception as exc:
        rospy.logerr("Unexpected gripper keep-alive error: %s", exc)


def send_final_close(close_value):
    """Send one final close target after the lift finishes."""
    service_name = "/base/send_gripper_command"
    try:
        rospy.wait_for_service(service_name, timeout=3.0)
        send_command = rospy.ServiceProxy(service_name, SendGripperCommand)
        send_command(make_gripper_request(close_value))
        rospy.loginfo("Final gripper close target sent after lift.")
    except Exception as exc:
        rospy.logwarn("Could not send final gripper close target: %s", exc)


def main():
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("vlm_lift_after_grasp", anonymous=False)

    lift_distance_m = float(rospy.get_param("~lift_distance_m", 0.15))
    execute = bool(rospy.get_param("~execute", False))
    allow_motion = bool(rospy.get_param("~allow_motion", False))
    max_lift_m = float(rospy.get_param("~max_lift_m", 0.25))
    eef_step = float(rospy.get_param("~eef_step", 0.003))
    min_fraction = float(rospy.get_param("~min_fraction", 0.98))

    keep_closing_during_lift = bool(
        rospy.get_param("~keep_closing_during_lift", True)
    )
    close_joint_value = float(rospy.get_param("~close_joint_value", 1.0))
    gripper_repeat_hz = float(rospy.get_param("~gripper_repeat_hz", 3.0))
    final_close_after_lift = bool(
        rospy.get_param("~final_close_after_lift", True)
    )

    close_joint_value = max(0.0, min(1.0, close_joint_value))
    gripper_repeat_hz = max(0.5, min(10.0, gripper_repeat_hz))

    if lift_distance_m <= 0.0 or lift_distance_m > max_lift_m:
        raise RuntimeError(
            f"Lift distance {lift_distance_m:.4f} m is outside safe range "
            f"(0, {max_lift_m:.4f}]"
        )

    group = moveit_commander.MoveGroupCommander("arm")
    current = group.get_current_pose().pose
    count = max(
        2,
        int(math.ceil(lift_distance_m / max(eef_step, 1e-4))) + 1,
    )

    waypoints = []
    for i in range(1, count + 1):
        pose = copy.deepcopy(current)
        pose.position.z = current.position.z + lift_distance_m * (i / count)
        waypoints.append(pose)

    rospy.logwarn(
        "Lift target: current z=%.4f -> target z=%.4f",
        current.position.z,
        current.position.z + lift_distance_m,
    )

    plan, fraction = group.compute_cartesian_path(
        waypoints,
        eef_step,
        avoid_collisions=True,
    )
    rospy.loginfo("Cartesian lift fraction: %.3f", fraction)

    if fraction < min_fraction:
        raise RuntimeError(f"Lift path incomplete: fraction={fraction:.3f}")

    if not execute or not allow_motion:
        rospy.logwarn("DRY RUN ONLY: lift was planned but not executed.")
        return

    stop_event = threading.Event()
    gripper_thread = None

    if keep_closing_during_lift:
        gripper_thread = threading.Thread(
            target=gripper_keep_closing_worker,
            args=(stop_event, close_joint_value, gripper_repeat_hz),
            daemon=True,
            name="gripper_keep_closing",
        )
        gripper_thread.start()

    try:
        rospy.logwarn(
            "Executing lift%s...",
            " while continuously commanding gripper close"
            if keep_closing_during_lift
            else "",
        )
        ok = group.execute(plan, wait=True)
    finally:
        stop_event.set()
        if gripper_thread is not None:
            gripper_thread.join(timeout=2.0)
        group.stop()
        group.clear_pose_targets()

    if not ok:
        raise RuntimeError("Lift execution failed")

    if final_close_after_lift:
        send_final_close(close_joint_value)

    rospy.loginfo("Lift complete.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        rospy.logerr("%s", exc)
        sys.exit(1)
