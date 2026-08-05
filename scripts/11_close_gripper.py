#!/usr/bin/env python3

import sys
import rospy

from kortex_driver.srv import (
    SendGripperCommand,
    SendGripperCommandRequest,
)
from kortex_driver.msg import Finger, GripperMode


def main():
    rospy.init_node("vlm_close_gripper")

    close_value = float(
        rospy.get_param("~close_joint_value", 1.0)
    )
    execute = rospy.get_param("~execute", False)
    allow_motion = rospy.get_param("~allow_motion", False)

    # Your Kortex driver exposes:
    # /base/send_gripper_command
    service_name = "/base/send_gripper_command"

    # Kortex normalized finger position:
    # 0.0 = fully open
    # 1.0 = fully closed
    close_value = max(0.0, min(1.0, close_value))

    rospy.loginfo("Gripper service: %s", service_name)
    rospy.logwarn(
        "Requested gripper position: %.3f",
        close_value,
    )

    if not execute:
        rospy.logwarn(
            "DRY RUN: execute is false. Gripper will not move."
        )
        return

    if not allow_motion:
        rospy.logwarn(
            "DRY RUN: allow_motion is false. Gripper will not move."
        )
        return

    try:
        rospy.loginfo(
            "Waiting for gripper service: %s",
            service_name,
        )
        rospy.wait_for_service(service_name, timeout=10.0)
    except rospy.ROSException:
        rospy.logerr(
            "Gripper service was not available after 10 seconds: %s",
            service_name,
        )
        sys.exit(1)

    try:
        send_command = rospy.ServiceProxy(
            service_name,
            SendGripperCommand,
        )

        request = SendGripperCommandRequest()
        request.input.mode = GripperMode.GRIPPER_POSITION

        finger = Finger()
        finger.finger_identifier = 0
        finger.value = close_value

        request.input.gripper.finger.append(finger)

        rospy.logwarn(
            "Sending Kortex native gripper command..."
        )

        response = send_command(request)

        rospy.loginfo(
            "Gripper command accepted by Kortex."
        )
        rospy.logdebug(
            "Service response: %s",
            response,
        )

        # Important:
        # This does not wait for the gripper to reach exactly 0.8.
        # Contact with an object is therefore not treated as a MoveIt error.

    except rospy.ServiceException as exc:
        rospy.logerr(
            "Kortex gripper service call failed: %s",
            exc,
        )
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
    except Exception as exc:
        rospy.logerr(
            "Unexpected gripper error: %s",
            exc,
        )
        sys.exit(1)
