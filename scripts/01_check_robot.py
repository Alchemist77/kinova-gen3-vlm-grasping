#!/usr/bin/env python3
"""Read-only Kinova connection diagnostic.

This node subscribes to JointState only. It never publishes robot commands.
Compatible with Python 3 / ROS Noetic.
"""

import sys
import rospy
from sensor_msgs.msg import JointState


def main() -> int:
    rospy.init_node("vlm_gen3_robot_check", anonymous=False)
    topic = rospy.get_param("~joint_state_topic", "/my_gen3/joint_states")
    timeout = float(rospy.get_param("~timeout", 20.0))

    rospy.loginfo("Waiting for Kinova joint states on: %s", topic)
    rospy.logwarn("This diagnostic is read-only and sends no motion command.")

    try:
        msg = rospy.wait_for_message(topic, JointState, timeout=timeout)
    except rospy.ROSException:
        rospy.logerr("No JointState received within %.1f seconds.", timeout)
        rospy.logerr("Find the real topic with: rostopic list | grep joint_states")
        return 1

    if not msg.name:
        rospy.logwarn("JointState received, but it contains no joint names.")
    else:
        rospy.loginfo("Robot connection OK. Received %d joints.", len(msg.name))
        for name, position in zip(msg.name, msg.position):
            rospy.loginfo("  %-25s %.6f rad", name, position)

    rospy.loginfo("Current robot pose has not been changed by this node.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except rospy.ROSInterruptException:
        sys.exit(0)
