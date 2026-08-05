#!/usr/bin/env python3
import os
import sys
import cv2
import rospy
import rospkg
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


def default_output_path():
    package_path = rospkg.RosPack().get_path("vlm_gen3_basic")
    return os.path.join(package_path, "vlm", "data", "captured_rgb.jpg")


def main():
    rospy.init_node("vlm_gen3_capture_rgb", anonymous=False)
    image_topic = rospy.get_param("~image_topic", "/camera/color/image_raw")
    output_path = rospy.get_param("~output_path", default_output_path())
    timeout_sec = float(rospy.get_param("~timeout_sec", 15.0))

    rospy.loginfo("Waiting for one RGB frame on %s ...", image_topic)
    msg = rospy.wait_for_message(image_topic, Image, timeout=timeout_sec)

    try:
        image = CvBridge().imgmsg_to_cv2(msg, desired_encoding="bgr8")
    except CvBridgeError as exc:
        rospy.logerr("cv_bridge failed: %s", exc)
        return 1

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if not cv2.imwrite(output_path, image):
        rospy.logerr("Failed to save RGB image: %s", output_path)
        return 1

    h, w = image.shape[:2]
    rospy.loginfo("Saved RGB image: %s", output_path)
    rospy.loginfo("Image size: %dx%d; frame_id=%s", w, h, msg.header.frame_id)
    print(output_path)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except rospy.ROSException as exc:
        rospy.logerr("Timed out waiting for RGB image: %s", exc)
        sys.exit(2)
