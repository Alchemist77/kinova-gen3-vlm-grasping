#!/usr/bin/env python3
"""Display the wrist-camera RGB stream without controlling the robot."""

import sys
import cv2
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image


class WristCameraViewer:
    def __init__(self) -> None:
        self.bridge = CvBridge()
        self.topic = rospy.get_param("~image_topic", "/camera/color/image_raw")
        self.window_name = "Kinova Wrist Camera - press q to close"
        self.received_first_image = False

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        self.subscriber = rospy.Subscriber(
            self.topic,
            Image,
            self.image_callback,
            queue_size=1,
            buff_size=2 ** 24,
        )
        rospy.on_shutdown(self.shutdown)
        rospy.loginfo("Waiting for wrist-camera images on: %s", self.topic)

    def image_callback(self, msg: Image) -> None:
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as exc:
            rospy.logerr_throttle(5.0, "cv_bridge conversion failed: %s", exc)
            return

        if not self.received_first_image:
            rospy.loginfo(
                "Camera stream OK: %dx%d, frame=%s",
                frame.shape[1],
                frame.shape[0],
                msg.header.frame_id,
            )
            self.received_first_image = True

        cv2.imshow(self.window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            rospy.signal_shutdown("Viewer closed by user")

    def shutdown(self) -> None:
        cv2.destroyAllWindows()


def main() -> int:
    rospy.init_node("vlm_gen3_camera_viewer", anonymous=False)
    WristCameraViewer()
    rospy.spin()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except rospy.ROSInterruptException:
        sys.exit(0)
