import rospy
import numpy as np
import tf2_ros
import cv2
import cv_bridge
import message_filters
from sensor_msgs.msg import Image, CameraInfo

class RedDetectorNode:
    def __init__(self):

        self._rgb_sub = message_filters.Subscriber('color', Image, queue_size=10)

        self._camera_info = rospy.wait_for_message('camera_info', CameraInfo, 60)
        
        self._bridge = cv_bridge.CvBridge()

        self._red_pub = rospy.Publisher('global_map/red_detections',Image,queue_size=10)

    def callback(self, img: Image):
        # Update camera matrix
        self._detector.set_camera_matrix(self._camera_info.K)

        # Convert colors
        cv_rgb = self._bridge.imgmsg_to_cv2(rgb, desired_encoding="rgb8")
        cv_rgb = cv2.cvtColor(cv_rgb, cv2.COLOR_RGB2BGR)
        cv_hsv = cv2.cvtColor(cv_rgb, cv2.COLOR_BGR2HSV)

	# create red mask
	lower_red1 = np.array([0, 100, 100])
	upper_red1 = np.array([10, 255, 255])
	lower_red2 = np.array([160, 100, 100])
	upper_red2 = np.array([180, 255, 255])
	mask1 = cv2.inRange(cv_hsv, lower_red1, upper_red1)
	mask2 = cv2.inRange(cv_hsv, lower_red2, upper_red2)

	red_mask = cv2.bitwise_or(mask1, mask2)
	contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	
	# identify red mask on the original image
	cv2.drawContours(cv_rgb, contours, -1, (0, 0, 255), 2)
	
	rosimg = self._bridge.cv2_to_imgmsg(cv_rgb, encoding="rgb8")

	# publish to new image topic
        self._red_pub.publish(rosimg)

def main():
    rospy.init_node('red_detector', anonymous=True)
    r = RedDetectorNode()
    rospy.spin()

main()
