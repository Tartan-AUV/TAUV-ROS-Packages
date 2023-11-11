
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge



class ReddishFilter:
    def __init__(self):
        rospy.init_node('reddish_filter_node')
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/kf/vehicle/oakd_front/color/image_raw", Image, self.image_callback)
        self.image_pub = rospy.Publisher("/kf/vehicle/oakd_front/color/reddish_image_raw", Image, queue_size=10)

    def image_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            hsv_img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        except Exception as e:
            print(e)
            return

        #lower mask(0-10)
        lower_red = np.array([0,50,50])
        upper_red = np.array([10,255,255])
        mask0 = cv2.inRange(hsv_img, lower_red, upper_red)

        #upper mask(170-180)
        lower_red = np.array([170,50,50])
        upper_red = np.array([180,255,255])
        mask1 = cv2.inRange(hsv_img, lower_red, upper_red)

        #join the mask
        mask = mask0+mask1

        img_copy = cv_image.copy()
        img_copy[np.where(mask==0)] = 0

        filtered_img = self.bridge.cv2_to_imgmsg(img_copy, 'bgr8')

        try:
            # Convert the filtered image back to ROS format and publish
            self.image_pub.publish(filtered_img)
        except Exception as e:
            print(e)

def main():
    reddish_filter = ReddishFilter()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

if __name__ == '__main__':
    main()