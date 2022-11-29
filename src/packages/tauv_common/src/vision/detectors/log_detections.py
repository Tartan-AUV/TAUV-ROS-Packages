#!/usr/bin/env python3

import rospy
from tauv_msgs.msg import Pose, FeatureDetection, FeatureDetections
from std_msgs.msg import Header
import numpy as np
import cv2
from cv_bridge import CvBridge
#from darknet_ros_msgs.msg import BoundingBoxes, BoundingBox
from sensor_msgs.msg import Image, CameraInfo
from vision.depth_estimation.depth_estimation import DepthEstimator
from geometry_msgs.msg import Point, PointStamped, Point
import math
from tauv_util import transforms
from tauv_util import types
from scipy.spatial.transform import Rotation
from std_srvs.srv import Trigger
import tf2_ros
import tf2_geometry_msgs

FT = 0.3048
IN = FT / 12

FORCE_DEPTH_BOTTOM = 4 * FT + 3 * IN

class LogDetections():
    def __init__(self):
        self.NODE_NAME = "detection_logger"
        self.NODE_NAME_FMT = "[{}]".format(self.NODE_NAME)
        self.SAMPLE_RATE = 60 # hz

        rospy.init_node(self.NODE_NAME, anonymous = True)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.front_camera_frame_id = "zedm_A_left_camera_optical_frame"
        self.bottom_camera_frame_id = "zedm_B_left_camera_optical_frame"
        
        self.camera_info_by_frame_id = {
            self.front_camera_frame_id: None,
            self.bottom_camera_frame_id: None
        }

        self.camera_data_by_frame_id = {
            self.front_camera_frame_id: None,
            self.bottom_camera_frame_id: None
        }

        self.camera_flags_by_frame_id = {
            self.front_camera_frame_id: False,
            self.bottom_camera_frame_id: False
        }

        #self.bounding_boxes = BoundingBoxes()
        self.cv_bridge = CvBridge()

        #rospy.wait_for_message("/darknet_ros/bounding_boxes", BoundingBoxes)
        
        # initialize subscribers for front camera depthmaps and camera info
        rospy.Subscriber("/zedm_A/zed_node_A/depth/depth_registered", Image, self.front_depth_callback)
        rospy.Subscriber("/zedm_A/zed_node_A/left/camera_info", CameraInfo, self.camera_info_callback)

        # initialize subscribers for downward camera depthmaps and camera info
        #rospy.Subscriber("/zedm_B/zed_node_B/depth/depth_registered", Image, self.depth_callback)
        #rospy.Subscriber("/zedm_B/zed_node_B/left/camera_info", CameraInfo, self.camera_info_callback)

        # initialize subscriber for darknet NN bounding boxes
        #rospy.Subscriber("/darknet_ros/bounding_boxes", BoundingBoxes, self.bbox_callback)

        self.detector = rospy.Publisher("register_object_detection", FeatureDetections,
                                        queue_size=10)

        #rospy.Subscriber("gnc/pose", Pose, self.update_position)

    def start(self):
        rospy.spin()

    # camera subscriber callbacks
    def camera_info_callback(self, msg):
        self.camera_info_by_frame_id[msg.header.frame_id] = msg

    def front_depth_callback(self, msg):
        self.camera_data_by_frame_id[msg.header.frame_id] = \
            self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        self.camera_flags_by_frame_id[msg.header.frame_id] = True

    # flag for invalid object detection position estimates
    def valid_pos(self, objdet):
        return (objdet.position.z >= 0) and \
               (not np.isnan(objdet.position.x) and \
                not np.isnan(objdet.position.y) and \
                not np.isnan(objdet.position.z))

    def bbox_callback(self, bboxes):
        camera_frame_id = bboxes.header.frame_id

        if not self.camera_flags_by_frame_id[camera_frame_id]:
            rospy.logerr("{} Waiting for depth for camera {}".format(
                self.NODE_NAME_FMT, camera_frame_id))
            return

        objects = FeatureDetections()
        objects.detections = list()

        for bbox in bboxes.bounding_boxes:
            objdet = FeatureDetection()
            objdet.tag = bbox.Class

            # calculate depth of the object in a relative coordinate frame, returned as an (x, y, z)
            # s.t. x is the
            relative_pos = DepthEstimator.estimate_absolute_depth(self.front_depth_image,
                                                                  bbox,
                                                                  self.camera_info_by_frame_id[camera_frame_id])

            if relative_pos == np.nan: # invalid depth estimate
                continue
            
            relative_pos_wrapped = PointStamped()
            relative_pos_wrapped.header = bboxes.header
            relative_pos_wrapped.point = Point(
                relative_pos[0], relative_pos[1], relative_pos[2]) # relative position

            # transform point from sensor coordinate frame to world coordinate frame
            sensor_to_world_tf = self.tf_buffer.lookup_transform(
                "world_ned",
                bboxes.header.frame_id,
                bboxes.header.stamp,
                rospy.Duration(1.0, 0.0) # 1 second timeout if the transform cannot be found
            )
            
            transformed_pos = tf2_geometry_msgs.do_transform_point(
                relative_pos_wrapped, sensor_to_world_tf
            )

            if self.valid_pos(objdet):
                objdet.position = transformed_pos.point
                objects.objdets.append(objdet)

        objects.detector_tag = bboxes.header.frame_id

        self.detector.publish(objects)
        
def main():
    s = LogDetections()
    s.start()

if __name__=='__main__':
    main()