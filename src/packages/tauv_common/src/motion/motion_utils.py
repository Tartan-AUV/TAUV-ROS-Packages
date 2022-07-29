# motion_utils.y
#
# Simple python library intended to expose motion control of the AUV to mission scripts.
# Only one MotionUtils class should be instantiated at a time. (TODO: make this a python singleton)
# MotionUtils is intended to be used in conjunction with the mpc_trajectory_follower planner.
#
#
# To use: one MotionUtils instance should be created at sub startup, then passed to any mission scripts
# from the mission manager node.
# The MotionUtils class will automatically provide updates to the trajectory follower.
# Use update_trajectory to update the reference trajectory object to be used by motion_utils.
#
#

import rospy
from scipy.spatial.transform import Rotation
from tauv_msgs.msg import Pose as PoseMsg
from geometry_msgs.msg import PoseArray
from std_msgs.msg import Header
from tauv_msgs.srv import GetTraj, GetTrajResponse
from std_srvs.srv import SetBool, SetBoolRequest
from tauv_util.transforms import twist_body_to_world
from nav_msgs.msg import Path, Odometry as OdometryMsg
from .trajectories import Trajectory, TrajectoryStatus


class MotionUtils:
    def __init__(self):
        self.initialized = False
        self.traj_service = rospy.Service('/gnc/get_traj', GetTraj, self._traj_callback)

        self.arm_proxy = rospy.ServiceProxy('/arm', SetBool)
        self.traj = None
        self.path_pub = rospy.Publisher('/gnc/path_viz', Path, queue_size=10)
        self._pose_array_pub = rospy.Publisher('gnc/pose_array', PoseArray, queue_size=10)

        self.pose = None
        self.twist = None

        self.holdpos = None

        self._odom_sub = rospy.Subscriber('/gnc/odom', OdometryMsg, self._handle_odom)

        # 10Hz status update loop:
        rospy.Timer(rospy.Duration.from_sec(0.1), self._update_status)
        while not self.initialized:
            rospy.sleep(0.05)

    def abort(self):
        self.traj = None

    def set_trajectory(self, traj):
        assert isinstance(traj, Trajectory)
        self.traj = traj
        if self.traj.get_status() != TrajectoryStatus.INITIALIZED:
            return
        self.traj.set_executing()

    def get_robot_state(self):
        return self.pose, self.twist

    def get_position(self):
        return (self.pose.position.x, self.pose.position.y, self.pose.position.z)

    def get_motion_status(self):
        if self.traj is None:
            return TrajectoryStatus.TIMEOUT
        return self.traj.get_status()

    def arm(self, armed):
        try:
            self.arm_proxy(SetBoolRequest(armed))
        except rospy.ServiceException:
            rospy.logwarn_throttle('[Motion Utils] Cannot arm/disarm: Arm server not responding. (This could be due '
                                   'to running in simulation)')

    def _update_status(self, timer_event):
        if self.traj is not None:
            path = Path()
            path.header.frame_id = 'odom'
            path.header.stamp = rospy.Time.now()

            path = self.traj.as_path(dt=0.5)

            self.path_pub.publish(path)


        # TODO: also post current status, such as eta for current trajectory, percent done, etc.

    def _traj_callback(self, req):
        response = GetTrajResponse()

        if self.traj is None:
            response.success = False
            return response

        response = self.traj.get_points(req)

        if response.success:
            msg = PoseArray()
            msg.header = Header()
            msg.header.frame_id = 'odom'
            msg.poses = response.poses
            self._pose_array_pub.publish(msg)

        return response

    def _handle_odom(self, msg: OdometryMsg):
        self.pose = msg.pose.pose
        self.twist = twist_body_to_world(msg.pose.pose, msg.twist.twist)
        self.initialized = True