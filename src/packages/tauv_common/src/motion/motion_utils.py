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
import numpy as np
from geometry_msgs.msg import PoseArray, Pose, PoseStamped, Twist
from tauv_msgs.srv import GetTraj, GetTrajResponse
from std_srvs.srv import SetBool, SetBoolRequest
from tauv_util.transforms import twist_body_to_world
from nav_msgs.msg import Path, Odometry as OdometryMsg
from motion.trajectories import Trajectory, TrajectoryStatus
from motion.trajectories.fixed_linear_trajectory import LinearTrajectory
import typing
from tauv_msgs.msg import TrajPoint

class MotionUtils:
    def __init__(self):
        self.initialized = False
        self.traj_service = rospy.Service('/gnc/get_traj', GetTraj, self._handle_get_traj)

        self.arm_proxy = rospy.ServiceProxy('/arm', SetBool)
        self.traj = None
        self.path_pub = rospy.Publisher('/gnc/path', Path, queue_size=10)

        self.pose = None    # pose
        self.twist = None   # world frame velocities

        self.holdpos = None

        self._odom_sub = rospy.Subscriber('/gnc/odom', OdometryMsg, self._handle_odom)

        rospy.wait_for_message("/gnc/odom", rospy.AnyMsg, timeout=None)
        # 10Hz status update loop:
        # rospy.Timer(rospy.Duration.from_sec(0.1), self._update_status)

        self.target_pub = rospy.Publisher("/gnc/traj_target", TrajPoint, queue_size=10)
        self.target_pose_stamped_pub = rospy.Publisher("/gnc/traj_target_pose_stamped", PoseStamped, queue_size=10)

        rospy.Timer(rospy.Duration.from_sec(0.05), self._pub_target)

    def abort(self):
        self.traj = None

    def set_trajectory(self, traj):
        assert isinstance(traj, Trajectory)
        self.traj = traj
        self.traj.start()

    def get_robot_state(self):
        return self.pose, self.twist

    def get_position(self):
        return np.array([self.pose.position.x, self.pose.position.y, self.pose.position.z])

    def get_motion_status(self):
        if self.traj is None:
            return TrajectoryStatus.PENDING

        return self.traj.get_status(self.pose)

    def arm(self, armed):
        try:
            self.arm_proxy.call(SetBoolRequest(armed))
        except rospy.ServiceException:
            rospy.logwarn_throttle('[Motion Utils] Cannot arm/disarm: Arm server not responding. (This could be due '
                                   'to running in simulation)')

    def goto(self, pos: typing.Tuple[float],
                   heading: float = None, 
                   block_until: TrajectoryStatus = TrajectoryStatus.FINISHED, 
                   v=.4, a=.4, j=.4,
                   threshold_lin=0.5, threshold_ang=0.5):
        start_pose, start_twist = self.get_target()
        newtraj = LinearTrajectory(start_pose, start_twist, [pos], [heading], v=v, a=a, j=j)
        self.set_trajectory(newtraj)

    def get_target(self) -> typing.Tuple[Pose, Twist]:
        if self.get_motion_status() == TrajectoryStatus.PENDING:
            # Return current position? or none?
            print("pending")
            return self.pose, self.twist
        else:
            req = GetTraj._request_class()
            req.curr_time = rospy.Time.now()
            req.len = 1
            req.curr_pose = self.pose
            req.curr_twist = self.twist
            req.dt = .1 # i guess
            res: GetTrajResponse = self.traj.get_points(req)
            return res.poses[0], res.twists[0]

    def _pub_target(self, timer_event):
        pose, twist = self.get_target()
        tp = TrajPoint(pose, twist)
        self.target_pub.publish(tp)

        tp_pose_stamped = PoseStamped()
        tp_pose_stamped.header.frame_id = "odom_ned"
        tp_pose_stamped.pose = pose
        self.target_pose_stamped_pub.publish(tp_pose_stamped)

        if self.traj is not None:
            path = self.traj.as_path()

            self.path_pub.publish(path)

    def _handle_get_traj(self, req):
        response = GetTrajResponse()

        if self.traj is None:
            response.success = False
            return response

        response = self.traj.get_points(req)

        return response

    def _handle_odom(self, msg: OdometryMsg):
        self.pose = msg.pose.pose
        self.twist = twist_body_to_world(msg.pose.pose, msg.twist.twist)
        self.initialized = True