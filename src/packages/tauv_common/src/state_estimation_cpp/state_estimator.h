#include <variant>
#include <array>
#include <ros/ros.h>
#include <boost/heap/priority_queue.hpp>
#include <boost/variant.hpp>
#include <tauv_msgs/XsensImuData.h>
#include <tauv_msgs/TeledyneDvlData.h>
#include <tauv_msgs/FluidDepth.h>
#include <tauv_msgs/Pose.h>
#include <tf/transform_broadcaster.h>
#include "ekf.h"

#pragma once

class StateEstimator {
  public:
    StateEstimator(ros::NodeHandle& n);

    void update(const ros::TimerEvent& e);

    void handle_imu(const tauv_msgs::XsensImuData::ConstPtr& msg);
    void handle_dvl(const tauv_msgs::TeledyneDvlData::ConstPtr& msg);
    void handle_depth(const tauv_msgs::FluidDepth::ConstPtr& msg);

    class SensorMsg;

  private:
    ros::NodeHandle& n;
    ros::Timer timer;

    ros::Subscriber imu_sub;
    ros::Subscriber dvl_sub;
    ros::Subscriber depth_sub;

    ros::Publisher pose_pub;
    ros::Publisher odom_pub;

    tf::TransformBroadcaster odom_tf_broadcaster;

    Ekf ekf;

    using SensorMsgQueue = boost::heap::priority_queue<SensorMsg>;

    SensorMsgQueue msg_queue;

    ros::Duration dt;
    ros::Duration horizon_delay;
    ros::Time last_horizon_time;
    Eigen::Vector3d dvl_offset;
    Eigen::Matrix<double, 15, 1> process_covariance;
    Eigen::Matrix<double, 9, 1> imu_covariance;
    Eigen::Vector3d dvl_covariance;
    double depth_covariance;

    void load_config();

    void apply_imu(const tauv_msgs::XsensImuData::ConstPtr &msg);
    void apply_dvl(const tauv_msgs::TeledyneDvlData::ConstPtr &msg);
    void apply_depth(const tauv_msgs::FluidDepth::ConstPtr &msg);
};

class StateEstimator::SensorMsg {
  public:
    enum Type { IMU, DVL, DEPTH };

    Type type;
    ros::Time time;
    boost::variant<tauv_msgs::XsensImuData::ConstPtr, tauv_msgs::TeledyneDvlData::ConstPtr, tauv_msgs::FluidDepth::ConstPtr> msg;

    SensorMsg(const tauv_msgs::XsensImuData::ConstPtr &raw_msg) : msg(raw_msg) {
      this->type = Type::IMU;  
      this->time = raw_msg->header.stamp;
    };
    SensorMsg(const tauv_msgs::TeledyneDvlData::ConstPtr &raw_msg) : msg(raw_msg) {
      this->type = Type::DVL;  
      this->time = raw_msg->header.stamp;
    };
    SensorMsg(const tauv_msgs::FluidDepth::ConstPtr &raw_msg) : msg(raw_msg) {
      this->type = Type::DEPTH;  
      this->time = raw_msg->header.stamp;
    };
    
    bool is_imu() { return this->type == Type::IMU; };
    bool is_dvl() { return this->type == Type::DVL; };
    bool is_depth() { return this->type == Type::DEPTH; };

    const tauv_msgs::XsensImuData::ConstPtr& as_imu() {
      return boost::get<const tauv_msgs::XsensImuData::ConstPtr&>(this->msg);
    };
    const tauv_msgs::TeledyneDvlData::ConstPtr& as_dvl() {
      return boost::get<const tauv_msgs::TeledyneDvlData::ConstPtr&>(this->msg);
    };
    const tauv_msgs::FluidDepth::ConstPtr& as_depth() {
      return boost::get<const tauv_msgs::FluidDepth::ConstPtr&>(this->msg);
    };
};

bool operator < (const StateEstimator::SensorMsg &lhs, const StateEstimator::SensorMsg &rhs) 
{
  return lhs.time >= rhs.time;
}
