// **THIS IS AN AUTOGENERATED FILE, DO NOT EDIT MANUALLY**
// 
// Generated by generate_alarms.py (Tom Scherlis 2022)
// This file is generated whenever tauv_common is built using catkin build.
// See alarms/CMakeLists.txt for configuring the autogen build step
// 
// Generated on Thu Jul 28 15:44:08 2022

#pragma once

namespace tauv_alarms {
  enum AlarmType {
     UNKNOWN_ALARMS=0, 
     THRUSTERS_NOT_INITIALIZED=1, 
     ARDUINO_NOT_INITIALIZED=2, 
     IMU_NOT_INITIALIZED=3, 
     IMU_SYNC_NOT_INITIALIZED=4, 
     DVL_NOT_INITIALIZED=5, 
     DVL_NO_TIMESTAMPS=6, 
     DVL_NO_LOCK=7, 
     SONAR_NOT_INITIALIZED=8, 
     STATE_ESTIMATION_NOT_INITIALIZED=9, 
     STATE_ESTIMATION_DELAYED=10, 
     KILL_SWITCH_ACTIVE=11, 
     CONTROLLER_NOT_INITIALIZED=12, 
     MPC_PLANNER_NOT_INITIALIZED=13, 
  };
}
