#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Temperature
from std_msgs.msg import Float64
from std_msgs.msg import Header
from ms5837lib import ms5837

class DepthSensor():
    def __init__(self):
        if not rospy.get_param('/vehicle_params/has_depth_sensor'):
            raise ValueError('''Error: Vehicle does not support depth sensor.
         Is the has_depth_sensor rosparam set in the vehicle_params.yaml?
         If not, then don't launch this node! ''')

        self.pub_depth = rospy.Publisher('depth', Float64, queue_size=10)
        self.pub_temp = rospy.Publisher('temperature', Temperature, queue_size=10)

        model_name = rospy.get_param('/vehicle_params/depth_sensor')
        if model_name == 'bar30':
            self.ms5837 = ms5837.MS5837_30BA()
        elif model_name == 'bar02':
            self.ms5837 = ms5837.MS5837_02BA()
        else:
            raise ValueError('''Error: specified depth sensor not supported.
         Supported depth sensors are \'bar30\' and \'bar02\', set with the
         depth_sensor tag in the vehicle_params.yaml.''')

        while not self.ms5837.init():
            rospy.sleep(10)
            print("Failed to initialize depth sensor, retrying in 10 seconds")
        print("Depth sensor initialized!")

    def start(self):
        r = rospy.Rate(10)  # 10hz
        while not rospy.is_shutdown():
            self.ms5837.read()
            self.pub_depth.publish(Float64(self.ms5837.depth()))
            tempmsg = Temperature()
            tempmsg.header = Header()
            tempmsg.header.stamp = rospy.Time.now()
            tempmsg.temperature = self.ms5837.temperature()
            self.pub_temp.publish(tempmsg)
            r.sleep()


def main():
    rospy.init_node('depth_sensor')
    d = DepthSensor()
    d.start()
