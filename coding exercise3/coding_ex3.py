# Student name: 

import math
from turtle import distance
import numpy as np
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Point, Pose, Quaternion, Twist, Vector3, PoseStamped, TransformStamped
from std_msgs.msg import String, Float32
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, LaserScan
import matplotlib.pyplot as plt
import time
from tf2_msgs.msg import TFMessage
from copy import copy
from visualization_msgs.msg import Marker

# Further info:
# On markers: http://wiki.ros.org/rviz/DisplayTypes/Marker
# Laser Scan message: http://docs.ros.org/en/melodic/api/sensor_msgs/html/msg/LaserScan.html

class CodingExercise3(Node):
   

    
    def __init__(self):
        super().__init__('CodingExercise3')

        self.ranges = [] # lidar measurements
        
        self.point_list = [] # A list of points to draw lines
        self.line = Marker()
        self.line_marker_init(self.line)

        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.theta = 0.0


        # Ros subscribers and publishers
        self.subscription_ekf = self.create_subscription(Odometry, 'terrasentia/ekf', self.callback_ekf, 10)
        self.subscription_scan = self.create_subscription(LaserScan, 'terrasentia/scan', self.callback_scan, 10)
        self.pub_lines = self.create_publisher(Marker, 'lines', 10)
        self.timer_draw_line_example = self.create_timer(0.1, self.draw_line_example_callback)

    
    def callback_ekf(self, msg):
        # You will need this function to read the translation and rotation of the robot with respect to the odometry frame
        position_msg = msg.pose.pose
        self.x = position_msg.position.x
        self.y = position_msg.position.y
        self.z = position_msg.position.z

        ori_x = position_msg.orientation.x
        ori_y = position_msg.orientation.y
        ori_z = position_msg.orientation.z
        ori_w = position_msg.orientation.w
        _, _, self.yaw = self.quat2euler(np.array([ori_w, ori_x, ori_y, ori_z]))
    
    def quat2euler(self, q): 
        q0, q1, q2, q3 = q.squeeze().tolist()

        m=np.eye(3,3)
        m[0,0] = 1.0 - 2.0*(q2*q2 + q3*q3)
        m[0,1] = 2.0*(q1*q2 - q0*q3)
        m[0,2] = 2.0*(q1*q3 + q0*q2)
        m[1,0] = 2.0*(q1*q2 + q0*q3)
        m[1,1] = 1.0 - 2.0*(q1*q1 + q3*q3)
        m[1,2] = 2.0*(q2*q3 - q0*q1)
        m[2,0] = 2.0*(q1*q3 - q0*q2)
        m[2,1] = 2.0*(q2*q3 + q0*q1)
        m[2,2] = 1.0 - 2.0*(q1*q1 + q2*q2)
        phi = math.atan2(m[2,1], m[2,2])
        theta = -math.asin(m[2,0])
        psi = math.atan2(m[1,0], m[0,0])
        return phi, theta, psi
   
    def callback_scan(self, msg):
        self.ranges = list(msg.ranges) # Lidar measurements
        self.theta = np.linspace(msg.angle_min,msg.angle_max,len(self.ranges))
        print("some-ranges:", self.ranges[0:5])
        print("Number of ranges:", len(self.ranges))
    
    def calculate_distance(self,x,y,slope,b):
        distance = np.abs(slope * x + b - y) / np.sqrt(slope ** 2 + 1)
        return distance
    
    def calculate_point2point_distance(self,x1,x2,y1,y2):
        distance = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
        return distance


    def draw_line_example_callback(self):
        # Here is just a simple example on how to draw a line on rviz using line markers. Feel free to use any other method
        theta = np.delete(self.theta,np.where(np.array(self.ranges) > 50))
        ranges = np.delete(self.ranges,np.where(np.array(self.ranges) > 50))
        x_test = self.x + ranges * np.cos(self.yaw + theta)
        y_test = self.y + ranges * np.sin(self.yaw + theta)

        threshold = 0.5
        pointDistance_threshold = 0.05

        start_point_index = 0
        end_point_index = 1

        threshold_exceed_indices = []
        threshold_exceed_indices.append(start_point_index)
    
        while end_point_index < len(x_test) - 1:
            slope,b = np.polyfit(x_test[start_point_index:end_point_index + 1],y_test[start_point_index:end_point_index + 1],1)
            end_point_index = end_point_index + 1
            while end_point_index < len(x_test):
                distance = self.calculate_distance(x_test[end_point_index],y_test[end_point_index],slope,b)
                distance2 = self.calculate_point2point_distance(x_test[end_point_index],x_test[end_point_index - 1],y_test[end_point_index],y_test[end_point_index - 1])
                if (distance < threshold) and (distance2 < pointDistance_threshold):
                    slope,b = np.polyfit(x_test[start_point_index:end_point_index + 1],y_test[start_point_index:end_point_index + 1],1)
                    end_point_index = end_point_index + 1
                else:
                    threshold_exceed_indices.append(end_point_index)
                    start_point_index = end_point_index
                    end_point_index = end_point_index + 1
                    break
            

        for i in range(len(threshold_exceed_indices)) :
            start_index = threshold_exceed_indices[i] 
        
            if i < len(threshold_exceed_indices) - 1:
                end_index = threshold_exceed_indices[i+1] - 1
            else:
                end_index = len(x_test) - 1
            
            if(end_index - start_index) < 10:
                continue
            
            p0 = Point()
            p0.x = x_test[start_index]
            p0.y = y_test[start_index]
            p0.z = 0.0

            p1 = Point()
            p1.x = x_test[end_index]
            p1.y = y_test[end_index]
            p1.z = 0.0

            self.point_list.append(copy(p0)) 
            self.point_list.append(copy(p1)) # You can append more pairs of points
        
        self.line.points = self.point_list
        self.pub_lines.publish(self.line) # It will draw a line between each pair of points

    def line_marker_init(self, line):
        line.header.frame_id="/odom"
        line.header.stamp=self.get_clock().now().to_msg()

        line.ns = "markers"
        line.id = 0

        line.type=Marker.LINE_LIST
        line.action = Marker.ADD
        line.pose.orientation.w = 1.0

        line.scale.x = 0.05
        line.scale.y= 0.05
        
        line.color.r = 1.0
        line.color.a = 1.0
        #line.lifetime = 0


def main(args=None):
    rclpy.init(args=args)

    cod3_node = CodingExercise3()
    
    rclpy.spin(cod3_node)

    cod3_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
