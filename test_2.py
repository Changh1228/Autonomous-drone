#!/usr/bin/env python

import math
import rospy
import a_star
import tf2_ros
import tf2_geometry_msgs
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from geometry_msgs.msg import PoseStamped, Point
from crazyflie_driver.msg import Position, Hover
from std_msgs.msg import String

class TestPoseToMap:
    tf_buf = None
    tf_lstn = None

    my_position = Position()

    def __init__(self):
        rospy.init_node("test_pose_to_map")

        rospy.Subscriber("/cf1/pose", PoseStamped, self.pose_handler)
        rospy.Subscriber("/cf1/plan", String, self.plan_path)

        self.tf_buf = tf2_ros.Buffer()
        self.tf_lstn = tf2_ros.TransformListener(self.tf_buf)

        pass

    def pose_handler(self, msg):
        if not self.tf_buf.can_transform('map', msg.header.frame_id, msg.header.stamp):
            rospy.logwarn_throttle(5.0, 'No transform from %s to map' % msg.header.frame_id)
            return

        msg_map = self.tf_buf.transform(msg, 'map')

        roll, pitch, yaw = euler_from_quaternion((msg_map.pose.orientation.x,
                                              msg_map.pose.orientation.y,
                                              msg_map.pose.orientation.z,
                                              msg_map.pose.orientation.w))
        
        yaw = math.degrees(yaw)

        self.my_position.x = msg_map.pose.position.x
        self.my_position.y = msg_map.pose.position.y
        self.my_position.z = msg_map.pose.position.z
        self.my_position.yaw = yaw

    def plan_path(self, msg):
        print(self.my_position)
        path_x, path_y = a_star.aStarPlanning(self.my_position.x, self.my_position.y, 1.5, -0.75)
        path_yaw = a_star.yaw_planning(path_x, path_y)

if __name__ == "__main__":
    ros_node = TestPoseToMap()
    
    rospy.spin()