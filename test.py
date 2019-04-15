#!/usr/bin/env python
import math
import numpy as np
import rospy
import sys
import json
import a_star
import tf2_ros
import tf2_geometry_msgs
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import PoseStamped, Point
from crazyflie_driver.msg import Position

class DroneMovement:
    command_publisher = None
    tf_buf = None
    tf_lstn = None

    a = [1.5, -0.75, 135]
    b = [0.5, 0.25, 135]
    c = [-1.25, 1.00, 180]
    d = [-2.75, 0.5, 180]
    e = [-2.5, -0.5, -90]
    f = [-1.75, -0.75, 0]
    g = [0.00, -0.75, 45]
    h = [1.00, 0.25, 45]
    path = [a, b, c, d, e, f, g, h]

    current_target = 0
    goal = Point(0, 0, 0.4)
    goal_yaw = 0
    checkpoints = []
    checkpoints_yaw = []

    threshold = 0.075

    clearance_distance = 0.5
    passing_gate = False

    def __init__(self, argv=sys.argv):
        rospy.init_node("drone_movement")
        rospy.loginfo("Starting DroneMovement.")

        self.command_publisher  = rospy.Publisher("/cf1/cmd_position", Position, queue_size=2)
        
        rospy.Subscriber("/cf1/pose", PoseStamped, self.target_handler)
        rospy.loginfo("Subscribing to the /cf1/pose topic")

        self.tf_buf = tf2_ros.Buffer()
        self.tf_lstn = tf2_ros.TransformListener(self.tf_buf)

        # Let ROS filter through the arguments
        args = rospy.myargv(argv=argv)

        # Load world JSON
        with open(args[1], 'rb') as f:
            world = json.load(f)

        # Create a checkpoint in front of each gate
        self.path = [[g["position"][0] + self.clearance_distance / 2 * np.cos(np.deg2rad(180 + g["heading"])), g["position"][1] + self.clearance_distance / 2 * np.sin(np.deg2rad(180 + g["heading"])), g["heading"]] for g in world['gates']]

    def distance(self, a, b):
        if (a == b):
            return 0
        elif (a < 0) and (b < 0) or (a > 0) and (b > 0):
            if (a < b):
                return (abs(abs(a) - abs(b)))
            else:
                return -(abs(abs(a) - abs(b)))
        else:
            return math.copysign((abs(a) + abs(b)),b)

    def move(self):
        target = Position()

        target.header.stamp = rospy.Time.now()

        target.x = self.goal.x
        target.y = self.goal.y
        target.z = self.goal.z
        target.yaw = self.goal_yaw

        self.command_publisher.publish(target)            

    def target_handler(self, msg):
        roll, pitch, yaw = euler_from_quaternion((msg.pose.orientation.x,
                                              msg.pose.orientation.y,
                                              msg.pose.orientation.z,
                                              msg.pose.orientation.w))
        
        yaw = math.degrees(yaw)

        if self.goal.x + self.threshold > msg.pose.position.x > self.goal.x -self.threshold and self.goal.y + self.threshold > msg.pose.position.y > self.goal.y - self.threshold and self.goal.z + self.threshold > msg.pose.position.z > self.goal.z - self.threshold:
            idx = self.checkpoints.index(self.goal) if self.goal in self.checkpoints else -1
            if idx != -1 and idx < len(self.checkpoints) - 1:
                self.goal = self.checkpoints[idx + 1]
                self.goal_yaw = self.checkpoints_yaw[idx + 1]

                '''
                # Need to tell TF that the goal was just generated
                target = PoseStamped()

                target.header.stamp = rospy.Time.now()
                target.header.frame_id = 'map'

                target.pose.position.x = self.checkpoints[idx + 1].x
                target.pose.position.y = self.checkpoints[idx + 1].y
                target.pose.position.z = self.checkpoints[idx + 1].z

                if not self.tf_buf.can_transform('map', 'cf1/odom', target.header.stamp):
                    rospy.logwarn_throttle(5.0, 'No transform from map to odom')
                    return

                goal_odom = self.tf_buf.transform(target, 'cf1/odom')
                self.goal = Point(goal_odom.pose.position.x, goal_odom.pose.position.y, goal_odom.pose.position.z)
                self.goal_yaw = self.checkpoints_yaw[idx + 1]
                '''
            else:
                if self.passing_gate:
                    angle_to_pass_gate = np.deg2rad(self.path[self.current_target][2])
                    self.goal = Point(msg.pose.position.x + self.clearance_distance * np.cos(angle_to_pass_gate), msg.pose.position.y + self.clearance_distance * np.sin(angle_to_pass_gate), 0.4)
                    self.current_target = self.current_target + 1
                    self.passing_gate = False
                else:
                    path_x, path_y = a_star.aStarPlanning(msg.pose.position.x, msg.pose.position.y, self.path[self.current_target][0], self.path[self.current_target][1])
                    path_yaw = a_star.yaw_planning(path_x, path_y)
                    self.checkpoints = []
                    self.checkpoints_yaw = []

                    for i in range(len(path_x)):
                        self.checkpoints.append(Point(path_x[i], path_y[i], 0.4))
                        self.checkpoints_yaw.append(path_yaw[i])
                    
                    '''
                    yaw_increment = self.distance(yaw, self.path[self.current_target][2]) / len(path_x)
                    for i in range(len(path_x)):
                        self.checkpoints.append(Point(path_x[len(path_x) - i - 1], path_y[len(path_x) - i - 1], 0.4))
                        self.checkpoints_yaw.append(yaw + i * yaw_increment)
                    '''

                    #print(self.checkpoints)
                    self.goal = self.checkpoints[0]
                    self.goal_yaw = self.checkpoints_yaw[0]

                    '''
                    # Need to tell TF that the goal was just generated
                    target = PoseStamped()

                    target.header.stamp = rospy.Time.now()
                    target.header.frame_id = 'map'

                    target.pose.position.x = self.checkpoints[0].x
                    target.pose.position.y = self.checkpoints[0].y
                    target.pose.position.z = self.checkpoints[0].z

                    if not self.tf_buf.can_transform('map', 'cf1/odom', target.header.stamp):
                        rospy.logwarn_throttle(5.0, 'No transform from map to odom')
                        return

                    goal_odom = self.tf_buf.transform(target, 'cf1/odom')
                    self.goal = Point(goal_odom.pose.position.x, goal_odom.pose.position.y, goal_odom.pose.position.z)
                    self.goal_yaw = self.checkpoints_yaw[0]
                    '''

                    self.passing_gate = True

if __name__ == "__main__":
    ros_node = DroneMovement()
    
    rate = rospy.Rate(20)  # Hz

    while not rospy.is_shutdown():
        if ros_node.goal:
            ros_node.move()
        else:
            if len(ros_node.checkpoints) == 0:
                path_x, path_y = a_star.aStarPlanning(0, 0, ros_node.path[ros_node.current_target][0], ros_node.path[ros_node.current_target][1])
                for i in range(len(path_x)):
                    ros_node.checkpoints.append(Point(path_x[i], path_y[i], 0.4))

                ros_node.current_target = 1
            ros_node.goal = ros_node.checkpoints[0]
        rate.sleep()
