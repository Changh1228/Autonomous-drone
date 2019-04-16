#!/usr/bin/env python
import math
import numpy as np
import rospy
import sys
import json
import a_star
import tf2_ros
import tf2_geometry_msgs
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from geometry_msgs.msg import PoseStamped, Point
from crazyflie_driver.msg import Position, Hover
from aruco_msgs.msg import MarkerArray, Marker

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
    path_ids = []

    current_target = 0
    goal = None
    goal_yaw = 0
    checkpoints = []
    checkpoints_yaw = []

    detected_arucos = []

    threshold = 0.025
    degree_threshold = 10

    wait = 0

    clearance_distance = 1.2
    passing_gate = False

    def __init__(self, argv=sys.argv):
        rospy.init_node("drone_movement")
        rospy.loginfo("Starting DroneMovement.")

        self.command_publisher  = rospy.Publisher("/cf1/cmd_position", Position, queue_size=2)
        self.command_hover_publisher  = rospy.Publisher("/cf1/cmd_hover", Hover, queue_size=2)
        
        rospy.Subscriber("/cf1/pose", PoseStamped, self.target_handler)
        rospy.Subscriber("/aruco/markers", MarkerArray, self.aruco_detected)
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
        self.path_ids = [g["id"] for g in world['gates']]

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
        if self.goal is None:
            return

        target = PoseStamped()

        target.header.stamp = rospy.Time.now()
        target.header.frame_id = 'map'

        target.pose.position.x = self.goal.x
        target.pose.position.y = self.goal.y
        target.pose.position.z = self.goal.z

        # target.pose.orientation.x, target.pose.orientation.y, target.pose.orientation.z, target.pose.orientation.w = quaternion_from_euler(0, 0, math.radians(self.checkpoints_yaw[idx + 1]))

        if not self.tf_buf.can_transform('cf1/odom', 'map', target.header.stamp):
            rospy.logwarn_throttle(5.0, 'No transform from map to odom')
            return

        goal_odom = self.tf_buf.transform(target, 'cf1/odom')

        # Position to send
        goal = Position()

        goal.header.stamp = rospy.Time.now()

        goal.x = goal_odom.pose.position.x
        goal.y = goal_odom.pose.position.y
        goal.z = goal_odom.pose.position.z
        goal.yaw = self.goal_yaw

        self.command_publisher.publish(goal)

    def aruco_detected(self, msg):
        self.detected_arucos = msg.markers

    def target_handler(self, msg):
        if not self.tf_buf.can_transform('map', 'cf1/odom', msg.header.stamp):
            rospy.logwarn_throttle(5.0, 'No transform from odom to map')
            return

        msg_map = self.tf_buf.transform(msg, 'map')

        roll, pitch, yaw = euler_from_quaternion((msg_map.pose.orientation.x,
                                              msg_map.pose.orientation.y,
                                              msg_map.pose.orientation.z,
                                              msg_map.pose.orientation.w))
        
        yaw = math.degrees(yaw)

        if self.goal is None:
            '''
            goal = Hover()

            goal.header.stamp = rospy.Time.now()

            goal.zDistance = 0.4

            self.command_hover_publisher.publish(goal)
            '''
            print("wtf?")
            print(msg_map.pose.position)
            path_x, path_y = a_star.aStarPlanning(msg_map.pose.position.x, msg_map.pose.position.y, self.path[self.current_target][0], self.path[self.current_target][1])
            path_yaw = a_star.yaw_planning(path_x, path_y)
            self.checkpoints = []
            self.checkpoints_yaw = []

            '''
            for i in range(len(path_x)):
                self.checkpoints.append(Point(path_x[i], path_y[i], 0.4))
                if i == len(path_x) - 1:
                    self.checkpoints_yaw.append(self.path[0][2])
                else:
                    self.checkpoints_yaw.append(path_yaw[i])
            '''
            
            yaw_increment = self.distance(yaw, self.path[self.current_target][2]) / len(path_x)
            for i in range(len(path_x)):
                self.checkpoints.append(Point(path_x[i], path_y[i], 0.4))
                self.checkpoints_yaw.append(yaw + i * yaw_increment)

            #print(self.checkpoints)
            self.goal = self.checkpoints[0]
            self.goal_yaw = self.checkpoints_yaw[0]

            print("First goal: " + str(self.goal))

            self.passing_gate = True

            return

        anglediff = (yaw - self.goal_yaw + 180 + 360) % 360 - 180
        # anglediff <= self.degree_threshold and anglediff >= -self.degree_threshold
        if self.goal.x + self.threshold > msg_map.pose.position.x > self.goal.x - self.threshold and self.goal.y + self.threshold > msg_map.pose.position.y > self.goal.y - self.threshold and self.goal.z + self.threshold > msg_map.pose.position.z > self.goal.z - self.threshold:
            idx = self.checkpoints.index(self.goal) if self.goal in self.checkpoints else -1
            if idx != -1 and idx < len(self.checkpoints) - 1:
                self.goal = self.checkpoints[idx + 1]
                self.goal_yaw = self.checkpoints_yaw[idx + 1]
                print("Next goal: " + str(self.goal))
            else:
                if self.wait <= 50 and self.passing_gate:
                    self.wait = self.wait + 1
                    print("wait")
                elif self.wait > 50 and self.passing_gate:
                    angle_to_pass_gate = np.deg2rad(self.path[self.current_target][2])
                    self.goal = Point(msg.pose.position.x + self.clearance_distance * np.cos(angle_to_pass_gate), msg.pose.position.y + self.clearance_distance * np.sin(angle_to_pass_gate), 0.4)
                    self.current_target = self.current_target + 1
                    self.passing_gate = False
                    self.wait = 0
                    print("Passing gate")
                    print("---------------------------")
                else:
                    path_x, path_y = a_star.aStarPlanning(msg_map.pose.position.x, msg_map.pose.position.y, self.path[self.current_target][0], self.path[self.current_target][1])
                    path_yaw = a_star.yaw_planning(path_x, path_y)
                    self.checkpoints = []
                    self.checkpoints_yaw = []

                    '''
                    for i in range(len(path_x)):
                        self.checkpoints.append(Point(path_x[i], path_y[i], 0.4))
                        if i == len(path_x) - 1:
                            self.checkpoints_yaw.append(self.path[0][2])
                        else:
                            self.checkpoints_yaw.append(path_yaw[i])
                    '''
                    
                    yaw_increment = self.distance(yaw, self.path[self.current_target][2]) / len(path_x)
                    for i in range(len(path_x)):
                        self.checkpoints.append(Point(path_x[i], path_y[i], 0.4))
                        self.checkpoints_yaw.append(yaw + i * yaw_increment)

                    #print(self.checkpoints)
                    self.goal = self.checkpoints[0]
                    self.goal_yaw = self.checkpoints_yaw[0]

                    print("First goal: " + str(self.goal))

                    self.passing_gate = True

if __name__ == "__main__":
    ros_node = DroneMovement()
    
    rate = rospy.Rate(20)  # Hz

    while not rospy.is_shutdown():
        #if ros_node.goal is not None:
            #ros_node.move()
        '''
        else:
            if len(ros_node.checkpoints) == 0:
                path_x, path_y = a_star.aStarPlanning(0, 0, ros_node.path[ros_node.current_target][0], ros_node.path[ros_node.current_target][1])
                path_yaw = a_star.yaw_planning(path_x, path_y)
                yaw_increment = ros_node.distance(0, ros_node.path[ros_node.current_target][2]) / len(path_x)
                for i in range(len(path_x)):
                    ros_node.checkpoints.append(Point(path_x[i], path_y[i], 0.4))
                    ros_node.checkpoints_yaw.append(i * yaw_increment)

                ros_node.passing_gate = True
            ros_node.goal = ros_node.checkpoints[0]
            ros_node.goal_yaw = ros_node.checkpoints_yaw[0]
        '''
        rate.sleep()
