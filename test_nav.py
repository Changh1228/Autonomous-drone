#!/usr/bin/env python
import numpy as np
import rospy
import sys
import json
import a_star
import tf2_ros
import tf2_geometry_msgs
from timeit import default_timer as timer
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from geometry_msgs.msg import PoseStamped, Point
from crazyflie_driver.msg import Position, Hover
from std_msgs.msg import Header

class DroneMovement:
    hover_publisher = None
    position_publisher = None

    tf_buf = None

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

    threshold = 0.01
    degree_threshold = 5

    clearance_distance = 1.0

    waiting_time = 10 # seconds
    starting_time = 0.0

    goal = None
    last_position_command = None
    current_target = 1
    passing_gate = False

    checkpoints = []

    def __init__(self, argv=sys.argv):
        rospy.init_node("drone_movement")
        rospy.loginfo("Starting DroneMovement.")

        self.hover_publisher  = rospy.Publisher("/cf1/cmd_hover", Hover, queue_size=2)
        self.position_publisher  = rospy.Publisher("/cf1/cmd_position", Position, queue_size=2)

        rospy.Subscriber("/cf1/pose", PoseStamped, self.pose_callback)
        rospy.loginfo("Subscribing to the /cf1/pose topic")

        self.tf_buf = tf2_ros.Buffer()
        tf2_ros.TransformListener(self.tf_buf)

        # Let ROS filter through the arguments
        args = rospy.myargv(argv=argv)

        # Load world JSON
        with open(args[1], 'rb') as f:
            world = json.load(f)

        # Create a checkpoint in front of each gate
        self.path = [Position(Header(), g["position"][0] + self.clearance_distance / 2 * np.cos(np.deg2rad(180 + g["heading"])), g["position"][1] + self.clearance_distance / 2 * np.sin(np.deg2rad(180 + g["heading"])), 0.41, g["heading"]) for g in world["gates"]]
        self.path_ids = [g["id"] for g in world["gates"]]
        
        starting_time = timer()
        rospy.loginfo("Node finished initialization")

    #----------------------------------------------------#
    #                  Helper functions                  #
    #----------------------------------------------------#

    def plan_path(self, current_position, target_position):
        path_x, path_y, path_yaw = a_star.aStarPlanning(current_position.x, current_position.y, target_position.x, target_position.y)
        self.checkpoints = []

        for i in range(len(path_x)):
            checkpoint = Position()
            checkpoint.x = path_x[i]
            checkpoint.y = path_y[i]
            checkpoint.z = 0.41
            checkpoint.yaw = path_yaw[i]

            self.checkpoints.append(checkpoint)
        
        self.goal = self.checkpoints[0]
        self.passing_gate = True

        print("Plan finished")

    #----------------------------------------------------#
    #                   Main functions                   #
    #----------------------------------------------------#

    def hover_in_place(self, yawrate):
        hover_command = Hover()

        hover_command.header.stamp = rospy.Time.now()
        hover_command.header.frame_id = "cf1/odom"

        hover_command.vx = 0.0
        hover_command.vy = 0.0
        hover_command.yawrate = yawrate
        hover_command.zDistance = 0.41

        self.hover_publisher.publish(hover_command)

    def move(self):
        target = PoseStamped()

        target.header.stamp = rospy.Time.now()
        target.header.frame_id = "map"

        target.pose.position.x = self.goal.x
        target.pose.position.y = self.goal.y
        target.pose.position.z = self.goal.z
        target.pose.orientation.x, target.pose.orientation.y, target.pose.orientation.z, target.pose.orientation.w = quaternion_from_euler(0, 0, np.deg2rad(self.goal.yaw))

        if not self.tf_buf.can_transform("cf1/odom", "map", target.header.stamp, timeout=rospy.Duration(0.2)):
            rospy.logwarn_throttle(5.0, "No transform from map to odom")
            
            # Send last command
            if self.last_position_command is not None:
                self.position_publisher.publish(self.last_position_command)
                print("Using last position")
            else:
                self.hover_in_place(0.0)
                print("Hovering")
        else:
            target_odom = self.tf_buf.transform(target, "cf1/odom", rospy.Duration(0.2))

            roll, pitch, yaw = euler_from_quaternion((target_odom.pose.orientation.x,
                                                target_odom.pose.orientation.y,
                                                target_odom.pose.orientation.z,
                                                target_odom.pose.orientation.w))

            yaw = np.rad2deg(yaw)

            # Position to send
            position_command = Position()

            position_command.header.stamp = rospy.Time.now()
            position_command.header.frame_id = "cf1/odom"

            position_command.x = target_odom.pose.position.x
            position_command.y = target_odom.pose.position.y
            position_command.z = target_odom.pose.position.z
            position_command.yaw = yaw

            self.last_position_command = position_command

            self.position_publisher.publish(position_command)
            print("Moving to goal")

    #----------------------------------------------------#
    #                  Callback functions                #
    #----------------------------------------------------#

    def pose_callback(self, msg):
        if not self.tf_buf.can_transform("map", msg.header.frame_id, msg.header.stamp, timeout=rospy.Duration(0.2)):
            rospy.logwarn_throttle(5.0, "No transform from %s to map" % msg.header.frame_id)
            return

        msg_map = self.tf_buf.transform(msg, "map", rospy.Duration(0.2))

        roll, pitch, yaw = euler_from_quaternion((msg_map.pose.orientation.x,
                                              msg_map.pose.orientation.y,
                                              msg_map.pose.orientation.z,
                                              msg_map.pose.orientation.w))
        
        yaw = np.rad2deg(yaw)

        detected_position = Position()
        detected_position.x = msg_map.pose.position.x
        detected_position.y = msg_map.pose.position.y
        detected_position.z = msg_map.pose.position.z
        detected_position.yaw = yaw

        if self.goal is None:
            self.plan_path(detected_position, self.path[1])
        else:
            anglediff = (detected_position.yaw - self.goal.yaw + 180 + 360) % 360 - 180

            if self.goal.x + self.threshold > detected_position.x > self.goal.x - self.threshold and self.goal.y + self.threshold > detected_position.y > self.goal.y - self.threshold and self.goal.z + self.threshold > detected_position.z > self.goal.z - self.threshold and anglediff <= self.degree_threshold and anglediff >= -self.degree_threshold:
            # if self.goal.x + self.threshold > detected_position.x > self.goal.x - self.threshold and self.goal.y + self.threshold > detected_position.y > self.goal.y - self.threshold and self.goal.z + self.threshold > detected_position.z > self.goal.z - self.threshold:
                idx = self.checkpoints.index(self.goal) if self.goal in self.checkpoints else -1
                if idx != -1 and idx < len(self.checkpoints) - 1:
                    self.goal = self.checkpoints[idx + 1]
                    print("Next checkpoint")
                    print(self.goal.yaw)
                else:
                    if self.passing_gate:
                        angle_to_pass_gate = np.deg2rad(self.path[self.current_target].yaw)
                        
                        other_side_of_gate = Position()
                        other_side_of_gate.x = detected_position.x + self.clearance_distance * np.cos(angle_to_pass_gate)
                        other_side_of_gate.y = detected_position.y + self.clearance_distance * np.sin(angle_to_pass_gate)
                        other_side_of_gate.z = 0.41
                        other_side_of_gate.yaw = self.path[self.current_target].yaw

                        print(other_side_of_gate)

                        '''
                        self.checkpoints = []

                        if abs(detected_position.x - other_side_of_gate.x) > abs(detected_position.y - other_side_of_gate.y):
                            steps = int(abs(detected_position.x - other_side_of_gate.x) / 0.2)

                            for s in range(steps):
                                checkpoint = Position()
                                checkpoint.x = detected_position.x + ((other_side_of_gate.x - detected_position.x) / steps) * (s + 1)
                                checkpoint.y = detected_position.y + ((other_side_of_gate.y - detected_position.y) / steps) * (s + 1)
                                checkpoint.z = 0.4
                                checkpoint.yaw = other_side_of_gate.yaw

                                self.checkpoints.append(checkpoint)
                            
                            self.goal = self.checkpoints[0]
                        else:
                            steps = int(abs(detected_position.y - other_side_of_gate.y) / 0.2)

                            for s in range(steps):
                                checkpoint = Position()
                                checkpoint.x = detected_position.x + ((other_side_of_gate.x - detected_position.x) / steps) * (s + 1)
                                checkpoint.y = detected_position.y + ((other_side_of_gate.y - detected_position.y) / steps) * (s + 1)
                                checkpoint.z = 0.4
                                checkpoint.yaw = other_side_of_gate.yaw

                                self.checkpoints.append(checkpoint)
                            
                            self.goal = self.checkpoints[0]
                        '''

                        self.goal = other_side_of_gate
                        self.current_target = self.current_target + 1
                        self.passing_gate = False
                        print("Passing gate")
                    else:
                        self.plan_path(detected_position, self.path[self.current_target])

if __name__ == "__main__":
    ros_node = DroneMovement()
    
    rate = rospy.Rate(20) # Hz

    while not rospy.is_shutdown():
        if ros_node.starting_time != 0.0 and timer() - ros_node.starting_time > ros_node.waiting_time:
            ros_node.hover_in_place(5.0)
        elif ros_node.goal is None:
            ros_node.hover_in_place(0.0)
        else:
            ros_node.move()
        rate.sleep()