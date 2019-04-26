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
    def __init__(self, argv=sys.argv):
        rospy.init_node("drone_movement")
        rospy.loginfo("Starting DroneMovement.")

        self.path = []

        self.threshold = 0.01
        self.degree_threshold = 5

        self.clearance_distance = 0.9

        self.goal = None
        self.last_position_command = None
        self.current_target = 0
        self.passing_gate = False

        self.checkpoints = []

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
        self.gates = [[g["position"][0] - np.cos(np.deg2rad(g["heading"] - 90)) * 0.3, g["position"][1] - np.sin(np.deg2rad(g["heading"] - 90)) * 0.3, g["position"][0] + np.cos(np.deg2rad(g["heading"] - 90)) * 0.3, g["position"][1] + np.sin(np.deg2rad(g["heading"] - 90)) * 0.3] for g in world["gates"]]
        self.walls = [[w["plane"]["start"][0], w["plane"]["start"][1], w["plane"]["stop"][0], w["plane"]["stop"][1]] for w in world["walls"]]

        self.starting_time = timer()
        rospy.loginfo("Node finished initialization")

    #----------------------------------------------------#
    #                  Helper functions                  #
    #----------------------------------------------------#

    def plan_path(self, current_position, target_position):
        path_x, path_y, path_z, path_yaw = a_star.aStarPlanning(current_position.x, current_position.y, target_position.x, target_position.y)
        self.checkpoints = []

        for i in range(len(path_x)):
            if i < len(path_x) - 1:
                if path_z[i] != path_z[i + 1]:
                    checkpoint = Position()
                    checkpoint.x = path_x[i]
                    checkpoint.y = path_y[i]
                    checkpoint.z = path_z[i]
                    checkpoint.yaw = path_yaw[i]
                    self.checkpoints.append(checkpoint)

                    # Add same point with next checkpoint height to level the drone before moving
                    additional_checkpoint = Position()
                    additional_checkpoint.x = path_x[i]
                    additional_checkpoint.y = path_y[i]
                    additional_checkpoint.z = path_z[i + 1]
                    additional_checkpoint.yaw = path_yaw[i]
                    self.checkpoints.append(additional_checkpoint)
                else:
                    checkpoint = Position()
                    checkpoint.x = path_x[i]
                    checkpoint.y = path_y[i]
                    checkpoint.z = path_z[i]
                    checkpoint.yaw = path_yaw[i]

                    self.checkpoints.append(checkpoint)
            else:
                checkpoint = Position()
                checkpoint.x = path_x[i]
                checkpoint.y = path_y[i]
                checkpoint.z = path_z[i]
                checkpoint.yaw = path_yaw[i]

                self.checkpoints.append(checkpoint)
        
        self.goal = self.checkpoints[0]
        self.passing_gate = True

        print("Plan finished")
        print(self.checkpoints)

    def determinant(self, v1, v2, v3, v4):
        return (v1*v4-v2*v3)

    def check_if_no_obstacle(self, start_x, start_y, goal_x, goal_y):
        for g in range(len(self.gates)):
            delta = self.determinant(start_x-goal_x, self.gates[g][0]-self.gates[g][2], start_y-goal_y, self.gates[g][1]-self.gates[g][3])
            if delta > 0.000001 or delta < -0.000001:
                return False

            namenda = self.determinant(self.gates[g][0]-goal_x, self.gates[g][0]-self.gates[g][2], self.gates[g][1]-goal_y, self.gates[g][1]-self.gates[g][3]) / delta
            if namenda <= 1 and namenda >= 0:
                return False
            
            miu = self.determinant(start_x-goal_x, self.gates[g][0]-goal_x, start_y-goal_y, self.gates[g][1]-goal_y) / delta
            if miu <= 1 and miu >= 0:
                return False

        for w in range(len(self.walls)):
            delta = self.determinant(start_x-goal_x, self.walls[w][0]-self.walls[w][2], start_y-goal_y, self.walls[w][1]-self.walls[w][3])
            if delta > 0.000001 or delta < -0.000001:
                return False

            namenda = self.determinant(self.walls[w][0]-goal_x, self.walls[w][0]-self.walls[w][2], self.walls[w][1]-goal_y, self.walls[w][1]-self.walls[w][3]) / delta
            if namenda <= 1 and namenda >= 0:
                return False
            
            miu = self.determinant(start_x-goal_x, self.walls[w][0]-goal_x, start_y-goal_y, self.walls[w][1]-goal_y) / delta
            if miu <= 1 and miu >= 0:
                return False
        
        return True

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

        if not self.tf_buf.can_transform("cf1/odom", "map", target.header.stamp, timeout=rospy.Duration(0.3)):
            rospy.logwarn_throttle(5.0, "No transform from map to odom")
            
            # Send last command
            if self.last_position_command is not None:
                self.position_publisher.publish(self.last_position_command)
                print("Using last position")
            else:
                self.hover_in_place(0.0)
                print("Hovering")
        else:
            target_odom = self.tf_buf.transform(target, "cf1/odom", rospy.Duration(0.3))

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

    #----------------------------------------------------#
    #                  Callback functions                #
    #----------------------------------------------------#

    def pose_callback(self, msg):
        if not self.tf_buf.can_transform("map", msg.header.frame_id, msg.header.stamp, timeout=rospy.Duration(0.3)):
            rospy.logwarn_throttle(5.0, "No transform from %s to map" % msg.header.frame_id)
            return

        msg_map = self.tf_buf.transform(msg, "map", rospy.Duration(0.3))

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

        # print("expected position map: " + str(self.goal))
        # print("current position map: " + str(detected_position))

        if self.goal is None:
            self.plan_path(detected_position, self.path[self.current_target])
        else:
            anglediff = (detected_position.yaw - self.goal.yaw + 180 + 360) % 360 - 180
            idx = self.checkpoints.index(self.goal) if self.goal in self.checkpoints else -1

            if self.goal.x + self.threshold > detected_position.x > self.goal.x - self.threshold and self.goal.y + self.threshold > detected_position.y > self.goal.y - self.threshold and self.goal.z + self.threshold > detected_position.z > self.goal.z - self.threshold and anglediff <= self.degree_threshold and anglediff >= -self.degree_threshold:
            # if self.goal.x + self.threshold > detected_position.x > self.goal.x - self.threshold and self.goal.y + self.threshold > detected_position.y > self.goal.y - self.threshold and self.goal.z + self.threshold > detected_position.z > self.goal.z - self.threshold:
                if idx != -1 and idx < len(self.checkpoints) - 1:
                    self.goal = self.checkpoints[idx + 1]
                    print("Next checkpoint")
                else:
                    if self.passing_gate:
                        if self.goal.z == 0.41:
                            angle_to_pass_gate = np.deg2rad(self.path[self.current_target].yaw)
                            
                            other_side_of_gate = Position()
                            other_side_of_gate.x = detected_position.x + self.clearance_distance * np.cos(angle_to_pass_gate)
                            other_side_of_gate.y = detected_position.y + self.clearance_distance * np.sin(angle_to_pass_gate)
                            other_side_of_gate.z = 0.41
                            # other_side_of_gate.yaw = self.path[self.current_target].yaw
                            other_side_of_gate.yaw = self.goal.yaw

                            self.goal = other_side_of_gate
                            self.current_target = self.current_target + 1
                            self.passing_gate = False
                            print("Passing gate")
                        else:
                            self.goal.z = 0.41
                            print("Leveling drone before passing the gate")
                    else:
                        self.plan_path(detected_position, self.path[self.current_target])
            else:
                if idx != -1 and idx < len(self.checkpoints) - 1:
                    next_goal = self.checkpoints[idx + 1]

                    if abs(self.goal.x - detected_position.x) + abs(self.goal.y - detected_position.y) >= abs(next_goal.x - detected_position.x) + abs(next_goal.y - detected_position.y):
                        if self.check_if_no_obstacle(detected_position.x, detected_position.y, next_goal.x, next_goal.y):
                            self.goal = self.checkpoints[idx + 1]

if __name__ == "__main__":
    ros_node = DroneMovement()
    
    rate = rospy.Rate(20) # Hz

    while not rospy.is_shutdown():
        if ros_node.goal is None:
            ros_node.hover_in_place(0.0)
        else:
            ros_node.move()
        rate.sleep()