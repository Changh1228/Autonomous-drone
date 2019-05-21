#!/usr/bin/env python

import math
import rospy
import tf2_ros
import tf2_geometry_msgs
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from geometry_msgs.msg import PoseStamped, TransformStamped, Vector3
from aruco_msgs.msg import MarkerArray
from crazyflie_driver.msg import Position

goal = None

def transform_callback(data):

    

    # transform pose in camera link to map link
    goal = PoseStamped()
    goal.header.stamp = rospy.Time.now()
    #print(goal.header.stamp)
    #goal.header = data.markers[0].header
    goal.header.frame_id = "cf1/camera_link"
    goal.pose = data.markers[0].pose.pose
    #rospy.sleep(0.02)

    tf_result = tf_buf.transform(goal, 'cf1/odom', rospy.Duration(10.0))

    # TF form aruco to map
    br2 = tf2_ros.TransformBroadcaster()
    tfA2M = TransformStamped()
    tfA2M.header.stamp = rospy.Time.now()
    tfA2M.header.frame_id = "map"
    tfA2M.child_frame_id = "aruco/detected 0"
    tfA2M.transform.translation = tf_result.pose.position
    tfA2M.transform.rotation = tf_result.pose.orientation
    br2.sendTransform(tfA2M)


rospy.init_node('detectMarkers')
sub_goal = rospy.Subscriber('/aruco/markers', MarkerArray, transform_callback)
tf_buf   = tf2_ros.Buffer()
tf_lstn  = tf2_ros.TransformListener(tf_buf)

def main():
    rate = rospy.Rate(10)  # Hz
    rospy.init_node('detectMarkers')
    rospy.Subscriber('/aruco/markers', MarkerArray, transform_callback)
    rospy.spin()

if __name__ == '__main__':
    main()
