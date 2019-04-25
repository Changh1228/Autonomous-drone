#!/usr/bin/env python

import rospy
import math
from std_msgs.msg import Float64
from aruco_msgs.msg import MarkerArray
from sensor_msgs.msg import Imu


'''
uncertainty means how reliable is the localization is
If no marker is seen, the uncertainty increase (faster if drone is moving)
If find marker, the uncertainty decrease (faster if marker is nearer and drone moving slower)
the region of unc is 0-100
Note: only care about z angular_velocity and x, y linear_acceleration when concern uncertainty
'''
aruco_marker = None
speed = 0


def aruco_callback(data):
    global aruco_marker
    aruco_marker = data


def imu_callback(data):
    global speed  # "speed" of drone
    speed = math.sqrt(pow(data.angular_velocity.z, 2) +
                      pow(data.linear_acceleration.x, 2) +
                      pow(data.linear_acceleration.y, 2))


def main():
    global aruco_marker, speed

    rospy.init_node('uncertainty')
    rospy.Subscriber('/aruco/markers', MarkerArray, aruco_callback)
    rospy.Subscriber('/cf1/imu', Imu, imu_callback)
    pub = rospy.Publisher('/localization/unc', Float64, queue_size=10)
    rate = rospy.Rate(10) # 10hz
    print('Uncertainty running')

    uncertainty = 0
    unc_imu = 0
    while not rospy.is_shutdown():
        if aruco_marker == None:
            # when starting no marker is updated, uncertainty is large
            uncertainty = 100
            pub.publish(100.0)
            continue
        else:
            # find nearest aruco
            d = 10000
            id = None
            for aruco in aruco_marker.markers:  # for all markers
                if aruco.id > 15:
                    continue
                d_now = math.sqrt(pow(aruco.pose.pose.position.x, 2) +
                                  pow(aruco.pose.pose.position.y, 2) +
                                  pow(aruco.pose.pose.position.z, 2))
                if d_now < d:
                    d = d_now  # max distance and its ID
                    id = aruco.id
            delta_t = rospy.Time.now().secs - aruco_marker.header.stamp.secs

            # uncertainty from measurement (camera)
            if delta_t < 1:
                uncertainty += (speed- 5 / d)
            else:
                uncertainty += speed
            if uncertainty < d*10:
                uncertainty = d*10
            elif uncertainty > 100:
                uncertainty = 100.0

            #print(uncertainty)
            pub.publish(uncertainty)
        rate.sleep()



if __name__ == '__main__':
    main()