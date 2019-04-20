#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import rospy
import cv2
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PoseStamped
from tf.transformations import euler_from_quaternion, quaternion_from_euler

class Localizer:
	"""
	Localizer provides the function used to localize a detected object in the map frame.
	It has to be instantiated once when the node is created (i.e. not everytime the callback
	is invoked.).
	"""
	def __init__(self): 

		self.tf_buf   = tf2_ros.Buffer()
		self.tf_lstn  = tf2_ros.TransformListener(self.tf_buf)

		self.camera_matrix = np.array(
			[[222.91335402, 0.000000, 325.80327746],
			[0.0000000, 221.76372749, 236.02284187],
			[0.0000000, 0.000000, 1.0000000]])

		self.distortion_model = np.array([2.47410560e-01, -3.03099446e-01, -2.62596533e-04, -8.35380154e-04, 9.12214101e-02])
		self.rectification_matrix = np.array(
								[[1.000000, 0.000000, 0.000000], 
								[0.000000, 1.000000, 0.000000], 
								[0.000000, 0.000000, 1.000000]])

		self.projection_matrix = np.array(
			[[222.91335402, 0.0, 325.80327746, 0.0],
			[0.0, 221.76372749, 236.02284187, 0.0],
			[0.0, 0.0, 1.0, 0.0]])
		#Standard dimension of a sign
		rw = 0.2 # 20 cm
		rh = 0.2 # 20 cm
		center = [0,0]
		self.template_points = np.array([
								(0				, 0				 , 0), # center
								(center[0]-rw/2, center[1]-rh/2, 0), # top left
								(center[0]+rw/2, center[1]-rh/2, 0), # top right
								(center[0]-rw/2, center[1]+rh/2, 0), # lower left
								(center[0]+rw/2, center[1]+rh/2, 0)  # lower right

							])
		
		
	def find_location(self, center, box, stamp):
		"""
		Finds the location of an object in the map frame given its center, its bounding box,
		and the timestamp at which the object was seen.	
		Returns the transformed PoseStamped in the map frame of the detected object, or None if
		it couldn't be transformed.	
		"""
		if not self.tf_buf.can_transform('cf1/camera_link', 'map', stamp, rospy.Duration(0.9)):
			rospy.logwarn_throttle(5.0, 'No transform from cf1/camera_link to map')

			return None
		
		rospy.loginfo("HEEEEEEY")
		imgPoints = np.array([
								(center[0]	, center[1]), # center
								(box[0][0]	, box[0][1]), # 1st point
								(box[1][0] 	, box[1][1]), # 2nd point
								(box[2][0]	, box[2][1]), # 3rd point
								(box[3][0]	, box[3][1])  # 4rth point
							], dtype = "double")

		
		rvec = []
		tvec = []
		(success, rotation_vector, translation_vector) = cv2.solvePnP(self.mdlPoints, imgPoints, self.camera_matrix, self.distortion_model)
		
		# Build PoseStamped of the object and transform it to the map frame
		pose = PoseStamped()
		pose.header.frame_id = "cf1/camera_link"
		pose.header.stamp = stamp
		pose.pose.position.x = translation_vector[0][0]
		pose.pose.position.y = translation_vector[1][0]
		pose.pose.position.z = translation_vector[2][0]

		(pose.pose.orientation.x, 
		pose.pose.orientation.y, 
		pose.pose.orientation.z, 
		pose.pose.orientation.w)  = quaternion_from_euler(rotation_vector[0], rotation_vector[1], rotation_vector[2])
		return self.tf_buf.transform(pose, 'map') # returns the sign pose in "map" frame

