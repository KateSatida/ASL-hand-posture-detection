#
#   20210408 - Pumipach (protan) Tanachotnarangkun
#       This script is implemented to test and predict hand alphabet postures
#

#
#   IMPORT
#

import os
import cv2
import pandas as pd
import numpy as np
import tensorflow as tf
import mediapipe as mp

#
#   GLOBAL VARIABLE
#

#	filter out warning messages from tensorflow
tf.get_logger().setLevel('ERROR')

testingDataFilePath = r'C:\Users\Pumipach\Desktop\testingData.csv'
savedModelFilePath = r'C:\Users\Pumipach\Desktop\trainedModel\trainedModel.model'

dataNameList = [ "x1", "y1", "z1","x2", "y2", "z2",
				"x3", "y3", "z3","x4", "y4", "z4",
				"x5", "y5", "z5","x6", "y6", "z6",
				"x7", "y7", "z7","x8", "y8", "z8",
				"x9", "y9", "z9","x10", "y10", "z10",
				"x11", "y11", "z11","x12", "y12", "z12",
				"x13", "y13", "z13","x14", "y14", "z14",
				"x15", "y15", "z15","x16", "y16", "z16",
				"x17", "y17", "z17","x18", "y18", "z18",
				"x19", "y19", "z19","x20", "y20", "z20",
				"x21", "y21", "z21", "label" ]

#
#   HELPER FUNCTION
#

def getLandmarkFromIdx( landmarkObj, index ):
	'''	this function will return normalized x, y, z position from landmark object
			from each index input
	'''
	#	get landmark object inside
	landmark = landmarkObj.landmark

	#	make sure the index input is between the range of 
	assert( index >= 0 )
	assert( index <= len( landmark ) )

	return np.array( [ landmark[index].x, landmark[index].y, landmark[index].z ] )

def getNormalizedScaleData( dataFrame ):
	'''	this function will normalized all data to the scale of each hand only
	'''
	dataFrame['x'] = ( dataFrame['x'] - dataFrame['x'].min() ) / ( dataFrame['x'].max() - dataFrame['x'].min() )
	dataFrame['y'] = ( dataFrame['y'] - dataFrame['y'].min() ) / ( dataFrame['y'].max() - dataFrame['y'].min() )
	dataFrame['z'] = ( dataFrame['z'] - dataFrame['z'].min() ) / ( dataFrame['z'].max() - dataFrame['z'].min() )
	return dataFrame

#
#   MAIN
#

def main():

	#	create model layers
	data_model = tf.keras.Sequential( [ 
		tf.keras.layers.Dense( 128, activation='relu' ),
		tf.keras.layers.Dense( 128, activation='relu' ),
		tf.keras.layers.Dense( 26 )
	] )

	#	set up model
	data_model.compile( optimizer='adam',
		  loss=tf.keras.losses.SparseCategoricalCrossentropy( from_logits=True ),
		  metrics=[ 'accuracy' ] )

	data_model.load_weights( savedModelFilePath )

	# tf.saved_model.save( data_model, r'C:\Users\Pumipach\Desktop\savedModel' )
	# tf.keras.models.save_model( data_model, r'C:\Users\Pumipach\Desktop\savedModel.h5' )
	# print('finish!')
	# return

	#	get testing data
	data_test = pd.read_csv( testingDataFilePath, encoding='ANSI', names=dataNameList )
	data_test.drop(columns=data_test.columns[-1], 
		axis=1, 
		inplace=True)

	#   initial object for mediapipe
	drawer = mp.solutions.drawing_utils
	myhands = mp.solutions.hands
	
	#	set up camera
	cap = cv2.VideoCapture(0)

	#	set up hand object as $hands
	with myhands.Hands(
		min_detection_confidence=0.5,
		min_tracking_confidence=0.5) as hands:

		#	loop over the camera read for video capturing
		while cap.isOpened():
			success, image = cap.read()

			if not success:
				print("Ignoring empty camera frame.")
				# If loading a video, use 'break' instead of 'continue'.
				continue

			# Flip the image horizontally for a later selfie-view display, and convert
			# the BGR image to RGB.
			image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

			# To improve performance, optionally mark the image as not writeable to
			# pass by reference.
			image.flags.writeable = False
			results = hands.process(image)

			# Draw the hand annotations on the image.
			image.flags.writeable = True
			image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
			if results.multi_hand_landmarks:
				for hand_landmarks in results.multi_hand_landmarks:
					drawer.draw_landmarks( image, hand_landmarks, myhands.HAND_CONNECTIONS)
				cv2.imshow('MediaPipe Hands', image)
			
			#	get key
			keyCV = cv2.waitKey(33)
			
			#	stop the code when pressing 'esc'
			if keyCV == 27:
				break
			
			elif keyCV == -1:
				#	skip the general key
				continue
			
			else:
				
				#	get a hand
				try:
					aHand = results.multi_hand_landmarks[0]
				except TypeError:
					#	skip error
					print( 'Warning! It has no hand detected!' )
					continue
					
				#	 initial list to contain all normalized data
				currentPosList = list()

				#	loop over the index
				for idx in range( 21 ):
					curPos = getLandmarkFromIdx( aHand, idx )
					currentPosList.append( curPos )

				#	inital pandas and normalize data
				df = pd.DataFrame( np.array( currentPosList ), columns=[ 'x', 'y', 'z' ] )
				df = getNormalizedScaleData( df )

				#	predict the result
				predictedResultList = data_model.predict( np.array( [ df.values.flatten(), ] ) )
				predictedChar = chr( predictedResultList.argmax(axis=1)[0] + ord('a') )
				print( df.values.flatten() )
				#	print out the result
				print( "The predicted result is character of \'{}\'".format( predictedChar ) )
				continue

		#	release camera
		cap.release()
	
#
#   running
#

main()