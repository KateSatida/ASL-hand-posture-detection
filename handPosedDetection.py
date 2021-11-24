#
#   IMPORT
#

import sys
import cv2
import pandas as pd
import numpy as np
import mediapipe as mp

#
#   GLOBAL VARIABLE
#

NumInputArgs = 0
outputFilePathStr = r'C:\Users\Pumipach\Desktop\trainingData_.csv'

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

	#   init args
	args = sys.argv
	programPathStr = args[ 0 ]

	#   make sure the input is equal to $NumInputArgs
	if( len( args ) != ( NumInputArgs + 1 ) ):
		raise AssertionError( "Error! - Number of input argument(s) should be {}".format( NumInputArgs ) )

	#   initial object for mediapipe
	drawer = mp.solutions.drawing_utils
	myhands = mp.solutions.hands
	
	#	set up camera
	cap = cv2.VideoCapture(0)

	#	create master list to contain all data
	masterArr = list()

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

				#	write all data
				exportedDf = pd.DataFrame( masterArr )

				try:
					outputFile = open( outputFilePathStr, 'a' )
				except PermissionError:
					outputFile = open( r'C:\Users\Pumipach\Desktop\trainingData_dummy.csv', 'w' )

				outputFile.write( exportedDf.to_csv( index=False, header=False, line_terminator='\n' ) )
				outputFile.close()
				break
			
			elif keyCV == -1:
				#	skip the general key
				continue
			
			else:
				
				#	get label
				character = chr( keyCV )
				
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

				#	flatten the data and keep in the master array
				flattendDf = df.values.flatten()
				flattendDf = np.append( flattendDf, [ keyCV - ord('a') ] )
				masterArr.append( flattendDf )

				#	print the progress and continue
				print( '{} is collected!'.format( character ) )
				continue

		#	release camera
		cap.release()

#
#   running
#

main()