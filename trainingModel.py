#
#   20210408 - Pumipach (protan) Tanachotnarangkun
#       This script is implemented to train and predict hand alphabet postures
#

#
#   IMPORT
#

import cv2
import pandas as pd
import numpy as np
import tensorflow as tf

#
#   GLOBAL VARIABLE
#

trainingDataFilePath = r'C:\Users\Pumipach\Desktop\trainingData.csv'
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

#
#   MAIN
#

def main():

	#	get data testing and 
	data_train = pd.read_csv( trainingDataFilePath, encoding='ANSI', names=dataNameList )

	#	define features and labels
	data_features = data_train.copy()
	data_labels = data_features.pop( "label" )

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

	#	training
	data_model.fit( data_features.values, data_labels.values, epochs=60, batch_size=400 )

	# #	get testing data
	# data_test = pd.read_csv( testingDataFilePath, encoding='ANSI', names=dataNameList )
	# data_test.drop(columns=data_test.columns[-1], 
	# 	axis=1, 
	# 	inplace=True)

	#	save model
	# data_model.save_weights( savedModelFilePath )
	
	tf.saved_model.save( data_model, r'C:\Users\Pumipach\Desktop\savedModel' )
	tf.keras.models.save_model( data_model, r'C:\Users\Pumipach\Desktop\savedModel.h5' )
	print('finish!')


#
#   running
#

main()