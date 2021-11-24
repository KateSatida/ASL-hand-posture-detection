#
#   20210408 - Pumipach (protan) Tanachotnarangkun
#       This script is implemented to convert model to tflite for Raspberry pi

#
#   IMPORT
#

import tensorflow as tf


#
#   GLOBAL VARIABLE
#

savedModelDirPath = r'C:\Users\Pumipach\Desktop\savedModel'
outputFilePath = r'C:\Users\Pumipach\Desktop\handPostDetection.tflite'

#
#   HELPER FUNCTION
#

#
#   MAIN
#

def main():

	# Convert the model
	converter = tf.lite.TFLiteConverter.from_saved_model( savedModelDirPath ) # path to the SavedModel directory
	tflite_model = converter.convert()

	# Save the model
	with open( outputFilePath, 'wb' ) as f:
		f.write( tflite_model )
	
#
#   running
#

main()