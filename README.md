# ASL-hand-posture-detection
This project is about detect the hand posture using Raspberry pi4, pi camera with coral.  
Full report is written on https://s-nathnaree.medium.com/asl-hand-posture-detection-using-camera-for-communication-ca895818da02
## Software and model
### Software
Python3.7  
Python — numpy  
TensorflowLite  
openCV  
### Model
MediaPipe — Hand Landmark Model TFLite (source: https://google.github.io/mediapipe/solutions/models.html)  
Google Coral — libedgetpu (source: https://github.com/google-coral/libedgetpu)
## Implementation
1. collect the data: handPosedDetection.py
2. training model: trainingModel.py
3. testing model: testingModel.py
4. convert model to tflite (to use the model with Raspberry pi): get_tflite

## Project Member
This project is the part of Hardware design, TAIST- Tokyo Tech 2021  
6322040301 — Nathnaree Smunyahirun  
6314552770 — Satida Sookpong  
6322040426 — Pumipach Tanachotnarangkun  
