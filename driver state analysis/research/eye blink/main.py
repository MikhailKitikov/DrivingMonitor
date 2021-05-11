from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import playsound
import argparse
import imutils
import time
import dlib
import cv2
import pyttsx3


def sound_alarm(path, engine):
	playsound.playsound(path)
	engine.say("Please! Do not sleep while driving!")
	engine.runAndWait()
	playsound.playsound(path)


def eye_aspect_ratio(eye):
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	C = dist.euclidean(eye[0], eye[3])
	return (A + B) / (2.0 * C)


def rotate_image(image, angle):
	image_center = tuple(np.array(image.shape[:2]) / 2)
	rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
	result = cv2.warpAffine(image, rot_mat, image.shape[:2], flags=cv2.INTER_CUBIC)
	return result


# hyperparameters
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 48

MODELS = ['CNN', 'HOG']
MODEL_TYPE = MODELS[1]

COUNTER = 0
ALARM_ON = False

# init speaker
engine = pyttsx3.init()
engine.setProperty('volume', 1.0)


print("[INFO] loading face detector...")
if MODEL_TYPE == 'CNN':
	detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")
else:
	detector = dlib.get_frontal_face_detector()


print("[INFO] loading landmark detector...")
if MODEL_TYPE == 'CNN':
	predictor = dlib.shape_predictor("shape_predictor_5_face_landmarks.dat")
else:
	predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


# get eye detections
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


print("[INFO] starting video stream thread...")
vs = VideoStream(src=0).start()

while True:
	# get frame
	frame = vs.read()
	frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# detect faces
	rects = detector(gray, 0)

	if not rects:
		cv2.putText(frame, "No faces detected!", (120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

	else:	
		for ind, rect in enumerate(rects):

			# detect landmarks

			if MODEL_TYPE == 'CNN':
				rect = rect.rect

			# draw face boundaries

			pt1 = (rect.left(), rect.top())
			pt2 = (rect.right(), rect.bottom())

			cv2.rectangle(frame, pt1, pt2, (255, 255, 255), 1)

			# get keypoints

			shape = predictor(gray, rect)
			shape = face_utils.shape_to_np(shape)

			# visualize keypoints
			for (x, y) in shape:
				cv2.circle(frame, (x, y), 1, (255, 255, 255), 2)
			
			# extract eye coordinates
			leftEye = shape[lStart: lEnd]
			rightEye = shape[rStart: rEnd]
			
			# compute average eye aspect ratio
			ear = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2
			
			# visualize eyes
			cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (0, 255, 0), 1)
			cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0, 255, 0), 1)
			
			# check if eye is closed now
			if ear < EYE_AR_THRESH:
				COUNTER += 1
				
				# check if eye is closed too long
				if COUNTER >= EYE_AR_CONSEC_FRAMES:
					
					# start alarm in new thread
					if not ALARM_ON:
						ALARM_ON = True
						alarm_thread = Thread(target=sound_alarm, args=("alarm.wav", engine))
						alarm_thread.deamon = True
						alarm_thread.start()
							
					# draw an alarm on the frame
					cv2.putText(frame, "DROWSINESS ALERT!", (120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
					
			# otherwise, reset counter
			else:
				COUNTER = 0
				ALARM_ON = False

	# show frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# check if break pressed
	if key == ord("q"):
		break

cv2.destroyAllWindows()
vs.stop()
engine.stop()
