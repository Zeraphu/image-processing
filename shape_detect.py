#!/usr/bin/env python
import numpy as np
import cv2

cap = cv2.VideoCapture(0)
#function to detect sqaure, rectangle and circle shapes
def shapedetect(approx, cnt, cx, cy):
	shape = None
	if len(approx) == 4: 
		(x, y, w, h) = cv2.boundingRect(approx)
		ar = w/h
		shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle" #distinguishing square and rectangle
		if shape != None:
			cv2.putText(frame, shape, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
			cv2.drawContours(frame, [cnt], 0, (0, 255, 0), 1)
	if len(approx)>10: 
		cv2.putText(frame, "Circle", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
		cv2.drawContours(frame, [cnt], 0, (0, 255, 0), 2)	


while True:
	ret, frame = cap.read()
	#shifting to gray, then making canny image 
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
	gray = cv2.blur(gray, (3,3))
	canny = cv2.Canny(gray, 100, 200, True) 

	img, cnts, heir = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) 
	for cnt in cnts:
		if cv2.contourArea(cnt)<1000:
			continue
		peri = cv2.arcLength(cnt, True)
		approx = cv2.approxPolyDP(cnt, 0.01 * peri, True)
		x = approx.ravel()[0] #finding starting points of contour
		y = approx.ravel()[1]
		shapedetect(approx, cnt, x, y) #detecting shapes

	cv2.imshow('cg', canny)
	cv2.imshow('shapes', frame) 
	if cv2.waitKey(50) & 0xFF == ord('q'):
		break
cap.release()
cv2.destroyAllWindows()
