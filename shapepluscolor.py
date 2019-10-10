import cv2
import numpy as np
import imutils 

shape0 = "unidentified"
lower_red = np.array([0,50,50])
upper_red = np.array([10,255,255])

lower_blue = np.array([94,80,2])
upper_blue = np.array([102,255,255])

lower_green = np.array([25,52,70])
upper_green = np.array([102,252,255])

def detect(c):#detecting shapess
    shape = "unidentified"
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.04 * peri, True)
    if len(approx) == 3:
        shape = "triangle"
 
    elif len(approx) == 4:
        (x, y, w, h) = cv2.boundingRect(approx)
        ar = w/h
        shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"

    elif len(approx) == 5:
        shape = "pentagon"	
    else:
        shape = "circle"

    return shape

def findcnt(mask, c=None):
	contour = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	contours = imutils.grab_contours(contour)
	mu = [None]*len(contours)
	for i in range(len(contours)):
		mu[i] = cv2.moments(contours[i])
		cx = int((mu[i]['m10'] / (mu[i]['m00'] + 1e-5)))
		cy = int((mu[i]['m01'] / (mu[i]['m00'] + 1e-5)))
		
		shape = detect(contours[i])
		if shape0==shape:
			continue
		area = cv2.contourArea(contours[i])
		if area>3000:
			cv2.drawContours(frame, contours, i, (0,0,0), 2)
			cv2.putText(frame, shape, (cx+50,cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
			cv2.putText(frame, c, (cx,cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
	
cap = cv2.VideoCapture(0)

while(1):
	_,frame = cap.read()
	
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	
	maskR = cv2.inRange(hsv, lower_red, upper_red)
	findcnt(maskR, c= "Red")

	maskB = cv2.inRange(hsv, lower_blue, upper_blue)
	findcnt(maskB, c= "Blue")

	maskG = cv2.inRange(hsv, lower_green, upper_green)
	findcnt(maskG, c= "Green")
	
	cv2.imshow('frame', frame)
	cv2.imshow('red', maskR)
	cv2.imshow('blue', maskB)
	cv2.imshow('green', maskG)
	
	if cv2.waitKey(5) & 0xFF == ord('q'):
		break

cv2.destroyAllWindows()	
