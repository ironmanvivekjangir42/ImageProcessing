#import necessary modules
import cv2
import numpy as np

#region of interest
top, right, bottom, left = 60, 350, 275, 590
top1, right1, bottom1, left1 = 60, 500, 275, 590
camera =cv2.VideoCapture(0)#capture video feed ,0 for default camera

font=cv2.FONT_HERSHEY_COMPLEX

while(True):
    (grabbed, frame) =camera.read()
    frame = cv2.flip(frame, 1)
    
    #region of interest
    roi = frame[top:bottom, right:left]
    roi1 = frame[top1:bottom1, right1:left1]

    #convert region of interest to gray scale    
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray= cv2.GaussianBlur(gray, (7, 7), 0)#apply gausion blur to the region of interest

    #draw rectanglr around region of interest    
    cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
    _, thresh = cv2.threshold(gray, 220, 255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        mask = np.ones(frame.shape[:2], dtype="uint8") * 255
        #finding convex hull for contours of the hand
        hull = cv2.convexHull(cnt)
        M = cv2.moments(thresh)
 
        # calculate x,y coordinate of center
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        approx =cv2.approxPolyDP(hull,0.0095*cv2.arcLength(cnt,True),True)        
        cv2.drawContours(roi, [approx],0,(170,0,34),2)
        cv2.circle(roi,(cX,cY), 100, (0,0,255), 1)
        print(len(approx))

        #using length function to predict number of fingers
        if len(approx)== 4:
            cv2.putText(frame,"two",(400,350),font,1,(0,0,255))

        if len(approx)== 5 :
            cv2.putText(frame,"three",(400,350),font,1,(0,0,255))

        if len(approx)== 6 :
            cv2.putText(frame,"four",(400,350),font,1,(0,0,255))

        if len(approx)== 7:
            cv2.putText(frame,"five",(400,350),font,1,(0,0,255))
    
    cv2.imshow("Video Feed", frame)
    cv2.imshow("Video section", thresh)
    
    keypress = cv2.waitKey(1) & 0xFF
    if keypress == ord("q"):
            break


camera.release()
cv2.destroyAllWindows()
