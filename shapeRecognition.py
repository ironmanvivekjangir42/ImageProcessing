#importing necessary libraries
import cv2
import numpy as np

#reading the images with basic shapes
img= cv2.imread('shape.jpg',cv2.IMREAD_GRAYSCALE)

#finding contours for the object
_, threshold=cv2.threshold(img,100,255,cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(threshold,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

font=cv2.FONT_HERSHEY_COMPLEX

for cnt in contours:
    approx =cv2.approxPolyDP(cnt,0.007*cv2.arcLength(cnt,True),True)#approximating the cotours
    cv2.drawContours(img, [approx],0,(0,0,0),2)#drawing approximated contours on our image

    #finding the length of approximated contours predicting the shape
    if len(approx)== 3:
        x=approx.ravel()[0]
        y=approx.ravel()[1]
        cv2.putText(img,"Triangle",(x,y),font,1,(0))

    if len(approx)== 4:
        x=approx.ravel()[0]
        y=approx.ravel()[1]
        cv2.putText(img,"square",(x,y),font,1,(0))

    elif len(approx) >=5:
        x=approx.ravel()[0]
        y=approx.ravel()[1]
        cv2.putText(img,"circle",(x,y),font,1,(0))

#output 
cv2.imshow("shapes",img)
cv2.imshow("mask",mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
