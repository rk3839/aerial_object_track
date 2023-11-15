import cv2
import numpy as np
face_cascade=cv2.CascadeClassifier('haar.xml')
cap=cv2.VideoCapture('test video sample.mp4')


_, frame = cap.read()
rows, cols, _ = frame.shape
x_medium = int(cols / 2)
y_medium = int(rows / 2)
center = int(cols / 2)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('r_1.mp4',fourcc, 5, (1280,720))

while True:
    _,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,1.3,5)
    for(x,y,w,h)in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        font=cv2.FONT_HERSHEY_SIMPLEX
        #cv2.putText(img,'ram',(200,200),font,2,(255,255,0),2,cv2.LINE_AA)
        roi_gray=gray[y:y+h,x:x+w]
        roi_color=frame[y:y+h,x:x+w]
    frame1 = frame

    hsv_frame=cv2.cvtColor(frame1,cv2.COLOR_BGR2HSV)
    low_green=np.array([36,25,25])
    high_green=np.array([70,255,255])
    green_mask=cv2.inRange(hsv_frame,low_green,high_green)
    contours, _ = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x:cv2.contourArea(x), reverse=True)
    cv2.line(frame1,(10,30),(10,370),(200,0,155),2) #y-axis
    cv2.line(frame1,(10,800),(10,430),(200,0,155),2) #y-axis
    cv2.line(frame1,(30,15),(680,15),(200,0,155),2) #x-axis
    cv2.line(frame1,(830,15),(1480,15),(200,0,155),2) #x-axis
    cv2.line(frame1,(30,15),(50,30),(200,0,155),2) #left upper arrow for x
    cv2.line(frame1,(30,15),(50,5),(200,0,155),2) #left upper arrow for x
    cv2.line(frame1,(1460,25),(1480,15),(200,0,155),2) #right upper arrow for x
    cv2.line(frame1,(1460,5),(1480,15),(200,0,155),2) #right upper arrow for x
    cv2.line(frame1,(10,30),(20,50),(200,0,155),2) #left upper arrow for y
    cv2.line(frame1,(10,30),(1,50),(200,0,155),2) #left upper arrow- for y
    cv2.line(frame1,(10,800),(20,780),(200,0,155),2) #left lower arrow for y
    cv2.line(frame1,(10,800),(1,780),(200,0,155),2) #left lower arrow for y
    font=cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame1,'X-AXIS',(700,30),font,1,(200,255,155),2,cv2.LINE_AA)
    cv2.putText(frame1,'Y-AXIS',(5,410),font,1,(200,255,155),2,cv2.LINE_AA)

    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)
        x_medium = int((x + x + w) / 2)
        y_medium = int((y + y + h) / 2)
        area=cv2.contourArea(cnt)
        if area > 50:
            cv2.line(frame1, (x_medium, 0), (x_medium, 1070), (0, 255, 0), 2)
            cv2.line(frame1, (0, y_medium), (1920, y_medium), (0, 255, 0), 2)
            cv2.rectangle(frame1,(x,y),(x+w,y+h),(255,0,0),2)
            #cv2.circle(frame1,(x+w,y+h),10,(255,0,0),2)
    
        #if ((x_medium < center -2)&(y_medium < center -4)):
        #    cv2.putText(frame1, "location of object: {}".format('left side up'), (100, 290), cv2.LINE_AA,
        #                1, (0, 255, 255), 2)
            #print ('left side up')
        #elif ((x_medium < center -2)&(y_medium < center +4)):
        #    cv2.putText(frame1, "location of object: {}".format('left side down'), (100, 290), cv2.LINE_AA,
        #                1, (0, 255, 255), 2)
            #print ('left side down')
        #elif ((x_medium < center +2)&(y_medium > center +4)):
        #    cv2.putText(frame1, "location of object: {}".format('right side down'), (100, 290), cv2.LINE_AA,
        #                1, (0, 255, 255), 2)
            #print ('right side down')
        #elif ((x_medium < center +2)&(y_medium < center -4)):
        #    cv2.putText(frame1, "location of object: {}".format('right side up'), (100, 290), cv2.LINE_AA,
        #                1, (0, 255, 255), 2)
            #print ('right side up')
        
        x=str(x_medium)
        y=str(y_medium)
        cv2.putText(frame1, "location in X-Axis: {} pixels".format(x_medium), (100, 160), cv2.LINE_AA,
                    1, (0, 255, 255), 2)
        cv2.putText(frame1, "location in Y-Axis: {} pixels".format(y_medium), (100, 240), cv2.LINE_AA,
                    1, (0, 255, 255), 2)
    
    cv2.imshow('Frame',frame1)
    #print(x_medium, y_medium)
    b = cv2.resize(frame1,(1280,720),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
    out.write(b)

    
    if cv2.waitKey(100) == ord('q'):
        break
    
cap.release()
out.release()
cv2.destroyAllWindows()
