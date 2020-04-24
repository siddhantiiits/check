#For capturing dataset
#Siddhant Tiwari

import cv2
import os

cam = cv2.VideoCapture(0)
cam.set(3, 640) # width
cam.set(4, 480) # height

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


face_id = input('\n enter user id end press <return> ==>  ') #id in form of integers for each person

print("\n Initializing Cam. Look at the camera and wait ...")
# Initialize individual sampling face count
count = 0

while(True):
    ret, img = cam.read()
    #img = cv2.flip(img, -1) # flip video image vertically
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
        count += 1


        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w]) # Save the captured samples into the dataset folder

        cv2.imshow('image', img)


    k = cv2.waitKey(100) & 0xff # ord = ESC for exiting
    if k == 27:
        break
    elif count >= 50: # Take 30 face sample and stop video
         break

# Do a bit of cleanup
print("\n Thankyou! Registered your face!")
cam.release()
cv2.destroyAllWindows()