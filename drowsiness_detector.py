import cv2
import dlib 
import imutils
import numpy as np
from pygame import mixer
from keras.models import load_model
from imutils import face_utils as f
#Imported all the required libraries

#Calculating Euclidian Distance
def distance_euclidean(x,y):
    a=np.sum(np.square(x-y))  
    return np.sqrt(a)
#Finding Biggest Face in image from all the detected faces
def getface(list_faces):
    if(len(list_faces)==0):
        return
    facial_area=[]
    for myface in list_faces:
        facial_area.append(myface.area())
    return list_faces[facial_area.index(max(facial_area))]
#Calculating Mouth Aspect Ratio
def getMAR(lips):
    deno=distance_euclidean(lips[0],lips[4])
    num=0
    for coord in range(1,4):
        num+=distance_euclidean(lips[coord],lips[8-coord])
    return num/(deno*3)
#Initiating Pygame Music Player
mixer.init()
sound=mixer.Sound(r'data/danger.wav')
sound2=mixer.Sound(r'data/bomb.wav')
#Reading Detector Files
get_face=dlib.get_frontal_face_detector()
fff = cv2.CascadeClassifier(r'data/face.xml')
lll = cv2.CascadeClassifier(r'data/eyeleft.xml')
rrr = cv2.CascadeClassifier(r'data/eyeright.xml')
get_face_points=dlib.shape_predictor("data/shape68.dat")
#Reading Model
model = load_model('data/eye_classifier_cnn.h5')
#Starting WebCam
cap = cv2.VideoCapture(0)
#Setting Display Font
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
#Indxex counters
score,yawn,border,mar=0,0,2,0
#Open Close Predictor variable
rpred=[99]
lpred=[99]
#Infinite Loop to read frames from web cam
while(True):    
    #Reading a frame
    _,frame=cap.read()
    #Finding Dimension of the frame
    height,width=frame.shape[:2]
    #Converting Image to Grayscale from BGR
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #Applying Median Filter
    gray = cv2.medianBlur(gray,5);
    #Getting all faces from Image
    list_faces= get_face(gray, 0)
    #Calculating Biggest Face
    bigface=getface(list_faces)
    if bigface!=None:
        #Getting Facial Landmarks
        shape=get_face_points(gray,bigface)
        #Converting it to Numpy Array
        shape=f.shape_to_np(shape)
        #Extracting Inner Lips Coordinates
        lips=shape[60:68]
        #Calculating Mouth Aspect Ratio
        mar=getMAR(lips)
        #Drawing ROI for mouth
        polygon=cv2.convexHull(lips)
        cv2.drawContours(gray,[polygon],-1,(255,255,255),1)
    #Getting Face Coordinates
    faces=fff.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))
    #Getting Eye Coordinates
    lefteye=lll.detectMultiScale(gray)
    #Getting Eye Coordinates
    righteye=rrr.detectMultiScale(gray)
    #Converting Image to RGB for display
    gray=cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)
    #Drawing notice area to display indices
    cv2.rectangle(gray, (0,height-50),(300,height),(0,0,0),thickness=cv2.FILLED)
    #Drawing ROI for faces
    for (x,y,w,h) in faces:
        cv2.rectangle(gray,(x,y),(x+w,y+h),(0,128,0),3)
    for (x,y,w,h) in righteye:
        #Drawing ROI for Right Eye
        cv2.rectangle(gray,(x,y),(x+w,y+h),(0,0,255),2)
        #Slicing image matrix to extract eye
        eye1=frame[y:y+h,x:x+w]
        #Resizing image to feed it to model
        eye1=(cv2.resize(cv2.cvtColor(eye1,cv2.COLOR_BGR2GRAY),(24,24)))/255
        #Converting image to numpy
        eye1 = np.expand_dims(eye1.reshape(24,24,-1),axis=0)
        #Sending image to model to detect open/close state of eye
        rpred = model.predict_classes(eye1)
    for (x,y,w,h) in lefteye:
        #Drawing ROI for Left Eye
        cv2.rectangle(gray,(x,y),(x+w,y+h),(0,0,255),2)
        #Slicing image matrix to extract eye
        eye2=frame[y:y+h,x:x+w]
        #Resizing image to feed it to model
        eye2=(cv2.resize(cv2.cvtColor(eye2,cv2.COLOR_BGR2GRAY),(24,24)))/255
        #Converting image to numpy
        eye2 = np.expand_dims(eye2.reshape(24,24,-1),axis=0)
        #Sending image to model to detect open/close state of eye
        lpred = model.predict_classes(eye2)
    #Check if both eyes are close
    if(rpred[0]==0 and lpred[0]==0):
        #If both eyes are close, increase eye index by 1
        score=score+1
    else:
        #If one of the or both eyes are open, decrease eye index by 1
        score=score-1


    if(score<0):
        score=0
        #If person is awake, no need for alarm
        sound.fadeout(1)    
    #if sleepy for enough time, call emergency contact
    elif(score>30):
        #Glow the output feed window
        if(border<16):
            border= border+4
        else:
            border=border-2
            if(border<2):
                border=2
        #Display Alert Message
        cv2.putText(gray,'PARKING THE VEHICLE & CALLING YOUR EMERGENCY CONTACT',(70,100), font, 5,(0,0,255),1,cv2.LINE_AA)
        cv2.rectangle(gray,(0,0),(width,height),(0,0,255),border)

    #If more than 800ms eye close, raise alarm
    elif(score>8):
        try:
            sound.play() 
        except:
            pass
        #Glow the output feed window
        if(border<16):
            border= border+2
        else:
            border=border-2
            if(border<2):
                border=2
        #Display Alert Message
        cv2.putText(gray,'ALERT',(70,100), font, 5,(0,0,255),1,cv2.LINE_AA)
        cv2.rectangle(gray,(0,0),(width,height),(0,0,255),border)
    elif(score>7):
        #Display Warning Message
        cv2.putText(gray,'WARNING',(100,130), font, 2,(255,0,0),1,cv2.LINE_AA)




    if(mar>0.34):
        #If MAR is more than 0.34, increase yawn index by 2
        yawn=yawn+2
    else:
        #If MAR is more than 0.34, decrease yawn index by 2
        yawn=yawn-1



    if(yawn<0):
        yawn=0
        #If no yawning, fadeout the alarm sound
        sound2.fadeout(1)
    if(yawn>40):
        yawn=40
    #Displaying indices on output feed
    message="Index Eye: "+str(score)+" Mouth: "+str(yawn)
    cv2.putText(gray,message,(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    if(yawn>10):
        try:
            sound2.play()
        except:
            pass
        #Glow the Output Feed
        if(border<16):
            border= border+2
        else:
            border=border-2
            if(border<2):
                border=2
        #Display Alert Message
        cv2.putText(gray,'Yawning Alert',(70,200), font, 3,(0,0,255),1,cv2.LINE_AA)
        cv2.rectangle(gray,(0,0),(width,height),(0,0,255),border)
    #Display Original Camera Feed
    cv2.imshow('WEBCAM',frame)
    #Display Analysed Feed
    cv2.imshow('DETECTOR',gray)
    #Close Webcam by pressing q
    if cv2.waitKey(1)&0xFF==ord('q'):
        mixer.stop()
        break
#Release Webcam
cap.release()
#Destroying all windows before quitting
cv2.destroyAllWindows()
