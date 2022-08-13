
import tkinter as tk
from tkinter import *
from tkinter import messagebox
from tkinter.tix import COLUMN
import cv2
from PIL import Image, ImageTk
import os
import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.optimizers import Adam
from keras.layers import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

emotion_model = Sequential()

emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))
emotion_model.load_weights('model.h5')

cv2.ocl.setUseOpenCL(False)

emotion_dict = {0: "   Angry   ", 1: "Disgusted", 2: "  Fearful  ", 3: "   Happy   ", 4: "  Neutral  ",
                5: "    Sad    ", 6: "Surprised"}


# global last_frame1
# last_frame1 = np.zeros((480, 640, 3), dtype=np.uint8)

global cap1
show_text = [0]

cap1 = cv2.VideoCapture(0)

def info():
     messagebox.showinfo('Information', 'Welcome to our emotion recognition program!\nPress start to initiate the program\n Press exit to quit the program\n When you initiate the program, the camera is on, you can press q to quit the camera')

def show_vid():
    global cap1
    while True:
        ret, frame = cap1.read()
        if not ret:
            break
        bounding_box = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        num_faces = bounding_box.detectMultiScale(gray_frame,scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in num_faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
            roi_gray_frame = gray_frame[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
            emotion_prediction = emotion_model.predict(cropped_img)
            maxindex = int(np.argmax(emotion_prediction))
            cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('Video', cv2.resize(frame,(1200,860),interpolation = cv2.INTER_CUBIC))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            exit(0)
    cap1.release()
    
if __name__ == '__main__':
    
    root = Tk()
    root.geometry('800x900')
    root.title("Emotion detection")
   
    #heading
    heading = Label(root, text="Ha Noi University of Science and Technology", pady=20, font=('arial', 20, 'bold'))
    heading.pack()
    heading.place(x=100,y=5)

    
    heading1 = Label(root, text="Graduation Project", font=('arial', 20, 'bold'))
    heading1.pack()
    heading1.place(x=280,y=70)

    heading2 = Label(root, text="Facial Emotion Recognition Program", font=('arial', 20, 'bold'))
    heading2.pack()
    heading2.place(x=150,y=120)
   
    #UniLogo
    hust = PhotoImage(file="hust.png")
    hustLabel = Label(root, image = hust, pady = 5)
    hustLabel.pack()
    hustLabel.place(x=350,y=170)

    
    name = Label(root, text = "Instructor : Dr Nguyen Thanh Huong", font =('arial', 14))
    name.pack()
    name.place(x=480,y=290)

    name1 = Label(root, text = "Student1 : Vu Nguyen Duc Anh", font =('arial', 14))
    name1.pack()
    name1.place(x=480,y=320)

    name2 = Label(root, text = "Student1 : Hoang Ngoc Vu", font =('arial', 14))
    name2.pack()
    name2.place(x=480,y=350)

    #logo
    logo = PhotoImage(file="EmotionDetection.png")
    logoLabel = Label(root, image = logo)
    logoLabel.pack()
    logoLabel.place(x=10, y=310)
   
    #startButton
    start = PhotoImage(file="start3.png")
    startbutton = Button(root, image = start, command = show_vid, borderwidth= 0 )
    startbutton.pack()
    startbutton.place(x=680,y=480)

    #exitButton
    exit1 = PhotoImage(file="exit.png")
    exitbutton = Button(root, image = exit1 , command=root.destroy, borderwidth=0)
    exitbutton.pack()
    exitbutton.place(x = 580, y =480)
    
    #helpButton
    help = PhotoImage(file="help.png")
    helpbutton = Button(root, command = info, image = help , borderwidth=0)
    helpbutton.pack()
    helpbutton.place(x = 480, y= 480)
    
    root.mainloop()