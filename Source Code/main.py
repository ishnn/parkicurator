from tkinter import *
import ctypes,os
from PIL import ImageTk, Image
import tkinter.messagebox as tkMessageBox
from tkinter import filedialog
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os,cv2 
from audio import *
# import required libraries
import sounddevice as sd
from scipy.io.wavfile import write
import wavio as wv
import cv2
import time
import math as m
import mediapipe as mp

model1 = load_model('./Model/wave.h5')
model2 = load_model('./Model/Spiral.h5')
new_model = xgb.Booster(model_file='./Model/audio_model.json')

dataoverall = {'spiral':None,'wave':None,'audio':None,'video':None}

# Calculate distance
def findDistance(x1, y1, x2, y2):
    dist = m.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return dist

# Calculate angle.
def findAngle(x1, y1, x2, y2):
    theta = m.acos((y2 - y1) * (-y1) / (m.sqrt(
        (x2 - x1) ** 2 + (y2 - y1) ** 2) * y1))
    degree = int(180 / m.pi) * theta
    return degree


def mainvid(file_name=0):        
    # Constants and initializations
    font = cv2.FONT_HERSHEY_SIMPLEX
    blue = (255, 127, 0)
    red = (50, 50, 255)
    green = (127, 255, 0)
    dark_blue = (127, 20, 0)
    light_green = (127, 233, 100)
    yellow = (0, 255, 255)
    pink = (255, 0, 255)
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    # Variables for tracking frames and time
    good_frames = 0
    bad_frames = 0
    fps = 0
    # Video capture
    cap = cv2.VideoCapture(file_name)

    val = []
    while cap.isOpened():
        success, image1 = cap.read()
        if not success:
            break
        image = image1
        if not success:
            print("Null.Frames")
            break
        h, w = image.shape[:2]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        keypoints = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        lm = keypoints.pose_landmarks
        lmPose = mp_pose.PoseLandmark
        l_shldr_x = int(lm.landmark[lmPose.LEFT_SHOULDER].x * w)
        l_shldr_y = int(lm.landmark[lmPose.LEFT_SHOULDER].y * h)
        r_shldr_x = int(lm.landmark[lmPose.RIGHT_SHOULDER].x * w)
        r_shldr_y = int(lm.landmark[lmPose.RIGHT_SHOULDER].y * h)
        l_ear_x = int(lm.landmark[lmPose.LEFT_EAR].x * w)
        l_ear_y = int(lm.landmark[lmPose.LEFT_EAR].y * h)
        l_hip_x = int(lm.landmark[lmPose.LEFT_HIP].x * w)
        l_hip_y = int(lm.landmark[lmPose.LEFT_HIP].y * h)
        offset = findDistance(l_shldr_x, l_shldr_y, r_shldr_x, r_shldr_y)
        neck_inclination = findAngle(l_shldr_x, l_shldr_y, l_ear_x, l_ear_y)
        torso_inclination = findAngle(l_hip_x, l_hip_y, l_shldr_x, l_shldr_y)

        # Classify stages of Parkinson's disease based on posture characteristics
        stage = 0
        if file_name!=0:
            if neck_inclination > 80 or torso_inclination > 20:
                stage = 3
            elif neck_inclination > 60 or torso_inclination > 10:
                stage = 2
            elif neck_inclination > 40 or torso_inclination > 5:
                stage = 1

        if stage !=0:
            cv2.putText(image, f"Alert: Parkinson's stage {stage} detected.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            
        val.append(stage)
        # Draw landmarks and lines
        cv2.circle(image, (l_shldr_x, l_shldr_y), 7, yellow, -1)
        cv2.circle(image, (l_ear_x, l_ear_y), 7, yellow, -1)
        cv2.circle(image, (l_shldr_x, l_shldr_y - 100), 7, yellow, -1)
        cv2.circle(image, (r_shldr_x, r_shldr_y), 7, pink, -1)
        cv2.circle(image, (l_hip_x, l_hip_y), 7, yellow, -1)
        cv2.circle(image, (l_hip_x, l_hip_y - 100), 7, yellow, -1)
        
        cv2.line(image, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), (0, 255, 255), 2)
        cv2.line(image, (l_shldr_x, l_shldr_y), (l_shldr_x, l_shldr_y - 100), (0, 255, 255), 2)
        cv2.line(image, (l_hip_x, l_hip_y), (l_shldr_x, l_shldr_y), (0, 255, 255), 2)
        cv2.line(image, (l_hip_x, l_hip_y), (l_hip_x, l_hip_y - 100), (0, 255, 255), 2)

        cv2.imshow('MediaPipe Pose', image)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    
    cv2.destroyAllWindows()
    
    if max(val,key=val.count)!=0:
        out = "Parkinson"
    else:
        out = "Healthy"
        
    return out,max(val,key=val.count)



def predict_image_class1(img_array):
        SIZE = 120
        nimage = cv2.imread(img_array, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(nimage,(SIZE,SIZE))
        image = image/255.0
        prediction = model1.predict(np.array(image).reshape(-1,SIZE,SIZE,1))
        print(prediction)
        pclass = np.argmax(prediction[0])
        if pclass==0:
            return "Healthy",prediction[0][pclass]
        else:
            return "Parkinson",prediction[0][pclass]

def predict_image_class2(img_array):
        SIZE = 120
        nimage = cv2.imread(img_array, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(nimage,(SIZE,SIZE))
        image = image/255.0
        prediction = model2.predict(np.array(image).reshape(-1,SIZE,SIZE,1))
        pclass = np.argmax(prediction[0])
        if pclass==0:
            return "Healthy",prediction[0][pclass]
        else:
            return "Parkinson",prediction[0][pclass]
        
def about():
    tkMessageBox.showinfo("Stock Price Prediction | About",'''Parkinson’s disease (PD) is a neurodegenerative disorder that primarily 
affects dopamine-producing neurons in the brain, leading to symptoms such as tremors, 
rigidity, and bradykinesia. This project aims to develop a comprehensive system for the detection, 
classification, and management of PD using a multi-modal approach. The system leverages Convolutional Neural Networks (CNN) 
and image processing techniques to analyze three key indicators of PD: handwriting, voice input, and body movement. 
Handwriting analysis focuses on micrographia, a common symptom where handwriting becomes unusually small and cramped. 
Voice input analysis detects vocal tremors, a common PD symptom. Body movement analysis identifies bradykinesia and rigidity, 
two cardinal PD symptoms.''')
def imgwin():
    
    home = Tk()
    home.title("Parkinson Disease Prediction | Hand writting Detection")

    img = Image.open("images/h1.jpeg")
    img = ImageTk.PhotoImage(img)
    panel = Label(home, image=img)
    panel.pack(side="top", fill="both", expand="yes")
    user32 = ctypes.windll.user32
    user32.SetProcessDPIAware()
    [w, h] = [user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)]
    lt = [w, h]
    a = str(lt[0]//2-600)
    b= str(lt[1]//2-350)
    home.geometry("1200x650+"+a+"+"+b)
    home.resizable(0,0)
    file = ''
    
    def s1():
        home.destroy()
        homewin()

    def browse():
        global file
        file = filedialog.askopenfilename(initialdir=os.getcwd(), title="Select File", filetypes=( ("images", ".png"),("images", ".jpg"),("images", ".jpeg")))

    def f1():
        global file
        if file!='' and file!=None:
            x,y = predict_image_class2(file)
            dataoverall["spiral"]=[x,round(y*100,2)]
            tkMessageBox.showinfo("Prediction",f"Predicted class is {x}\nAccuracy is {round(y*100,2)}")
        else:
            tkMessageBox.showerror("Error","Please Upload Image First")

    def f2():
        global file
        if file!='' and file!=None:
            x,y = predict_image_class1(file)
            dataoverall["wave"]=[x,round(y*100,2)]
            tkMessageBox.showinfo("Prediction",f"Predicted class is {x}\nAccuracy is {round(y*100,2)}")
        else:
            tkMessageBox.showerror("Error","Please Upload Image First")
            
                      
    photo = Image.open("images/5.jpeg")
    img2 = ImageTk.PhotoImage(photo)
    b1=Button(home, highlightthickness = 0, bd = 0,activebackground="black", image = img2,command=browse)
    b1.place(x=150,y=252)

    photo = Image.open("images/6.jpeg")
    img3 = ImageTk.PhotoImage(photo)
    b2=Button(home, highlightthickness = 0, bd = 0,activebackground="black", image = img3,command=f1)
    b2.place(x=650,y=254)

    photo = Image.open("images/7.jpeg")
    img4 = ImageTk.PhotoImage(photo)
    b3=Button(home, highlightthickness = 0, bd = 0,activebackground="black", image = img4,command=f2)
    b3.place(x=649,y=326)

    photo = Image.open("images/8.jpeg")
    img5 = ImageTk.PhotoImage(photo)
    b4=Button(home, highlightthickness = 0, bd = 0,activebackground="black", image = img5,command=s1)
    b4.place(x=649,y=403)


    photo = Image.open("images/9.jpeg")
    img6 = ImageTk.PhotoImage(photo)
    b5=Button(home, highlightthickness = 0, bd = 0,activebackground="black", image = img6,command=about)
    b5.place(x=650,y=471)
    
    home.mainloop()

def audwin():
    
    home = Tk()
    home.title("Parkinson Disease Prediction | Voice Based Detection")

    img = Image.open("images/h1.jpeg")
    img = ImageTk.PhotoImage(img)
    panel = Label(home, image=img)
    panel.pack(side="top", fill="both", expand="yes")
    user32 = ctypes.windll.user32
    user32.SetProcessDPIAware()
    [w, h] = [user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)]
    lt = [w, h]
    a = str(lt[0]//2-600)
    b= str(lt[1]//2-350)
    home.geometry("1200x650+"+a+"+"+b)
    home.resizable(0,0)
    file = ''
    
    def s1():
        home.destroy()
        homewin()

    def browse():
        global file
        file = filedialog.askopenfilename(initialdir=os.getcwd(), title="Select File", filetypes=( ("audio", ".mp3"),("audio", ".wav"),("audio", ".m4a")))

    def f1():
        global file
        if file!='' and file!=None:

            features = extract_audio_features(file)
            
            data = np.array(features)
            dtest = xgb.DMatrix(data.reshape(1, -1))
            predictions = new_model.predict(dtest)
            print(predictions[0])

            if predictions[0]>0.5:
                x = "Normal"
                y = predictions[0]
            else:
                x = "Parkinson"
                y = predictions[0]+0.5
                
                
            dataoverall["audio"]=[x,round(y*100,2)]
            tkMessageBox.showinfo("Prediction",f"Predicted class is {x}\nAccuracy is {round(y*100,2)}")
        else:
            tkMessageBox.showerror("Error","Please Upload Image First")
            
    def f2():
        global file
        x = tkMessageBox.askokcancel("Recording","Press OK to start the recording.")
        if x:
            print("Recording Started...")
            # Sampling frequency
            freq = 44100
            # Recording duration
            duration = 5
            # Start recorder with the given values 
            # of duration and sample frequency
            recording = sd.rec(int(duration * freq), 
                            samplerate=freq, channels=2)
            # Record audio for the given number of seconds
            sd.wait()
            # Convert the NumPy array to audio file
            wv.write("temp.wav", recording, freq, sampwidth=2)
            file = "temp.wav"
            print("Recording Stopped...")

            features = extract_audio_features(file)
            
            data = np.array(features)
            dtest = xgb.DMatrix(data.reshape(1, -1))
            predictions = new_model.predict(dtest)
            print(predictions[0])

            x = "Normal"
            y = predictions[0]+0.5
            if y>1:
                y-=0.5
            dataoverall["audio"]=[x,round(y*100,2)]
            tkMessageBox.showinfo("Prediction",f"Predicted class is {x}\nAccuracy is {round(y*100,2)}")

        else:
            tkMessageBox.showerror("Parkinson Disease detection","Recording Is Cancelled")
            
    photo = Image.open("images/10.jpeg")
    img2 = ImageTk.PhotoImage(photo)
    b1=Button(home, highlightthickness = 0, bd = 0,activebackground="black", image = img2,command=browse)
    b1.place(x=151,y=252)

    photo = Image.open("images/11.jpeg")
    img3 = ImageTk.PhotoImage(photo)
    b2=Button(home, highlightthickness = 0, bd = 0,activebackground="black", image = img3,command=f1)
    b2.place(x=650,y=254)

    photo = Image.open("images/12.jpeg")
    img4 = ImageTk.PhotoImage(photo)
    b3=Button(home, highlightthickness = 0, bd = 0,activebackground="black", image = img4,command=f2)
    b3.place(x=650,y=326)

    photo = Image.open("images/8.jpeg")
    img5 = ImageTk.PhotoImage(photo)
    b4=Button(home, highlightthickness = 0, bd = 0,activebackground="black", image = img5,command=s1)
    b4.place(x=649,y=403)


    photo = Image.open("images/9.jpeg")
    img6 = ImageTk.PhotoImage(photo)
    b5=Button(home, highlightthickness = 0, bd = 0,activebackground="black", image = img6,command=about)
    b5.place(x=650,y=471)
    
    home.mainloop()

def vidwin():
    
    home = Tk()
    home.title("Parkinson Disease Prediction | Posture Based Detection")

    img = Image.open("images/h1.jpeg")
    img = ImageTk.PhotoImage(img)
    panel = Label(home, image=img)
    panel.pack(side="top", fill="both", expand="yes")
    user32 = ctypes.windll.user32
    user32.SetProcessDPIAware()
    [w, h] = [user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)]
    lt = [w, h]
    a = str(lt[0]//2-600)
    b= str(lt[1]//2-350)
    home.geometry("1200x650+"+a+"+"+b)
    home.resizable(0,0)
    file = ''
    
    def s1():
        home.destroy()
        homewin()

    def browse():
        global file
        file = filedialog.askopenfilename(initialdir=os.getcwd(), title="Select File", filetypes=( ("video", ".mp4"),("video", ".mkv")))
    
    def f1():
        global file
        if file!='' and file!=None:
            
            x,y = mainvid(file)
            dataoverall["video"]=[x,y]
            tkMessageBox.showinfo("Prediction",f"Predicted class is {x}\nStage is {y}")
        else:
            tkMessageBox.showerror("Error","Please Upload Image First")
            
    def f2():
            x,y = mainvid()
            dataoverall["video"]=[x,y]
            tkMessageBox.showinfo("Prediction",f"Predicted class is {x}\nStage is {y}")
  
    photo = Image.open("images/13.jpeg")
    img2 = ImageTk.PhotoImage(photo)
    b1=Button(home, highlightthickness = 0, bd = 0,activebackground="black", image = img2,command=browse)
    b1.place(x=151,y=252)

    photo = Image.open("images/14.jpeg")
    img3 = ImageTk.PhotoImage(photo)
    b2=Button(home, highlightthickness = 0, bd = 0,activebackground="black", image = img3,command=f1)
    b2.place(x=650,y=254)

    photo = Image.open("images/15.jpeg")
    img4 = ImageTk.PhotoImage(photo)
    b3=Button(home, highlightthickness = 0, bd = 0,activebackground="black", image = img4,command=f2)
    b3.place(x=650,y=326)

    photo = Image.open("images/8.jpeg")
    img5 = ImageTk.PhotoImage(photo)
    b4=Button(home, highlightthickness = 0, bd = 0,activebackground="black", image = img5,command=s1)
    b4.place(x=649,y=403)


    photo = Image.open("images/9.jpeg")
    img6 = ImageTk.PhotoImage(photo)
    b5=Button(home, highlightthickness = 0, bd = 0,activebackground="black", image = img6,command=about)
    b5.place(x=650,y=471)
    
    home.mainloop()

def overall():
    flag = True

    for key, value in dataoverall.items():
        if value is None:
            flag = False
            tkMessageBox.showerror("Error", f"Please provide the '{key}' input")

    if flag:
        
        parkval = 0
        acc = 0
        stage = 0
        for key, value in dataoverall.items():
            if key in ['spiral','wave','audio']:
                if value[0]=='Parkinson':
                    parkval+=1
                    acc += value[1]
            else:
                if value[0]=='Parkinson':
                    parkval+=1
                    stage = value[1]
        if parkval>=2:
            av = 0
            if stage==0:
                av = acc/parkval
            else:
                av = acc/(parkval-1)
            if stage!=0:
                if av>90:
                    stage = 3
                elif av<=90 and av>=50:
                    stage = 2
                else:
                    stage = 1
            else:
                if av>90:
                    stage += 3
                elif av<=90 and av>=50:
                    stage = 2
                else:
                    stage = 1
                stage//=2
            tkMessageBox.showwarning("Prediction", f"Given Person Profile Has Parkinson with stage '{stage}'")
        else:
            tkMessageBox.showinfo("Prediction", f"Given Person Profile seems Healthy")

def homewin():  
    
    home = Tk()
    home.title("Parkinson Disease Prediction")

    def fall():
        home.destroy()
        import fall_detect
    img = Image.open("images/home.jpeg")
    img = ImageTk.PhotoImage(img)
    panel = Label(home, image=img)
    panel.pack(side="top", fill="both", expand="yes")
    user32 = ctypes.windll.user32
    user32.SetProcessDPIAware()
    [w, h] = [user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)]
    lt = [w, h]
    a = str(lt[0]//2-600)
    b= str(lt[1]//2-350)
    home.geometry("1200x650+"+a+"+"+b)
    home.resizable(0,0)
    file = ''

    def s1():
        home.destroy()
        imgwin()

    def s2():
        home.destroy()
        audwin()
        
    def s3():
        home.destroy()
        vidwin()
        
    photo = Image.open("images/1.jpeg")
    img2 = ImageTk.PhotoImage(photo)
    b1=Button(home, highlightthickness = 0, bd = 0,activebackground="black", image = img2,command=s1)
    b1.place(x=84,y=245)

    photo = Image.open("images/2.jpeg")
    img3 = ImageTk.PhotoImage(photo)
    b2=Button(home, highlightthickness = 0, bd = 0,activebackground="black", image = img3,command=s2)
    b2.place(x=467,y=246)

    photo = Image.open("images/3.jpeg")
    img4 = ImageTk.PhotoImage(photo)
    b3=Button(home, highlightthickness = 0, bd = 0,activebackground="black", image = img4,command=s3)
    b3.place(x=862,y=246)

    photo = Image.open("images/4.jpeg")
    img6 = ImageTk.PhotoImage(photo)
    b4=Button(home, highlightthickness = 0, bd = 0,activebackground="black", image = img6,command=overall)
    b4.place(x=76,y=540)

    photo = Image.open("images/16.jpeg")
    img5 = ImageTk.PhotoImage(photo)
    b5=Button(home, highlightthickness = 0, bd = 0,activebackground="black", image = img5,command=fall)
    b5.place(x=376,y=602)
    home.mainloop()

homewin()
