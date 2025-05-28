from tkinter import *
import ctypes,os
from PIL import ImageTk, Image
import tkinter.messagebox as tkMessageBox
import cv2
import mediapipe as mp
import numpy as np
import time
import joblib
import datetime
import smtplib, ssl

home = Tk()
img = Image.open("image/home.png")
img = ImageTk.PhotoImage(img)
panel = Label(home, image=img)
panel.pack(side="top", fill="both", expand="yes")
user32 = ctypes.windll.user32
user32.SetProcessDPIAware()
[w, h] = [user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)]
lt = [w, h]
a = str(lt[0]//2-450)
b= str(lt[1]//2-283)
home.title("Fall Detection System")
home.geometry("900x566+"+a+"+"+b)
home.resizable(0,0)

def start(path):

    pose_knn = joblib.load('Model/PoseKeypoint.joblib')
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose
    prevTime = 0
    keyXYZ = [
        "nose_x",
        "nose_y",
        "nose_z",
        "left_eye_inner_x",
        "left_eye_inner_y",
        "left_eye_inner_z",
        "left_eye_x",
        "left_eye_y",
        "left_eye_z",
        "left_eye_outer_x",
        "left_eye_outer_y",
        "left_eye_outer_z",
        "right_eye_inner_x",
        "right_eye_inner_y",
        "right_eye_inner_z",
        "right_eye_x",
        "right_eye_y",
        "right_eye_z",
        "right_eye_outer_x",
        "right_eye_outer_y",
        "right_eye_outer_z",
        "left_ear_x",
        "left_ear_y",
        "left_ear_z",
        "right_ear_x",
        "right_ear_y",
        "right_ear_z",
        "mouth_left_x",
        "mouth_left_y",
        "mouth_left_z",
        "mouth_right_x",
        "mouth_right_y",
        "mouth_right_z",
        "left_shoulder_x",
        "left_shoulder_y",
        "left_shoulder_z",
        "right_shoulder_x",
        "right_shoulder_y",
        "right_shoulder_z",
        "left_elbow_x",
        "left_elbow_y",
        "left_elbow_z",
        "right_elbow_x",
        "right_elbow_y",
        "right_elbow_z",
        "left_wrist_x",
        "left_wrist_y",
        "left_wrist_z",
        "right_wrist_x",
        "right_wrist_y",
        "right_wrist_z",
        "left_pinky_x",
        "left_pinky_y",
        "left_pinky_z",
        "right_pinky_x",
        "right_pinky_y",
        "right_pinky_z",
        "left_index_x",
        "left_index_y",
        "left_index_z",
        "right_index_x",
        "right_index_y",
        "right_index_z",
        "left_thumb_x",
        "left_thumb_y",
        "left_thumb_z",
        "right_thumb_x",
        "right_thumb_y",
        "right_thumb_z",
        "left_hip_x",
        "left_hip_y",
        "left_hip_z",
        "right_hip_x",
        "right_hip_y",
        "right_hip_z",
        "left_knee_x",
        "left_knee_y",
        "left_knee_z",
        "right_knee_x",
        "right_knee_y",
        "right_knee_z",
        "left_ankle_x",
        "left_ankle_y",
        "left_ankle_z",
        "right_ankle_x",
        "right_ankle_y",
        "right_ankle_z",
        "left_heel_x",
        "left_heel_y",
        "left_heel_z",
        "right_heel_x",
        "right_heel_y",
        "right_heel_z",
        "left_foot_index_x",
        "left_foot_index_y",
        "left_foot_index_z",
        "right_foot_index_x",
        "right_foot_index_y",
        "right_foot_index_z"
    ]
    
    res_point = []

    cap = cv2.VideoCapture(path)
    with mp_pose.Pose(
            static_image_mode=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                
                continue
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            if results.pose_landmarks:
                for index, landmarks in enumerate(results.pose_landmarks.landmark):
                    # print(index, landmarks.x, landmarks.y, landmarks.z)
                    res_point.append(landmarks.x)
                    res_point.append(landmarks.y)
                    res_point.append(landmarks.z)
                shape1 = int(len(res_point) / len(keyXYZ))
                res_point = np.array(res_point).reshape(shape1, len(keyXYZ))
                pred = pose_knn.predict(res_point)
                res_point = []
                
                if pred == 0:
                    cv2.putText(image, "Fall", (200, 320), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 2)

                    port = 465  # For SSL
                    password = "ravj babn bdnp yksi"
                    txt = "Person Fall detected at "+str(datetime.datetime.now())
                    smtp_server = "smtp.gmail.com"
                    sender_email = "ishanagrawal1101@gmail.com"  # Enter your address
                    receiver_email = open("alert.txt","r").read()
                    message = """\
                    Subject: Alert !!! Person Fall 

                    """+txt

                    context = ssl.create_default_context()
                    with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
                        server.login(sender_email, password)
                        server.sendmail(sender_email, receiver_email, message)
                    break
                else:
                    cv2.putText(image, "Normal", (200, 320), cv2.FONT_HERSHEY_PLAIN, 5, (0, 255, 0), 2)
            # Draw the pose annotation on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

            # Flip the image horizontally for a selfie-view display.
            currTime = time.time()
            fps = 1 / (currTime - prevTime)
            prevTime = currTime
            cv2.putText(image, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 196, 255), 2)
            # cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
            cv2.imshow('MediaPipe Pose', image)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    cap.release()
    cv2.destroyAllWindows()


def Exit():
    result = tkMessageBox.askquestion(
        '', 'Are you sure you want to exit?', icon="warning")
    if result == 'yes':
        home.destroy()
        import main
    else:
        tkMessageBox.showinfo(
            'Return', 'You will now return to the main screen')
                    

def add(): 
    root= Tk()
    root.title("Fall Detection System")
    root.geometry("500x300")
    root.resizable(0,0)
    root.config(bg="#1b2b39")
    eml= StringVar()
    
    def ad():
        if '@' in a.get():
            f = open("alert.txt","w")
            f.write(a.get())
            f.close()
            root.destroy()
        else:
            tkMessageBox.showinfo(
            'warning', 'Please Enter A valid Email')
    Label(root,text="Add Email",font=("",30,"bold"),bg = "#1b2b39",fg="#fbfdff").place(x=150,y=30)
    a = Entry(root,textvariable=eml,font=("",18,"bold"),fg = "#1b2b39",bg="#fbfdff",width=32)
    a.place(x=40,y=120)

    f = open("alert.txt","r")
    f = f.read()
    a.insert('a',f)
    
    Button(root,text="ADD",font=("",14,"bold"),fg = "#1b2b39",bg="#fbfdff",width=34,command=ad).place(x=41,y=180)
    root.mainloop()

def st(): 
    root= Tk()
    root.title("Fall Detection System")
    root.geometry("500x300")
    root.resizable(0,0)
    root.config(bg="#1b2b39")
    eml= StringVar()
    
    def ad():
        dirc = a.get()
        if '0'==dirc:
            dirc=0
 
        start(dirc)
    Label(root,text="Video Directory (0 for realtime) :",font=("",15,"bold"),bg = "#1b2b39",fg="#fbfdff").place(x=35,y=30)
    a = Entry(root,textvariable=eml,font=("",18,"bold"),fg = "#1b2b39",bg="#fbfdff",width=32)
    a.place(x=40,y=95)
    a.insert('a','0')
    Button(root,text="Start",font=("",14,"bold"),fg = "#1b2b39",bg="#fbfdff",width=34,command=ad).place(x=41,y=180)
    root.mainloop()

def contact():
    contactwin = Toplevel()
    img = Image.open("image/about.png")
    img = ImageTk.PhotoImage(img)
    panel = Label(contactwin, image=img)
    panel.pack(side="top", fill="both", expand="yes")
    user32 = ctypes.windll.user32
    user32.SetProcessDPIAware()
    [w, h] = [user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)]
    lt = [w, h]
    a = str(lt[0]//2-450)
    b= str(lt[1]//2-283)
    contactwin.title("Fall Detection System")
    contactwin.geometry("900x566+"+a+"+"+b)
    contactwin.resizable(0,0)
    contactwin.mainloop()
    
photo = Image.open("image/1.png")
img1 = ImageTk.PhotoImage(photo)
b1=Button(home, highlightthickness = 0,activebackground="#0b2a4a", bd = 0, image = img1,command=st)
b1.place(x=0,y=149)

photo = Image.open("image/2.png")
img2 = ImageTk.PhotoImage(photo)
b2=Button(home, highlightthickness = 0,activebackground="#0b2a4a", bd = 0, image = img2,command=add)
b2.place(x=0,y=228)

photo = Image.open("image/3.png")
img3 = ImageTk.PhotoImage(photo)
b3=Button(home, highlightthickness = 0,activebackground="#0b2a4a", bd = 0, image = img3,command=contact)
b3.place(x=0,y=307)

photo = Image.open("image/4.png")
img4 = ImageTk.PhotoImage(photo)
b4=Button(home, highlightthickness = 0,activebackground="#0b2a4a", bd = 0, image = img4,command=Exit)
b4.place(x=0,y=386)

home.mainloop()
