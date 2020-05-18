import tkinter as tk
from tkinter import Message ,Text
import cv2,os
import shutil
import csv
import numpy as np
from PIL import Image, ImageTk
import pandas as pd
import datetime
import time
import tkinter.ttk as ttk
import tkinter.font as font
# import webbrowser
from firebase import  firebase

window = tk.Tk()
fbcon = firebase.FirebaseApplication("https://final-year-82636.firebaseio.com/", None)
window.title("Face_Recogniser")

dialog_title = 'QUIT'
dialog_text = 'Are you sure?'


window.configure(background='#c2c2f0')




window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)


message = tk.Label(window, text="Face-Recognition" ,fg="#1a0d00",bg="#c2c2f0"  ,height=2,font=('times', 30, 'italic bold underline')) 

message.place(x=500, y=20)

lbl = tk.Label(window, text="Enter ID",width=20  ,height=2  ,fg="red"  ,bg="#e6ffff" ,font=('times', 15, ' bold ') ) 
lbl.place(x=300, y=200)

txt = tk.Entry(window,width=20  ,bg="#e6ffff" ,fg="red",font=('times', 15, ' bold '))
txt.place(x=600, y=215)

lbl2 = tk.Label(window, text="Enter Name",width=20  ,fg="red"  ,bg="#e6ffff"    ,height=2 ,font=('times', 15, ' bold ')) 
lbl2.place(x=300, y=300)

txt2 = tk.Entry(window,width=20  ,bg="#e6ffff"  ,fg="red",font=('times', 15, ' bold ')  )
txt2.place(x=600, y=315)

lbl3 = tk.Label(window, text="Notification : ",width=20  ,fg="red"  ,bg="#e6ffff"  ,height=2 ,font=('times', 15, ' bold  ')) 
lbl3.place(x=300, y=400)

message = tk.Label(window, text="" ,bg="#e6ffff"  ,fg="red"  ,width=30  ,height=2, activebackground = "#e6ffff" ,font=('times', 15, ' bold ')) 
message.place(x=600, y=400)

lbl3 = tk.Label(window, text="Entry time : ",width=20  ,fg="red"  ,bg="#e6ffff"  ,height=2 ,font=('times', 15, ' bold  ')) 
lbl3.place(x=300, y=600)


message2 = tk.Label(window, text="" ,fg="red"   ,bg="#e6ffff",activeforeground = "green",width=30  ,height=2  ,font=('times', 15, ' bold ')) 
message2.place(x=600, y=600)


 
def clear():
    txt.delete(0, 'end')    
    res = ""
    message.configure(text= res)

def clear2():
    txt2.delete(0, 'end')    
    res = ""
    message.configure(text= res)    
    
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
 
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
 
    return False
 
def TakeImages():        
    Id=(txt.get())
    name=(txt2.get())
    if(is_number(Id) and name.isalpha()):
        cam = cv2.VideoCapture(0)
        harcascadePath = "haarcascade_frontalface_default.xml"
        detector=cv2.CascadeClassifier(harcascadePath)
        sampleNum=0
        while(True):
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)        
                 
                sampleNum=sampleNum+1
               
                cv2.imwrite("TrainingImage\ "+name +"."+Id +'.'+ str(sampleNum) + ".jpg", gray[y:y+h,x:x+w])
                
                cv2.imshow('frame',img)
            
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            
            elif sampleNum>60:
                break
        cam.release()
        cv2.destroyAllWindows() 
        res = "Images Saved for ID : " + Id +" Name : "+ name
        row = [Id , name]
        with open('StudentDetails\StudentDetails.csv','a+') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()
        message.configure(text= res)
        TrainImages()
    else:
        if(is_number(Id)):
            res = "Enter Alphabetical Name"
            message.configure(text= res)
        if(name.isalpha()):
            res = "Enter Numeric Id"
            message.configure(text= res)
    
def TrainImages():
    recognizer = cv2.face_LBPHFaceRecognizer.create()
    harcascadePath = "haarcascade_frontalface_default.xml"
    detector =cv2.CascadeClassifier(harcascadePath)
    faces,Id = getImagesAndLabels("TrainingImage")
    recognizer.train(faces, np.array(Id))
    recognizer.save("TrainingImageLabel\Trainner.yml")
    res = "Image Trained"
    message3.configure(text= res)

def getImagesAndLabels(path):
    
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)] 
    
    faces=[]
    
    Ids=[]
    
    for imagePath in imagePaths:
       
        pilImage=Image.open(imagePath).convert('L')
        
        imageNp=np.array(pilImage,'uint8')
        
        Id=int(os.path.split(imagePath)[-1].split(".")[1])
        
        faces.append(imageNp)
        Ids.append(Id)        
    return faces,Ids
entry=[]
def TrackImages_in():
    
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("TrainingImageLabel\Trainner.yml")
    harcascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath);    
    df=pd.read_csv("StudentDetails\StudentDetails.csv")
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX        
    col_names =  ['Id','Name','Date','Time','status']
    entry = pd.DataFrame(columns = col_names)    
    s=0
    u=0
    while True:
        stat=0
        status="in"
        ret, im =cam.read()
        gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        faces=faceCascade.detectMultiScale(gray, 1.2,5)    
        s+=1
        for(x,y,w,h) in faces:
            cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
            Id, conf = recognizer.predict(gray[y:y+h,x:x+w])                                   
            if(conf < 50):
                ts = time.time()      
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                aa=df.loc[df['Id'] == Id]['Name'].values
                tt=str(Id)+"-"+aa
                # entry.loc[len(entry)] = [Id,aa,date,timeStamp]
                entry=[Id,aa,date,timeStamp,status]
                stat+=1
                # url="demo.html"
                # webbrowser.open(url)
                
                
            else:
                stat-=1
                u+=1
                Id='Unknown'                
                tt=str(Id)  
                if u>1000:
                    break
                message2.configure(text="unknown")
                
            if(conf > 75):
                noOfFile=len(os.listdir("ImagesUnknown"))+1
                cv2.imwrite("ImagesUnknown\Image"+str(noOfFile) + ".jpg", im[y:y+h,x:x+w]) 
                message2.configure(text="unknown")     
                     
            cv2.putText(im,str(tt),(x,y+h), font, 1,(255,255,255),2)        
        #entry=entry.drop_duplicates(subset=['Id'],keep='first')    
        cv2.imshow('im',im) 
        if (cv2.waitKey(1)==ord('q')):
            break
        elif s>10:
            break
    if conf<50:
       
        res = fbcon.put('/','LED_Status',1)
    else:
        
         res = fbcon.put('/','LED_Status',0)
    ts = time.time()      
    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
    Hour,Minute,Second=timeStamp.split(":")
    # fileName="Entry\entry_"+date+"_"+Hour+".csv"
    fileName="C:\\xampp\\htdocs\\dumy\\entry.csv"
    #entry.to_csv(fileName,index=False)

    with open(fileName,'a+') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(entry)

    cam.release()
    cv2.destroyAllWindows()
    # url="demo.html"
    # webbrowser.open(url)
    print(entry)
    res=entry
    message2.configure(text= res)

# def TrackImages_out():
    
#     recognizer = cv2.face.LBPHFaceRecognizer_create()
#     recognizer.read("TrainingImageLabel\Trainner.yml")
#     harcascadePath = "haarcascade_frontalface_default.xml"
#     faceCascade = cv2.CascadeClassifier(harcascadePath);    
#     df=pd.read_csv("StudentDetails\StudentDetails.csv")
#     cam = cv2.VideoCapture(0)
#     font = cv2.FONT_HERSHEY_SIMPLEX        
#     col_names =  ['Id','Name','Date','Time','status']
#     exit1 = pd.DataFrame(columns = col_names)    
#     s=0
#     u=0
#     while True:
#         stat=0
#         status="out"
#         ret, im =cam.read()
#         gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
#         faces=faceCascade.detectMultiScale(gray, 1.2,5)    
#         s+=1
#         for(x,y,w,h) in faces:
#             cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
#             Id, conf = recognizer.predict(gray[y:y+h,x:x+w])                                   
#             if(conf < 50):
#                 ts = time.time()      
#                 date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
#                 timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
#                 aa=df.loc[df['Id'] == Id]['Name'].values
#                 tt=str(Id)+"-"+aa
#                 # entry.loc[len(entry)] = [Id,aa,date,timeStamp]
#                 exit1=[Id,aa,date,timeStamp,status]
#                 # url="demo.html"
#                 # webbrowser.open(url)
#                 stat+=1
                
#             else:
#                 stat-=1
#                 u+=1
#                 Id='Unknown'                
#                 tt=str(Id)  
#                 if u>1000:
#                     break
#                 message2.configure(text="unknown")
                
#             if(conf > 75):
#                 noOfFile=len(os.listdir("ImagesUnknown"))+1
#                 cv2.imwrite("ImagesUnknown\Image"+str(noOfFile) + ".jpg", im[y:y+h,x:x+w])    
#                 message2.configure(text="unknown")  
                     
#             cv2.putText(im,str(tt),(x,y+h), font, 1,(255,255,255),2)        
#         #entry=entry.drop_duplicates(subset=['Id'],keep='first')    
#         cv2.imshow('im',im) 
#         if (cv2.waitKey(1)==ord('q')):
#             break
#         elif s>10:
#             break
    
#     if stat>0:
       
#         res = fbcon.put('final-year-82636','LED_Status',1)
#     else:
        
#          res = fbcon.put('final-year-82636','LED_Status',0)
#     ts = time.time()      
#     date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
#     timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
#     Hour,Minute,Second=timeStamp.split(":")
#     # fileName="Entry\entry_"+date+"_"+Hour+".csv"
#     fileName="C:\\xampp\\htdocs\\dumy\\exit.csv"
#     #entry.to_csv(fileName,index=False)

#     with open(fileName,'a+') as csvFile:
#         writer = csv.writer(csvFile)
#         writer.writerow(exit1)

#     cam.release()
#     cv2.destroyAllWindows()
#     # url="demo.html"
#     # webbrowser.open(url)
#     print(exit1)
#     res=exit1
#     message2.configure(text= res)
    

  
  
takeImg = tk.Button(window, text="Take Images", command=TakeImages  ,fg="#b3ffd9"  ,bg="#995c00"  ,width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
takeImg.place(x=50, y=500)
message3 = tk.Label(window, text="" ,bg="#e6ffff"  ,fg="red"  ,width=23  ,height=2, activebackground = "#e6ffff" ,font=('times', 15, ' bold ')) 
message3.place(x=400, y=500)
trackImg = tk.Button(window, text="TRACK", command=TrackImages_in  ,fg="#b3ffd9"  ,bg="#995c00"  ,width=10  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
trackImg.place(x=800, y=500)
# trackImg = tk.Button(window, text="EXIT", command=TrackImages_out  ,fg="#b3ffd9"  ,bg="#995c00"  ,width=10  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
# trackImg.place(x=900, y=500)
quitWindow = tk.Button(window, text="Quit", command=window.destroy  ,fg="#b3ffd9"  ,bg="#995c00"  ,width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
quitWindow.place(x=1050, y=500)

 
window.mainloop()