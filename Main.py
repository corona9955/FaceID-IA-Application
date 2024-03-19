import tkinter as tk
from tkinter import ttk
import os
import cv2
import imutils
import numpy as np

class FormMenu:
    def __init__(self):
        self.CreateForm()

    def CreateForm(self):
        root = tk.Tk()
        root.title("FaceID IA Application")
        root.geometry("500x200")
        root.resizable(0,0)
        
        
        btnNewPerson = tk.Button(root, text="Create new person", 
                                 pady="5px", 
                                 padx="5px", 
                                 border=4,
                                 command= lambda: self.NewPerson())
        btnNewPerson.place(relx=0.5, rely=0.25, anchor="center")  
        btnIdentifyPerson = tk.Button(root, text="Identify person", 
                                      pady="5px", 
                                      padx="5px", 
                                      border=4,
                                      command= lambda: self.IndentifyPerson())
        btnIdentifyPerson.place(relx=0.5, rely=0.50, anchor="center")  

        root.mainloop()

    def NewPerson(self):
        root = tk.Tk()
        root.title("New person")
        root.geometry("400x200")
        root.resizable(0,0)
        lblName = tk.Label(root, text="Spell your name: ")
        lblName.place(relx=0.4, rely=0.25)
        txtName = tk.Text(root, width=20, height=1)
        txtName.place(relx=0.3, rely=0.40)

        btnAccept = tk.Button(root,   text="Accept",
                                      pady="5px", 
                                      padx="5px", 
                                      border=4,
                                      command= lambda: self.CameraLive("Data/" + txtName.get("1.0","end-1c")))
        btnAccept.place(relx=0.4, rely=0.60)  
        root.mainloop()

    def CameraLive(self, nameNewFolder):
        root = tk.Tk()
        root.title("Camera selector")
        root.geometry("500x200")
        root.resizable(0,0)      
        cameras = self.ListCameras()
        cboxSelectorCamera = ttk.Combobox(root, values=cameras, state="readonly")
        cboxSelectorCamera.pack(pady=20, padx=20)

        btnStart = tk.Button(root, text="Start", 
                                 pady="5px", 
                                 padx="5px", 
                                 border=4,
                                 state="disabled",
                                 command= lambda: self.OpenCamera(cboxSelectorCamera, nameNewFolder))
        btnStart.place(relx=0.5, rely=0.40, anchor="center")  

        if not os.path.exists(nameNewFolder):
            print('Carpeta creada: ',nameNewFolder)
            os.makedirs(nameNewFolder)

        def handle_combobox_selection(event):
            index_seleccionado = self.SelectedList(cboxSelectorCamera)
            if index_seleccionado is not None:
                btnStart.configure(state="normal")
                print(f"Índice seleccionado: {index_seleccionado}")

        cboxSelectorCamera.bind("<<ComboboxSelected>>", handle_combobox_selection)

        root.mainloop()

    def ListCameras(self):
        index = 0
        cameras = []
        while True:
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            if not cap.read()[0]:
                break
            else:
                cameras.append(index)
            cap.release()
            index += 1
        return cameras
    
    def SelectedList(self, cboxSelectorCamera):
        selected_item = cboxSelectorCamera.get()
        if selected_item:
            index = cboxSelectorCamera.current()
            return index
        else:
            return None
        
    def OpenCamera(self, cboxSelectorCamera, nameNewFolder):

        currentIndex = self.SelectedList(cboxSelectorCamera)
        cap = cv2.VideoCapture(currentIndex, cv2.CAP_DSHOW)
        faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        count = 0

        while True:
            ret, frame = cap.read()
            if ret == False:
                break
            frame = imutils.resize(frame, width=640)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            auxFrame = frame.copy()

            faces = faceClassif.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                rostro = auxFrame[y:y + h, x:x + w]
                rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(nameNewFolder + '/face_{}.jpg'.format(count), rostro)
                count += 1
            cv2.imshow('frame', frame)

            k = cv2.waitKey(1)
            if k == 27 or count >= 300:
                break

        cap.release()
        cv2.destroyAllWindows()

        dataPath = 'Data/'
        peopleList = os.listdir(dataPath)
        print('Lista de personas: ', peopleList)

        labels = []
        facesData = []
        label = 0

        for nameDir in peopleList:
            personPath = dataPath + '/' + nameDir
            print('Leyendo las imágenes')

            for fileName in os.listdir(personPath):
                print('Rostros: ', nameDir + '/' + fileName)
                labels.append(label)
                facesData.append(cv2.imread(personPath+'/'+fileName,0))
            label = label + 1
        face_recognizer = cv2.face.LBPHFaceRecognizer_create()

        print("Entrenando...")
        face_recognizer.train(facesData, np.array(labels))

        face_recognizer.write('modeloLBPHFace.xml')
        print("Modelo almacenado...")
        

    def IndentifyPerson(self):
        root = tk.Tk()
        root.title("Camera selector")
        root.geometry("500x200")
        root.resizable(0,0)      
        cameras = self.ListCameras()
        cboxSelectorCamera = ttk.Combobox(root, values=cameras, state="readonly")
        cboxSelectorCamera.pack(pady=20, padx=20)

        btnStart = tk.Button(root, text="Start", 
                                 pady="5px", 
                                 padx="5px", 
                                 border=4,
                                 state="disabled",
                                 command= lambda: self.CameraIdentifyOpen(cboxSelectorCamera))
        btnStart.place(relx=0.5, rely=0.40, anchor="center")  

        def handle_combobox_selection(event):
            index_seleccionado = self.SelectedList(cboxSelectorCamera)
            if index_seleccionado is not None:
                btnStart.configure(state="normal")
                print(f"Índice seleccionado: {index_seleccionado}")

        cboxSelectorCamera.bind("<<ComboboxSelected>>", handle_combobox_selection)

        root.mainloop()

    def CameraIdentifyOpen(self, cboxSelectorCamera):
        currentIndex = self.SelectedList(cboxSelectorCamera)
        dataPath = "Data/"
        imagePaths = os.listdir(dataPath)
        cap = cv2.VideoCapture(currentIndex, cv2.CAP_DSHOW)
        print('imagePaths=',imagePaths)

        face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        face_recognizer.read('modeloLBPHFace.xml')
        faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

        while True:
            ret,frame = cap.read()
            if ret == False: break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            auxFrame = gray.copy()

            faces = faceClassif.detectMultiScale(gray,1.3,5)

            for (x,y,w,h) in faces:
                rostro = auxFrame[y:y+h,x:x+w]
                rostro = cv2.resize(rostro,(150,150),interpolation= cv2.INTER_CUBIC)
                result = face_recognizer.predict(rostro)

                cv2.putText(frame,'{}'.format(result),(x,y-5),1,1.3,(255,255,0),1,cv2.LINE_AA)

                # LBPHFace
                if result[1] < 70:
                    cv2.putText(frame,'{}'.format(imagePaths[result[0]]),(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
                    cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
                else:
                    cv2.putText(frame,'Desconocido',(x,y-20),2,0.8,(0,0,255),1,cv2.LINE_AA)
                    cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)
                
            cv2.imshow('frame',frame)
            k = cv2.waitKey(1)
            if k == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    InitFormBase = FormMenu()