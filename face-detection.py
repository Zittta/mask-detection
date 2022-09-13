import cv2
import tensorflow
import win32com.client as wincl
from PIL import Image, ImageOps
import numpy as np

webcam = cv2.VideoCapture(0)
success, image_bgr = webcam.read()

face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_classifier = cv2.CascadeClassifier("haarcascade_eye_tree_eyeglasses.xml")
np.set_printoptions(suppress=True)
model = tensorflow.keras.models.load_model('keras_model.h5')
size = (224, 224)

def text_speech(): #โปรแกรมเสียง
    syn = wincl.Dispatch('SAPI.Spvoice')
    '''syn.Rate = 0
    syn.Volume = 100'''
    s = 'Please wear the mask' #ข้อความที่ใช้อ่าน
    syn.Speak(s)

#ถ้าดึงรูปจาก webcam ได้จะขึ้น True และทำงานใน while loop ต่อไป
while True:
 
    success, image_bgr = webcam.read()
    image_org = image_bgr.copy()
    image_bw = cv2.cvtColor(image_bgr,cv2.COLOR_BGR2GRAY)
    image_rgb = cv2.cvtColor(image_bgr,cv2.COLOR_BGR2RGB)
    faces = face_classifier.detectMultiScale(image_bw)
    eyes = eye_classifier.detectMultiScale(image_bw)

    #print(f'There are {len(faces)} faces found.')#f' หรือ f string มีไว้ใช้แทน

    for (x,y,w,h) in eyes:
        cv2.rectangle(image_bgr, (x, y), (x+w, y+h), (0, 255, 0), 2)

    for (x,y,w,h) in faces:
        cface_rgb = Image.fromarray(image_rgb[y:y+h,x:x+w])

        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

        image = cface_rgb 
        image = ImageOps.fit(image, size, Image.ANTIALIAS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

        data[0] = normalized_image_array

        prediction = model.predict(data)
        print(prediction)

        if prediction[0][0] > prediction[0][1]:
            cv2.putText(image_bgr,'Masked',(x,y-7),cv2.FONT_HERSHEY_TRIPLEX,0.5,(0,255,0),2)
            cv2.rectangle(image_bgr, (x, y), (x+w, y+h), (0, 255, 0), 2)

        else :
            cv2.putText(image_bgr,'Non-Masked',(x,y-7),cv2.FONT_HERSHEY_TRIPLEX,0.5,(0,0,255),2)
            cv2.rectangle(image_bgr, (x, y), (x+w, y+h), (0, 0, 255), 2)

            text_speech()#ตัวเรียกใช้ฟังก์ชัน

    cv2.imshow("Mask Detection", image_bgr)
    cv2.waitKey(1) 
