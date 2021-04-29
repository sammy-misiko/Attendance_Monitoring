from flask import Flask, request, redirect
from flask import Response, render_template, url_for
from datetime import datetime
import face_recognition
import numpy as np
import pandas as pd
from time import *
import os
import cv2

app = Flask(__name__)

df = pd.DataFrame(columns=["Name","Date","Time"])



# load known images 
def load(clas):
    image = []
    stud_names = []
    path = 'paths'
    f_path = os.listdir(path)
    file_list = []

    for file in f_path:
        if file == clas.upper():
            for dirpath, dirname, dirfile in os.walk(clas):
                file_list = dirfile

            for name in file_list:
                mg = face_recognition.load_image_file(f"{path}/{file}/{name}")
                image.append(mg)
                stud_names.append(os.path.splitext(name)[0])
    return image, stud_names




# taking the attendance
def attendance(name):

    global df
    now = datetime.now()
    time = now.strftime("%H:%M:%S")
    date = strftime("%m-%d-%Y", localtime())

    if any(df['Name'] == name):
        pass
    else:
        df = df.append({'Name':name, 'Date':date, 'Time':time}, ignore_index= True)


#getting the encodings
def find_encordings(image_list):

    encoding_list = []

    for img in image_list:
        mg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encod = face_recognition.face_encodings(mg)[0]

        encoding_list.append(encod)

    return encoding_list



def all(clas,sheet):

    image, stud_name = load(clas)
    encodings = find_encordings(image)

    cap = cv2.VideoCapture(0)

    global df
    while True:
        check, frame = cap.read()
        small_img = cv2.resize(frame, (0,0), None, 0.25, 0.25)
        small_img = cv2.cvtColor(small_img, cv2.COLOR_BGR2RGB)

        face_small_img = face_recognition.face_locations(small_img)
        encod_small_img = face_recognition.face_encodings(small_img, face_small_img)

        for face_small, encod_small in zip(face_small_img, encod_small_img):
            match = face_recognition.compare_faces(encodings, encod_small)
            face_dis = face_recognition.face_distance(encodings, encod_small)

            match_index = np.argmin(face_dis)

            if match[match_index]:
                name = stud_name[match_index]

                y1, x2, y2, x1 = face_small
                x1,x2,y1,y2 = x1 * 4, x2 * 4, y1 * 4, y2 * 4

                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 3)
                cv2.rectangle(frame, (x1,y2-35),(x2,y2),(0,255,0), cv2.FILLED)
                cv2.putText(frame, name, (x1+6,y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255))

                attendance(name)

        df.to_csv(sheet)

        cv2.imshow('Webcam', frame)
        key = cv2.waitKey(1)

        if key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/submit', methods=['POST','GET'])
def submit():
    if request.method == 'POST':
        data = request.form['cl']
        data1 = request.form['cll']
        return all(data,data1)
    else:
        return render_template('home.html')


if(__name__=='__main__'):
    app.run(debug=True)