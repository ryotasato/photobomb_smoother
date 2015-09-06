# -*- coding: utf-8 -*-

__author__  = "Ryota Sato"
__version__ = "1.0.0"
__date__    = "7 Sep 2015"

import cv2
import numpy as np
from sklearn.cluster import KMeans
import os

"""
OpenCVとscikit-learnを用いた簡易な自動Photobombボカし器
"""

######## ######## ######## ######## ######## ######## ######## ########

def detect_photobombs(faces):
    """
    顔領域を被写体とPhotobombに分ける
    """
    # 顔領域の大小で2クラス分類
    face_sizes = []
    for face in faces:
        face_sizes.append(face[2])
    face_sizes = np.vstack(face_sizes)
    kmeans_model = KMeans(n_clusters=2, random_state=10).fit(face_sizes)
    labels = kmeans_model.labels_

    # どちらのクラスが大きな顔のクラスかを調べる
    class_a = []
    class_b = []
    for i in range(len(face_sizes)):
        if labels[i] == 0:
            class_a.append(face_sizes[i])
        else:
            class_b.append(face_sizes[i])
    mean_a = np.mean(class_a)
    mean_b = np.mean(class_b)
    # Photobombのラベルがどちらか調べる
    label_photobomb = 0
    if mean_a > mean_b:
        label_photobomb = 1

    # 顔を被写体とPhotobombに分ける
    subjects = []
    photobombs = []
    for i in range(len(faces)):
        if labels[i] == label_photobomb:
            photobombs.append(faces[i])
        else:
            subjects.append(faces[i])

    return subjects, photobombs

######## ######## ######## ######## ######## ######## ######## ########
    
def compose_rectangles(img, faces):
    """
    顔領域に矩形を合成
    """
    for face in faces:
        cv2.rectangle(img,tuple(face[0:2]),tuple(face[0:2]+face[2:4]), (255, 255, 255), thickness=1)
    return img

######## ######## ######## ######## ######## ######## ######## ########

def smooth_faces(img, faces):
    """
    顔領域を平滑化
    """
    for face in faces:
        x = face[0]
        y = face[1]
        width = face[2]
        height = face[3]
        img[y:y+height, x:x+width] = cv2.blur(img[y:y+height, x:x+width], (64, 64), (0, 0))
    return img
    
######## ######## ######## ######## ######## ######## ######## ########

if __name__ == '__main__':

    cascade_path = "./haarcascade_frontalface_alt.xml"
    cascade = cv2.CascadeClassifier(cascade_path) 

    directory = "./../img/"
    img_paths = map(lambda a:os.path.join(directory,a),os.listdir(directory))
    
    for img_path in img_paths:
    	img = cv2.imread(img_path)
        img_gray = cv2.cvtColor(img, cv2.cv.CV_BGR2GRAY)
        # 画像から顔領域を検出
        faces = cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=2, minSize=(64,64))  
        # 顔が検出されたら
        if 0 < len(faces):
            # 検出された顔が複数ならPhotobombを探す
            if 1 < len(faces):
                # 顔を被写体とPhotobombに分ける
                subjects, photobombs = detect_photobombs(faces)
                # Photobombをボカす
                img = smooth_faces(img, photobombs)
            # すべての顔に矩形を表示
            img = compose_rectangles(img, faces)
        # 出力
        file_name = img_path.split("/")[-1]
        out_file_name = "./../res/res_" + file_name
        cv2.imwrite(out_file_name, img)
        print("done:" + file_name)