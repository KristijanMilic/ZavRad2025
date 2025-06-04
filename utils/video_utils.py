import os
import shutil

import cv2
import numpy as np

FPS_CAP = 60

def split_video(rel_path):
    capture = cv2.VideoCapture(rel_path)
    capture.set(cv2.CAP_PROP_FPS, FPS_CAP)

    try_create_data_folder()

    currentFrame = 0
    while(True):
        ret, frame = capture.read()

        if not ret:
            break

        name = './data/frame' + str(currentFrame) + '.jpg'
        cv2.imwrite(name, frame)
        currentFrame += 1

    capture.release()
    cv2.destroyAllWindows()

def synthesize_video(rel_path):
    writer = cv2.VideoWriter("output.mp4", 0, 60, (1920, 1080))
    writer.set(cv2.CAP_PROP_FPS, FPS_CAP)

    try_create_data_folder()

    images = [img for img in os.listdir(rel_path) if img.endswith('.jpg')]

    for image in images:
        image = cv2.imread(os.path.join(rel_path, image))
        image = cv2.resize(image, (1920, 1080))
        writer.write(image)

    writer.release()
    cv2.destroyAllWindows()

    try_delete_data_folder()

def try_create_data_folder():
    try:
        if not os.path.exists('data'):
            os.makedirs('data')
    except OSError:
        print ('Error: Creating directory of data')

def try_delete_data_folder():
    try:
        shutil.rmtree('./data')
    except OSError:
        print ('Error: Deleting directory of data')