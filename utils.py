import cv2
import os
from tqdm import tqdm
import shutil
import random
import numpy as np

def decompose_video(video_path):
    vidcap = cv2.VideoCapture(video_path)
    vid_name = video_path.split("/")[-1].split(".")[0]
    success,image = vidcap.read()
    count = 0
    if os.path.exists(vid_name):
        shutil.rmtree(vid_name)
    os.mkdir(vid_name)
    print("Reading video file...")
    with tqdm() as pbar:
        while success:
            cv2.imwrite(vid_name + "/_frame%d.jpg" % count, image)     # save frame as JPEG file      
            success,image = vidcap.read()
            count += 1
            pbar.update()

def recompose_video(img_dir_path, save_path):
    sample_path = random.choice(os.listdir(img_dir_path))
    sample = np.array(cv2.imread(os.path.join(img_dir_path, sample_path)))
    fourcc = cv2.VideoWriter_fourcc(*'MP4V') 
    writer = cv2.VideoWriter(save_path, fourcc, 30, (1280, 720))
    for root, dirs, files in os.walk(img_dir_path):
        with tqdm(total=len(files)) as pbar:
            for f in files.sort(key=int):
                print(f)
                img = cv2.imread(f)
                writer.write(img)



recompose_video("./sample", "recomposed.mp4")