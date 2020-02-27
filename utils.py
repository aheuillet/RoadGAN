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
            cv2.imwrite(vid_name + "/%d.jpg" % count, image)     # save frame as JPEG file      
            success,image = vidcap.read()
            count += 1
            pbar.update()

def recompose_video(img_dir_path, save_path):
    sample_path = random.choice(os.listdir(img_dir_path))
    sample = np.array(cv2.imread(os.path.join(img_dir_path, sample_path)))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    print(sample.shape[0])
    print(sample.shape[1]) 
    writer = cv2.VideoWriter(save_path, fourcc, 25, (sample.shape[1], sample.shape[0]))
    for root, dirs, files in os.walk(img_dir_path):
        with tqdm(total=len(files)) as pbar:
            files.sort(key=lambda x: int(x.split(".")[0]))
            for f in files:
                img = cv2.imread(os.path.join(img_dir_path, f))
                writer.write(img)
                pbar.update()
    writer.release()



#decompose_video("sample.mp4")
recompose_video("./sample", "recomposed.mp4")