import cv2
import os
from tqdm import tqdm
import shutil
import random
import numpy as np
from PIL import Image

def decompose_video(video_path):
    '''Decompose in png images the frames of the video located at video_path.
    Can optionally downscale them when writing to disk. 
        - video_path: str
        - resize: bool, default: True
    '''
    vidcap = cv2.VideoCapture(video_path)
    vid_name = os.path.basename(video_path).split(".")[0]
    success, image = vidcap.read()
    count = 0
    save_path = os.path.join(os.path.dirname(video_path), vid_name)
    os.makedirs(save_path, exist_ok=True)
    print("Reading video file...")
    with tqdm() as pbar:
        while success:
            pal = Image.open('inference/palette.png')
            image = Image.fromarray(image).quantize(colors=20, palette=pal, dither=Image.NONE)  
            image.save(os.path.join(save_path, "%d.png" % count))
            success, image = vidcap.read()
            count += 1
            pbar.update()
    return vid_name



def recompose_video(img_dir_path, save_path):
    resize_images(img_dir_path, (2048, 1024))
    print("Recomposing video...")
    sample_path = random.choice(os.listdir(img_dir_path))
    sample = np.array(cv2.imread(os.path.join(img_dir_path, sample_path)))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(save_path, fourcc, 24,
                             (sample.shape[1], sample.shape[0]))
    for f in tqdm(sorted(os.listdir(img_dir_path), key=lambda x: int(x.split(".")[0]))):
            #files.sort(key=lambda x: int(x.split(".")[0]))
        if f.split(".")[1] in ["png", "jpg", "jpeg", "PNG", "JPEG"]:
            img = cv2.imread(os.path.join(img_dir_path, f))
            writer.write(img)
    writer.release()

def resize_images(img_dir_path, size=(1024, 512)):
    '''Resize images present in the given directory to the given size.
        - img_dir_path: str
        - size: tuple (width, height), default: (1024, 512)
    '''
    print("Resizing images...")
    for f in tqdm(sorted(os.listdir(img_dir_path), key=lambda x: int(x.split(".")[0]))):
        p = os.path.join(img_dir_path, f)
        img = Image.open(p).resize(size, resample=Image.LANCZOS)
        img.save(p)



#recompose_video('/home/alexandre/Documents/RoadGAN/inference/test_palette', '/home/alexandre/Documents/RoadGAN/inference/test_palette.mp4')
#decompose_video('./inference/test_germany.mp4')
#resize_images('/home/alexandre/Documents/RoadGAN/inference/seq_1')