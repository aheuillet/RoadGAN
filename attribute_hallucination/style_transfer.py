import torch
import argparse
import os
from tqdm import tqdm
import sys
sys.path.append('.')

from WCT2.transfer import WCT2
from WCT2.utils.io import Timer, open_image, load_segment, compute_label_info
from torchvision.utils import save_image

def select_style_img_nb(frame_nb, frequency=5, frames_per_second=24):
    "Returns the number of the style image corresponding to input frame number."
    return (frame_nb//(frequency*frames_per_second))*(frequency*frames_per_second)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--video_folder', type=str)
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--style_frequency', type=int, default=5, help='Frequency (in s) at which change of style image')
    parser.add_argument('--frames_per_second', type=int, default=24)
    parser.add_argument('--save_folder', type=str, default='../tmp')
    opt = parser.parse_args()

    generated_path = os.path.join(opt.video_folder, 'generated')
    label_path = os.path.join(opt.video_folder, 'labels')
    stylized_path = os.path.join(opt.save_folder,  opt.video_folder.split('/')[-1] + '_stylized')
    os.makedirs(stylized_path, exist_ok=True)


    transfer_at = set()

    transfer_at.add('encoder')

    transfer_at.add('decoder')

    transfer_at.add('skip')
    device = 'cuda:0'
    device = torch.device(device)
    wct2 = WCT2(transfer_at=transfer_at, option_unpool='cat5',
                device=device, verbose=False)

    print("Applying style transfer...")
    for i in tqdm(sorted(os.listdir(opt.video_folder))):
        filepath = os.path.join(opt.video_folder, i)
        if os.path.isfile(filepath):
            frame_nb = int(i.split('.')[0])
            content = open_image(os.path.join(opt.video_folder, i), opt.image_size).to(device)
            style_img_nb = select_style_img_nb(frame_nb, opt.style_frequency, opt.frames_per_second)
            style = open_image(os.path.join(generated_path, str(style_img_nb) + '_G.png'), opt.image_size).to(device)
            content_segment = load_segment(
                os.path.join(label_path, str(frame_nb)+"_LayGray.png"), opt.image_size)
            style_segment = load_segment(
                os.path.join(label_path, str(style_img_nb)+"_LayGrayResized.png"), opt.image_size)
            with torch.no_grad():
                img = wct2.transfer(content, style, content_segment, style_segment, 1)
            save_image(img.clamp_(0, 1), os.path.join(stylized_path, i), padding=0)
