from PIL import Image
import argparse


def resize_image(path, size):
    '''Resize an image at a given size using Lanczos filter.
    Overwrites the original.'''
    img = Image.open(path)
    img = img.resize(size, resample=Image.LANCZOS)
    img_ext = path.split("/")[-1].split(".")[1]
    img.save(path.replace(img_ext, "jpg"))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--height", type=int, default=1024)
    args = parser.parse_args()
    resize_image(args.name, (args.width, args.height))

