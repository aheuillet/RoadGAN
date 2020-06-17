import gc
from model import create_model
from scipy.stats import truncnorm
import torchvision.transforms as transforms
from torch.autograd import Variable
import functools
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch
import random
from tqdm import tqdm
import argparse
import os
import scipy.io as sio
import numpy as np
from PIL import Image
import io
import sys
from time import sleep
sys.path.append('.')

# Semantic Segmentation Module
from semantic_segmentation_pytorch.lib.utils import as_numpy, mark_volatile
import semantic_segmentation_pytorch.lib.utils.data as torchdata
from semantic_segmentation_pytorch.dataset import TestDataset
from semantic_segmentation_pytorch.lib.nn import user_scattered_collate, async_copy_to
from semantic_segmentation_pytorch.models import ModelBuilder, SegmentationModule


class StyleGenerator:
    '''Class holding methods related to transient attributes style generation. 
    This will attempt to use a CUDA compatible GPU if available.'''

    def __init__(self, image_size, imageSize, video_path, model_path, selected_attribute, attribute_path='./fixed_attribute.npy', manual_seed=0, workers=2, gpu_ids=[0], batchSize=1, nz=100):
        if manual_seed is None:
            self.manual_seed = random.randint(1, 10000)
            random.seed(manual_seed)
            torch.manual_seed(manual_seed)

        if len(gpu_ids) > 0:
            torch.cuda.set_device(gpu_ids[0])
            torch.cuda.manual_seed_all(manual_seed)

        cudnn.benchmark = True

        ngpu = len(gpu_ids)
        self.nc = 3

        self.netG, self.netD = create_model(ngpu)
        self.input = torch.FloatTensor(
            batchSize, 3, imageSize, imageSize)
        self.segment = torch.FloatTensor(
            batchSize, 8, imageSize, imageSize)
        self.category = torch.FloatTensor(
            batchSize, 1, imageSize, imageSize)
        self.noise = torch.FloatTensor(batchSize, nz, 1, 1)
        netGDict = torch.load(model_path, map_location='cpu')

        model_dict = self.netG.state_dict()
        for k, v in netGDict.items():
            if k in model_dict and v.size() == model_dict[k].size():
                print(k + "\n")
                model_dict[k] = v
        self.netG.load_state_dict(model_dict)

        del netGDict
        del model_dict

        if ngpu > 0:
            self.netG = self.netG.cuda().eval()
            self.segment = self.segment.cuda()
            self.noise = self.noise.cuda()
            self.category = self.category.cuda()
        self.image_size = image_size

        self.attribute_name = '1234'
        self.attributes = open('attributes.txt', 'r').readlines()
        self.attributes = [a.replace('\n', '') for a in self.attributes]

        self.transient_attribute = np.zeros((1, 40))
        self.selected_attribute = selected_attribute
        self.video_path = video_path
        self.label_path = os.path.join(self.video_path, 'labels')
        self.generated_path = os.path.join(self.video_path, 'generated')

        binarycodesMat = sio.loadmat('binarycodes.mat')
        self.binarycodes = binarycodesMat['binarycodes']
        objectsMat = sio.loadmat('objectName150.mat')
        objects = objectsMat['objectNames']
        self.objects = objects[:, 0].tolist()
        
        self.images = self.detect_reference_images()
    
    def detect_reference_images(self):
        '''Detects how many source images will be used for hallucinating a new style.'''
        images = []
        count = 0
        for i in sorted(os.listdir(self.video_path)):
            filepath = os.path.join(self.video_path, i)
            if os.path.isfile(filepath):
                images.append(i)
        images.sort(key=lambda x: int(x.split(".")[0]))
        for i in images:
            if int(i.split('.')[0]) != count:
                images.remove(i)
            else:
                count += 5*24
        return images

    def process_selected_attributes(self):
        '''Update the transient attribute array to increase the selected attributes.'''
        for a in self.selected_attribute:
            index = self.attributes.index(a)
            self.transient_attribute[0, index] += 10
	
    def load_segmentation_module(self):
        '''Load the MIT CSAIL segmentation module used for segmenting the source images.'''
        model_path = "./semantic_segmentation_pytorch/ade20k-resnet50dilated-ppm_deepsup"
        suffix = "_epoch_20.pth"
        arch_encoder = 'resnet50dilated'
        arch_decoder = 'ppm_deepsup'
        fc_dim = 2048
        num_class = 150

        weights_encoder = os.path.join(model_path,
                                       'encoder' + suffix)
        weights_decoder = os.path.join(model_path,
                                       'decoder' + suffix)
        builder = ModelBuilder()
        net_encoder = builder.build_encoder(
            arch=arch_encoder,
            fc_dim=fc_dim,
            weights=weights_encoder)
        net_decoder = builder.build_decoder(
            arch=arch_decoder,
            fc_dim=fc_dim,
            num_class=num_class,
            weights=weights_decoder,
            use_softmax=True)

        crit = nn.NLLLoss(ignore_index=-1)

        self.segmentation_module = SegmentationModule(
            net_encoder, net_decoder, crit)
        self.segmentation_module.cuda()
        self.segmentation_module.eval()
        print("Segmentation Module was created!")
    
    def segment_images(self):
        '''Segment all the source images.'''
        self.load_segmentation_module()
        num_val = -1
        list_test = []
        os.makedirs(self.label_path, exist_ok=True)

        for i in os.listdir(self.video_path):
            filename = os.path.join(self.video_path, i)
            if os.path.isfile(filename):
                list_test.append({'fpath_img': filename})

        dataset_val = TestDataset(
            list_test, opt, max_sample=num_val)
        loader_val = torchdata.DataLoader(
            dataset_val,
            batch_size=1,
            shuffle=False,
            collate_fn=user_scattered_collate,
            num_workers=5,
            drop_last=True)
        
        print("Semantic segmentation of images...")
        pbar = tqdm(total=len(loader_val))
        for batch_data in loader_val:
            # process data
            batch_data = batch_data[0]
            filename = os.path.basename(batch_data['info']).split('.')[0]
            segSize = (batch_data['img_ori'].shape[0],
                    batch_data['img_ori'].shape[1])
            img_resized_list = batch_data['img_data']

            with torch.no_grad():
                scores = torch.zeros(1, 150, segSize[0], segSize[1])
                scores = async_copy_to(scores, 0)

                for img in img_resized_list:
                    feed_dict = batch_data.copy()
                    feed_dict['img_data'] = img
                    del feed_dict['img_ori']
                    del feed_dict['info']
                    feed_dict = async_copy_to(feed_dict, 0)

                    # forward pass
                    pred_tmp = self.segmentation_module(feed_dict, segSize=segSize)
                    scores = scores + pred_tmp / len(opt.imgSizes)

                _, preds = torch.max(scores, dim=1)
                preds = as_numpy(preds.squeeze(0).cpu())
                where_are_NaNs = np.isnan(preds)
                preds[where_are_NaNs] = 0
                preds = preds.astype('uint8') + 1
                imgGray = Image.fromarray(preds)
                imgGray.save(os.path.join(self.label_path, filename + '_LayGray.png'), "PNG")
                pbar.update()
					

        # Freeing segmentation module
        del self.segmentation_module

        gc.collect()
        torch.cuda.empty_cache()
    
    def binary_encode_image(self, catImage):
        '''Binary encode the given PIL image.
        
        type: catImage: PIL.Image.Image;
        param: catImage: Image to be binary encoded;'''
        img_np = np.array(list(catImage.getdata()))
        binaryim = np.zeros((self.image_size*self.image_size, 8))
        binaryim = np.zeros((catImage.size[0]*catImage.size[1], 8))
        for i in range(150):
            a = np.where(img_np==(i + 1))[0]
            for j in a:
                binaryim[j, :] = self.binarycodes[i, :]

        return binaryim.reshape(catImage.size[1], catImage.size[0], 8)

    def _colorencode(self, category_im):
        '''Color the given image using predefined color palette.
        
        type: category_im: numpy.array;
        param: category_im: the image to be colorized;'''
        colorcodes = sio.loadmat("./color150.mat")
        colorcodes = colorcodes['colors']
        idx = np.unique(category_im)
        h, w = category_im.shape
        colorCodeIm = np.zeros((h, w, 3)).astype(np.uint8)
        for i in range(idx.shape[0]):
            if idx[i] == 0:
                continue
            b = np.where(category_im == idx[i])
            rgb = colorcodes[idx[i] - 1]
            colorCodeIm[b] = rgb
        return colorCodeIm
    
    def transform_image(self, seg):
        '''Return colorized and grayscale versions of the given segmentation map.
        
        type: seg: PIL.Image.Image;
        param: seg: a segmentation map;'''
        graycode = np.array(seg)
        colorcode = self._colorencode(graycode)
        graycode = Image.fromarray(graycode)
        colorcode = Image.fromarray(colorcode, 'RGB')
        return colorcode, graycode

    def transform_and_resize_image(self, image, seg):
        '''Return colorized and grayscale versions of the given segmentation map and also 
        resize and crop the given image.
        
        type: seg: PIL.Image.Image;
        param: seg: a segmentation map;
        type: image: PIL.Image.Image;
        param: image: an hallucinated style image;'''
        # Resize
        resize = transforms.Resize(self.image_size)
        image = resize(image)
        resize = transforms.Resize(self.image_size, interpolation=Image.NEAREST)
        seg = resize(seg)
        # Center crop
        crop = transforms.CenterCrop((self.image_size, self.image_size))
        image = crop(image)
        seg = crop(seg)
        return image, self.transform_image(seg)
    
    def init_z(self, batchsize):
        '''Init a noise to be added to the generator latent space.
        
        type: batchsize: int;
        param: batchsize: the current batchsize used for generation;'''
        self.noise.resize_(batchsize, 100, self.image_size, self.image_size).normal_(0, 1)
    
    def inverse_transform(self, X):
        npx = self.image_size
        X = (np.reshape(X, (-1, self.nc, npx, npx)).transpose(0, 2, 3, 1)+1.)/2.*255
        return X
    
    def process_images(self):
        self.segment_images()
        self.process_selected_attributes()
        os.makedirs(self.generated_path)
        print("Generating style images...")

        for i in tqdm(self.images):
            img_path = os.path.join(self.video_path, i)
            basename = os.path.basename(i).split('.')[0]
            label_img_path = os.path.join(self.label_path, basename + '_LayGray.png') 
            img_original = Image.open(img_path)
            img_gray = Image.open(label_img_path)
            img_original_resized, img_original_color_resized, img_original_gray_resized = self.transform_and_resize_image(img_original, img_gray)
            img_original_gray_resized.save(os.path.join(self.label_path, basename + '_LayGrayResized.png'), "PNG")

            segment_binary = self.binary_encode_image(img_original_gray_resized)
            objectcategories = np.reshape(np.array(list(img_original_gray_resized.getdata())), (img_original_gray_resized.size[0], img_original_gray_resized.size[1]))
            cat_np = objectcategories
            cat = torch.from_numpy(cat_np).float()
            self.category.resize_as_(cat.cuda()).copy_(cat)
            self.init_z(1)
            self.generate_style_image(segment_binary, basename)



    def generate_style_image(self, segment_binary, basename):
        seg_np = segment_binary
        seg_np = seg_np[np.newaxis]
        seg_np = np.transpose(seg_np, (0, 3, 1, 2))
        seg = torch.from_numpy(seg_np).float()
        self.segment.resize_as_(seg.cuda()).copy_(seg)

        att_np = self.transient_attribute[:, :]
        att = torch.from_numpy(att_np).float()
        attribute = att.cuda()

        fixed_noise = self.noise
        sorted_inds = np.argsort(att_np[0])
        sorted_inds = sorted_inds[::-1]
        best_inds = sorted_inds[:5]
        for ix in best_inds:
            print(self.attributes[ix].strip() + ': ',
                  self.transient_attribute[0, ix])

        fake = self.netG(Variable(fixed_noise), Variable(
            self.segment), Variable(attribute))
        imnp = fake.data.cpu().numpy()
        gen_im = self.inverse_transform(imnp)
        gen_im = gen_im[0]
        im = Image.fromarray(np.uint8(gen_im))
        im.save(os.path.join(self.generated_path, basename + '_G.png'))
        
        torch.cuda.empty_cache()
        gc.collect()
        print('Generation completed!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int,
                        help='number of data loading workers', default=2)
    parser.add_argument('--batchSize', type=int,
                        default=1, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=128,
                        help='the height / width of the input image to network')
    parser.add_argument('--nz', type=int, default=100,
                        help='size of the latent z vector')
    parser.add_argument('--gpu_ids', type=str, default='0',
                        help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')

    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--model_path', default='./pretrained_models/sgn_enhancer_G_latest.pth', help="pretrained model path")
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--isTest', action='store_true', help='test')
    parser.add_argument('--video_path', default='./video')
    parser.add_argument('--attributes', type=str, nargs='+', help='attributes to be present in the synthesized images')

    # Semantic Segmentation Module Options
    parser.add_argument('--imgSizes', default=[300, 400, 500, 600],
                        nargs='+', type=int,
                        help='list of input image sizes.'
                        'for multiscale testing, e.g. 300 400 500')
    parser.add_argument('--imgMaxSize', default=1000, type=int,
                        help='maximum input image size of long edge')
    parser.add_argument('--padding_constant', default=8, type=int,
                        help='maxmimum downsampling rate of the network')
    parser.add_argument('--segm_downsampling_rate', default=8, type=int,
                        help='downsampling rate of the segmentation label')

    opt = parser.parse_args()
    print(opt)

    SG = StyleGenerator(opt.image_size, opt.imageSize, opt.video_path, opt.model_path, opt.attributes)
    SG.process_images()
