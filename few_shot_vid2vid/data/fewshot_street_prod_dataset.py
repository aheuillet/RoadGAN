from data.fewshot_street_dataset import FewshotStreetDataset
from data.image_folder import make_grouped_dataset, check_path_valid
from data.base_dataset import BaseDataset, get_img_params, get_video_params, get_transform
from PIL import Image
from os import path
import os

class FewShotStreetProdDataset(FewshotStreetDataset):

    def initialize(self, opt):
        self.opt = opt     
        self.L_is_label = True#self.opt.label_nc != 0 
          
        self.L_paths = sorted(self.make_dataset(opt.seq_path))
        self.ref_I_paths = sorted(self.make_dataset(opt.ref_img_path))
        self.ref_L_paths = sorted(self.make_dataset(opt.ref_img_path.replace('images', 'labels')))
    
    def __getitem__(self, index):    
        opt = self.opt        
        L_paths = self.L_paths
        ref_L_paths, ref_I_paths = self.ref_L_paths, self.ref_I_paths
        
        
        ### setting parameters                
        n_frames_total, start_idx, t_step, ref_indices = get_video_params(opt, self.n_frames_total, len(L_paths), index)        
        w, h = opt.fineSize, int(opt.fineSize / opt.aspect_ratio)
        img_params = get_img_params(opt, (w, h))
        is_first_frame = opt.isTrain or index == 0

        transform_I = get_transform(opt, img_params, color_aug=opt.isTrain)
        transform_L = get_transform(opt, img_params, method=Image.NEAREST, normalize=False) if self.L_is_label else transform_I

        ### read in reference image
        Lr, Ir = self.Lr, self.Ir
        if is_first_frame:            
            for idx in ref_indices:                
                Li = self.get_image(ref_L_paths[idx], transform_L, is_label=self.L_is_label)            
                Ii = self.get_image(ref_I_paths[idx], transform_I)
                Lr = self.concat_frame(Lr, Li.unsqueeze(0))
                Ir = self.concat_frame(Ir, Ii.unsqueeze(0))

            if not opt.isTrain: # keep track of non-changing variables during inference                
                self.Lr, self.Ir = Lr, Ir


        ### read in target images
        L, I = self.L, self.I
        for t in range(n_frames_total):
            idx = start_idx + t * t_step            
            Lt = self.get_image(L_paths[idx], transform_L, is_label=self.L_is_label)
            L = self.concat_frame(L, Lt.unsqueeze(0))
            
        if not opt.isTrain:
            self.L, self.I = L, I
        
        seq = path.basename(path.dirname(opt.ref_img_path)) + '-' + opt.ref_img_id + '_' + path.basename(path.dirname(opt.seq_path))
        
        return_list = {'tgt_label': L, 'ref_label': Lr, 'ref_image': Ir,
                    'path': L_paths[idx], 'seq': seq}
        return return_list

    def make_dataset(self, datapath):
        files = []
        for f in os.listdir(datapath):
            files.append(os.path.join(datapath, f))
        return files
