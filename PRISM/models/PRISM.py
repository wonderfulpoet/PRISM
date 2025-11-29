import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from .base import Base, BaseTrain
from ..networks import PRISM, NLayerDiscriminator, add_gan_loss
from ..utils import print_model, get_device, ensure_rgb
from scipy.io import savemat
import os
import time
from PIL import Image

save_path = '/path/to/outputs'
os.makedirs(save_path, exist_ok=True)

class PRISMTrain(BaseTrain):
    def __init__(self, learn_opts, loss_opts, g_type, d_type, **model_opts):
        super(PRISMTrain, self).__init__(learn_opts, loss_opts)
        self.t = 0
        g_opts, d_opts = model_opts[g_type], model_opts[d_type]

        model_dict = dict(
            PRISM = lambda: PRISM(**g_opts),
            nlayer = lambda: NLayerDiscriminator(**d_opts))

        self.model_g = self._get_trainer(model_dict, g_type) # generators
        self.model_dl = add_gan_loss(self._get_trainer(model_dict, d_type)) 
        self.model_dh = add_gan_loss(self._get_trainer(model_dict, d_type)) 

        loss_dict = dict(
               l1 = nn.L1Loss,
               gl = (self.model_dl.get_g_loss, self.model_dl.get_d_loss), 
               gh = (self.model_dh.get_g_loss, self.model_dh.get_d_loss)) #

        self.model_g._criterion["ll"] = self._get_criterion(loss_dict, self.wgts["ll"], "ll_")
        self.model_g._criterion["hh"] = self._get_criterion(loss_dict, self.wgts["hh"], "hh_")
        self.model_g._criterion["lh"] = self._get_criterion(loss_dict, self.wgts["lh"], "lh_")
        self.model_g._criterion["lhl"] = self._get_criterion(loss_dict, self.wgts["lhl"], "lhl_")
        self.model_g._criterion["hlh"] = self._get_criterion(loss_dict, self.wgts["hlh"], "hlh_")
        self.model_g._criterion["noise"] = self._get_criterion(loss_dict, self.wgts["noise"], "noise_")
        self.model_g._criterion["gl"] = self._get_criterion(loss_dict, self.wgts["gl"])
        self.model_g._criterion["gh"] = self._get_criterion(loss_dict, self.wgts["gh"])
        self.model_g._criterion["cont"] = self._get_criterion(loss_dict, self.wgts["cont"], "cont")
        self.model_g._criterion["contdeep"] = self._get_criterion(loss_dict, self.wgts["contdeep"], "contdeep_")

    def _nonzero_weight(self, *names):
        wgt = 0
        for name in names:
            w = self.wgts[name]
            if type(w[0]) is str: w = [w]
            for p in w: wgt += p[1]
        return wgt

    def optimize(self, date_name, a, a_, b, c):
        self.x_low, self.x_low_gt, self.x_high, self.x_low_LI = self._match_device(a, a_, b, c)
        self.model_g._clear()
        self.l_h, self.h_l, self.h_h, self.l_h_l, self.h_l_h, self.noise1, self.noise2, self._negtive, self._positive, self._anchor = self.model_g.forward3(self.x_low, self.x_low_gt, self.x_high, self.x_low_LI)
        
        if self._nonzero_weight("gl", "lh", 'hh'):
            self.model_dl._clear()
            #print(f"Debug: Value of self.l_h is {self.l_h}, Type is {type(self.l_h)}")
            self.model_g._criterion["gl"](self.l_h, self.x_high)
            self.model_g._criterion["lh"](self.l_h, self.x_low_gt)
            self.model_g._criterion["hh"](self.h_h, self.x_high)

        if self._nonzero_weight("gh"):
            self.model_dh._clear()
            self.model_g._criterion["gh"](self.h_l, self.x_low)

        if self._nonzero_weight("lhl"):
            self.model_g._criterion["lhl"](self.l_h_l, self.x_low)

        if self._nonzero_weight("hlh"):
            self.model_g._criterion["hlh"](self.h_l_h, self.x_high)

        if self._nonzero_weight("noise"):
            self.model_g._criterion["noise"](self.noise1,self.noise2)
        
        self.model_g._update()
        self.model_dl._update()
        self.model_dh._update()
        
        self.loss = self._merge_loss(
            self.model_dl._loss, self.model_dh._loss, self.model_g._loss)

    def get_visuals(self, n=8):
        lookup = [
            ("l", "x_low"), ("lh", "l_h"), ("lhl", "l_h_l"),
            ("h", "x_high"), ("hl", "h_l"), ("hh", "h_h"), ("hlh", "h_l_h")]
        def safe_func(x):
            if x.dim() == 3: 
                x = x.unsqueeze(1)                       
            return x
        return self._get_visuals(lookup, n, func = safe_func)

    def evaluate(self, loader, metrics):
        progress = tqdm(loader)
        res = defaultdict(lambda: defaultdict(float))
        cnt = 0
        for img_low, img_high, img_low_LI in progress:
            img_low, img_high, img_low_LI = self._match_device(img_low, img_high, img_low_LI)

            def to_numpy(*data):
                data = [loader.dataset.to_numpy(d, False) for d in data]
                return data[0] if len(data) == 1 else data

            pred_ll, pred_lh = self.model_g.forward3(img_low, img_low_LI)
            pred_hl, pred_hh = self.model_g.forward4(img_low, img_low_LI)
            pred_hlh = self.model_g.forward_lh(pred_hl)
            img_low, img_high, pred_ll, pred_lh, pred_hl, pred_hh, pred_hlh = to_numpy(
                img_low, img_high, pred_ll, pred_lh, pred_hl, pred_hh, pred_hlh)

            met = {
                "ll": metrics(img_low, pred_ll),
                "lh": metrics(img_high, pred_lh),
                "hl": metrics(img_low, pred_hl),
                "hh": metrics(img_high, pred_hh),
                "hlh":metrics(img_high, pred_hlh)}

            res = {n: {k: (res[n][k] * cnt + v) / (cnt + 1) for k, v in met[n].items()} for n in met}
            desc = "[{}]".format("/".join(met["ll"].keys()))
            for n, met in res.items():
                vals = "/".join(("{:.2f}".format(v) for v in met.values()))
                desc += " {}: {}".format(n, vals)
            progress.set_description(desc=desc)


class PRISMTest(Base):
    def __init__(self, g_type, **model_opts):
        super(PRISMTest, self).__init__()
        self.t = 0
        self.total_images = 0  
        self.total_generation_time = 0.0  

        g_opts = model_opts[g_type]
        model_dict = dict(PRISM = lambda: PRISM(**g_opts))
        self.model_g = model_dict[g_type]()

    def evaluate(self, A, A_, B, C, name=None):
        self.x_low, self.x_low_gt, self.x_high, self.x_low_LI = self._match_device(A, A_, B, C)
        self.name = name

        self.model_g.eval()

        start_time = time.time()

        with torch.no_grad():  
            self.l_h, self.h_l, self.h_h, self.l_h_l, self.h_l_h, self.noise1, self.noise2, self._negtive, self._positive, self._anchor = self.model_g.forward3(self.x_low, self.x_low_gt, self.x_high, self.x_low_LI)

            num_images = min(10, len(self.l_h))  
            
            for iii in range(num_images):
                WINDOW_MIN_COEFF = 0.1584  
                WINDOW_MAX_COEFF = 0.2448  

                arr = torch.squeeze(self.l_h[iii]).cpu().detach().numpy()
                ori = torch.squeeze(self.x_low[iii]).cpu().detach().numpy()

                arr_clipped = np.clip(arr, WINDOW_MIN_COEFF, WINDOW_MAX_COEFF)
                ori_clipped = np.clip(ori, WINDOW_MIN_COEFF, WINDOW_MAX_COEFF)

                arr_norm = (arr_clipped - WINDOW_MIN_COEFF) / (WINDOW_MAX_COEFF - WINDOW_MIN_COEFF)
                ori_norm = (ori_clipped - WINDOW_MIN_COEFF) / (WINDOW_MAX_COEFF - WINDOW_MIN_COEFF)

                arr_uint8 = (arr_norm * 255).astype(np.uint8)
                ori_uint8 = (ori_norm * 255).astype(np.uint8)

                arr_rgb = ensure_rgb(arr_uint8)
                ori_rgb = ensure_rgb(ori_uint8)

                img1 = Image.fromarray(arr_rgb, mode='RGB')
                img2 = Image.fromarray(ori_rgb, mode='RGB')

                name = self.name[iii].split("test_640geo/")[-1].replace("/", "-")

                save_file1 = os.path.join(save_path, name + ".png")
                img1.save(save_file1)
                save_file2 = os.path.join(save_path, name + "_.png")
                img2.save(save_file2)
                
                self.t += 1

            end_time = time.time()
            batch_time = end_time - start_time

            self.total_images += num_images
            self.total_generation_time += batch_time
            
            avg_time_per_image = batch_time / num_images if num_images > 0 else 0
            print(f"{num_images}pages, Time: {batch_time:.4f}s, Average: {avg_time_per_image:.4f}s/page")
    def get_pairs(self):
        return [
            ("before", (self.x_low, self.x_high)), 
            ("after", (self.l_h, self.x_high))], self.name

    def get_visuals(self, n=8):
        lookup = [
            ("l", "x_low"), ("lh", "l_h"),
            ("h", "x_high"), ("hl", "h_l"), ("hh", "h_h")]
        def func(x):
            if x.dim() == 3:  
                x = x.unsqueeze(1)                       
            return x
        return self._get_visuals(lookup, n, func, False)
    
    def resume(self, checkpoint_file):
        self._checkpoint_path = checkpoint_file
        print(f"Loading ckpt: {checkpoint_file}")
        
        try:
            checkpoint = torch.load(checkpoint_file, map_location='cpu')
            print(f"key: {list(checkpoint.keys())}")
            
            if 'model_g' in checkpoint:
                missing_keys, unexpected_keys = self.model_g.load_state_dict(checkpoint['model_g'], strict=False)
                if missing_keys:
                    print(f"Missed key: {missing_keys}")
                if unexpected_keys:
                    print(f"Unexpected key: {unexpected_keys}")
            else:
                missing_keys, unexpected_keys = self.model_g.load_state_dict(checkpoint, strict=False)
                if missing_keys:
                    print(f"Missed key: {missing_keys}")
                if unexpected_keys:
                    print(f"Unexpected key: {unexpected_keys}")
            
        except Exception as e:
            print(f"Failed: {e}")

        print("Load Completed\n")
