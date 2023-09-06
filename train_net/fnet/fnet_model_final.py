import os
import torch
import torch.nn as nn
import importlib
import pdb
from torch.autograd import Variable
from UNext import R2U_Net
from vit import ViT
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
from ResNet_2 import ResUnet   
import torch.nn.functional as F
from UNet2 import Unet
from memory_profiler import profile
from tifffile import imread, imwrite
from PIL import Image

def tv_loss(y):
    a, b, c, d = y.size()
    REGULARIZATION = a * b * c * d
    reg_loss = (
        torch.sum(torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])) + 
        torch.sum(torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :]))
    ) / REGULARIZATION
    
    return reg_loss

   
def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    g = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return g.div(a * b * c * d)
    
class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature)

    def forward(self, input):
        g = gram_matrix(input)
        return F.mse_loss(g, self.target)

class Model(object):
    def __init__(
            self,
            nn_module = None,
            init_weights = True,
            lr = 0.001,
            # criterion_fn = MyLoss, 
            retrain = False,
            nn_kwargs={},
            gpu_ids = -1,
            in_dim = 1,
            out_dim = 1,
            # patch_size = None
    ):
        self.nn_module = nn_module
        self.nn_kwargs = nn_kwargs
        self.init_weights = init_weights
        self.lr = lr
        self.retrain = retrain
        self.count_iter = 0
        self.gpu_ids = [gpu_ids] if isinstance(gpu_ids, int) else gpu_ids
        self.device = "cuda"
        self.bce = torch.nn.BCELoss()
        self.mse = torch.nn.MSELoss()
        self.l1 = torch.nn.L1Loss()
        self._init_model(in_dim, out_dim)


    def _init_model(self, in_dim, out_dim):

        self.net = ResUnet(in_ch = in_dim, output_ch = out_dim)
        # self.net = Unet(dim = patch_size, out_dim = out_dim, channels = in_dim)
        self.net.apply(_weights_init)

        # self.opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, betas=(0.9, 0.999))
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, betas=(0.9, 0.999))
        # self.opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.net.parameters()), lr=self.lr, betas=(0.9, 0.999))
        # self.net = nn.DataParallel(self.net)
        self.net.to(self.device)

        self.mse = torch.nn.MSELoss().to(self.device)
        self.l1 = torch.nn.L1Loss().to(self.device)


    def __str__(self):
        out_str = '{:s} | {:s} | iter: {:d}'.format(
            self.nn_module,
            str(self.nn_kwargs),
            self.count_iter,
        )
        return out_str

    def get_state(self):
        return dict(
            nn_module = self.nn_module,
            nn_kwargs = self.nn_kwargs,
            nn_net_state = self.net.state_dict(),
            optim_state =  self.opt.state_dict(),
            count_iter = self.count_iter,
        )


    def save_state(self, path_save):
        # curr_gpu_ids = self.gpu_ids
        # dirname = os.path.dirname(path_save)
        # if not os.path.exists(dirname):
        #     os.makedirs(dirname)
        # self.to_gpu(-1)
        torch.save(self.get_state(), path_save)
        # self.to_gpu(curr_gpu_ids)

    def load_state(self, path_load, in_dim, out_dim, gpu_ids=-1):
        state_dict = torch.load(path_load)
        self.nn_module = state_dict['nn_module']
        
        # D_state_dict(state_dict['nn_D_state'])
        # self.nn_kwargs = state_dict.get('nn_kwargs', {})
        self._init_model(in_dim, out_dim)

        self.opt.load_state_dict(state_dict['optim_state'])
        self.net.load_state_dict(state_dict['nn_net_state'])
        self.count_iter = state_dict['count_iter']
        
        # self.to_gpu(gpu_ids)
        return state_dict

    def load_state_dict(self, state_dict, in_dim, out_dim, gpu_ids=-1):
        self.nn_module = state_dict['nn_module']
        
        # D_state_dict(state_dict['nn_D_state'])
        # self.nn_kwargs = state_dict.get('nn_kwargs', {})
        self._init_model(in_dim, out_dim)

        self.opt.load_state_dict(state_dict['optim_state'])
        self.net.load_state_dict(state_dict['nn_net_state'])
        self.count_iter = state_dict['count_iter']
        
        

    def do_train_iter(self, signal, target, i):
        
        x_ = Variable(signal.to("cuda"))
        y_ = Variable(target.to("cuda")) 
        
        self.net.zero_grad()
        pred = self.net(x_)
        # top2 = torch.topk(pred, 2, dim=1)[0]
        # exc_loss = self.l1(top2[:, 1], top2[:, 0])
        loss = self.mse(pred, y_)   
        # loss = self.l1(pred, y_)
        loss.backward()
        # G_optimizer.step()
        self.opt.step()


        self.count_iter += 1
        return loss.item(), r2_score(y_.detach().cpu().numpy().flatten(), pred.detach().cpu().numpy().flatten()), torch.max(y_), torch.min(y_), torch.max(pred), torch.min(pred)

    # @profile   
    def predict(self, val_input, val_output, patch_size, save_img=False, path_save=None):
         
        net = self.net
        net.eval()   
        
        patch_size = patch_size[0]
        step_size = 128  
        
        scores = []
        # scores2 = []
        losses = []
        losses2 = []

        predictions = []
        
        for ii in range(len(val_input)):
            # print(ii)
        
            input_ = val_input[ii]
            target = val_output[ii]
    
            input_ = np.expand_dims(input_, 0)  # add a new dimension
               
            pred = np.zeros_like(target)
            mask = np.zeros_like(target)
            

            for i in range((target.shape[1] - patch_size)//step_size + 2):  # for each step on length
                temp = []
                for j in range((target.shape[2] - patch_size)//step_size + 2):  # for each step on width
                    min1 = min(i*step_size + patch_size, target.shape[1])
                    min2 = min(j*step_size + patch_size, target.shape[2])
                    temp.append(input_[0, :, min1-patch_size:min1, min2-patch_size:min2])  # put all patches together 
        
                    mask[:target.shape[0], min1-patch_size:min1, min2-patch_size:min2] += np.ones((target.shape[0], patch_size, patch_size)) 
        
                
                temp = np.array(temp)
                with torch.no_grad():
                    in_ = torch.from_numpy(temp).type(torch.float32).cuda()
                    out_ = net(in_)
                    del in_
                    out_ = out_.cpu().numpy()
    
        
                k = 0
                for j in range((target.shape[2] - patch_size)//step_size + 2):
                    min1 = min(i*step_size + patch_size, target.shape[1])
                    min2 = min(j*step_size + patch_size, target.shape[2])
                    
                    pred[:, min1-patch_size:min1, min2-patch_size:min2] += out_[k]  
                    
                    k += 1
        
            pred /= mask
            # print("appending 'pred' into 'predictions' ")
            predictions.append(pred) 
            # print(len(predictions))
            # print(predictions[-1].shape)

            if save_img:
                # assert path_save is not None
                if path_save is not None:
                    img_rescale = image_normalization(pred[0]) 
                    img_rescale = Image.fromarray(img_rescale)
                    path_img = os.path.join(path_save, str(ii) + "_pred.jpeg")
                    img_rescale.save(path_img)
                    # print("prediction image "+str(ii)+"save to "+path_img)
                else:
                    print("the path to save prediction images is missing!!!")

                # path_img = os.path.join(path_save, str(ii) + "_pred.tiff")
                # imwrite(path_img, pred)
                # print("prediction image "+str(ii)+" save to "+path_img)         

        print("start to calculate statistic")
        num_valimg = len(val_input)
        num_out = val_output[0].shape[0]
        # print(num_valimg, num_out)
        for i in range(num_out): # for each predict channel
            targets_c = []
            predictions_c = []
            for j in range(num_valimg): # for each image
                tgt = val_output[j][i]
                pred = predictions[j][i]
                targets_c.append(tgt.flatten())
                predictions_c.append(pred.flatten())
            tgt = np.hstack(targets_c)
            pred = np.hstack(predictions_c)
            corr = pearsonr(pred, tgt)[0]
            mae = mean_absolute_error(pred, tgt)
            mse = mean_squared_error(pred, tgt)
            scores.append(corr)
            losses.append(mae)
            losses2.append(mse)
            del targets_c 
            del predictions_c 
            del tgt 
            del pred 
     
        scores = np.array(scores, dtype=object)
        losses = np.array(losses, dtype=object)
        losses2 = np.array(losses2, dtype=object)

        # print(scores.shape, scores[0].shape)
        self.net.train()
            
        # return np.mean(scores), scores, scores2, losses
        return losses, losses2, scores

   
                

def _weights_init(m):
    classname = m.__class__.__name__
    if classname.startswith('Conv'):
        # m.weight.data.normal_(0.0, 0.02)
        nn.init.kaiming_normal_(m.weight)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0) 
        

def image_normalization(img):
    _min = img.min()
    _max = img.max()
    img_norm = (img - _min) * 255.0 /(_max - _min)
    return np.uint8(img_norm)


