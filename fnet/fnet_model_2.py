import os
import torch
import torch.nn as nn
import importlib
import pdb
from torch.autograd import Variable
from UNext import R2U_Net
from vit import ViT
import numpy as np
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from ResNet import ResUnet
from joint_model2 import Discriminator


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
            out_dim = 1
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
        self._init_model(out_dim)


    def _init_model(self, out_dim):
        self.G = ResUnet(in_ch = 10, output_ch = out_dim)
        self.D = Discriminator(in_ch = 29)
        
        self.G_opt = torch.optim.AdamW(self.G.parameters(), lr=self.lr, betas=(0.9, 0.999))
        self.D_opt = torch.optim.AdamW(self.D.parameters(), lr=self.lr, betas=(0.9, 0.999))    
        self.G.to(self.device)
        self.D.to(self.device)

        self.scalar = torch.cuda.amp.GradScaler()

        self.bce = torch.nn.BCELoss().to(self.device)
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
            nn_G_state = self.G.state_dict(),
            nn_D_state = self.D.state_dict(),
            G_optimizer_state = self.G_opt.state_dict(),
            D_optimizer_state = self.D_opt.state_dict(),
            count_iter = self.count_iter,
        )


    def save_state(self, path_save):
        torch.save(self.get_state(), path_save)


    def load_state(self, path_load, gpu_ids=-1):
        state_dict = torch.load(path_load)

        self._init_model()

        self.G_opt.load_state_dict(state_dict['G_optimizer_state'])
        self.D_opt.load_state_dict(state_dict['D_optimizer_state'])
        self.G.load_state_dict(state_dict['nn_G_state'])
        self.D.load_state_dict(state_dict['nn_D_state'])
        self.count_iter = state_dict['count_iter']

        
        # self.to_gpu(gpu_ids)

    def do_train_iter(self, signal, target, i):
        # self.net.train()
        # signal = torch.tensor(signal, dtype=torch.float32, device=self.device)
        # target = torch.tensor(target, dtype=torch.float32, device=self.device)
        
        if i // 500 < 5:
            a = 0.02 * (5 - i // 500)
        else:
            a = 0.0
        x_ = Variable(signal.cuda())
        y_ = Variable(target.cuda()) 


        # Train discriminator with real data
        D_real_decision = self.D(torch.cat([x_, y_], 1)).squeeze()

        real_ = Variable((torch.ones(D_real_decision.size()) - a * torch.abs(torch.randn(D_real_decision.size()))).cuda())
        D_real_loss = self.bce(D_real_decision, real_)

        # Train discriminator with fake data
        gen_image = self.G(x_)
        D_fake_decision = self.D(torch.cat([x_, gen_image], 1)).squeeze()
        fake_ = Variable((torch.zeros(D_real_decision.size()) + a * torch.abs(torch.randn(D_real_decision.size()))).cuda())

        D_fake_loss = self.bce(D_fake_decision, fake_)

        # Back propagation
        D_loss = (D_real_loss + D_fake_loss) * 0.5
        self.D.zero_grad()
        self.scalar.scale(D_loss).backward()
        # D_optimizer.step()
        self.scalar.step(self.D_opt)

        # Train generator
        gen_image = self.G(x_)

        real_ = Variable(torch.ones(D_real_decision.size()).cuda())
        D_fake_decision = self.D(torch.cat([x_, gen_image], 1)).squeeze()
        G_fake_loss = self.bce(D_fake_decision, real_)

        l1_loss = self.mse(gen_image, y_)  

        G_loss = 0.2 * G_fake_loss + l1_loss 
        self.G.zero_grad()
        self.scalar.scale(G_loss).backward()
        # G_optimizer.step()
        self.scalar.step(self.G_opt)

        self.scalar.update()

        self.count_iter += 1
        return D_loss.item(), G_loss.item(), r2_score(y_.detach().cpu().numpy().flatten(), gen_image.detach().cpu().numpy().flatten()), torch.max(gen_image).item(), torch.min(gen_image).item(), torch.max(y_).item(), torch.min(y_).item()
        
    def predict(self, signal):
        signal = torch.tensor(signal, dtype=torch.float32, device=self.device)
        if len(self.gpu_ids) > 1:
            module = torch.nn.DataParallel(
                self.net,
                device_ids = self.gpu_ids,
            )
        else:
            module = self.net
        module.eval()
        with torch.no_grad():
            prediction = module(signal).cpu()
        return prediction

def _weights_init(m):
    classname = m.__class__.__name__
    if classname.startswith('Conv'):
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0) 

