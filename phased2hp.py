import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.optim as optim
from collections import OrderedDict


class Encoder(nn.Module):
    def __init__(self, input_channel, h_size, c_size, m_size, ngf,frames_num):
        super(Encoder, self).__init__()
        self.Encoder_net = nn.Sequential(*
                                         [nn.Conv2d(input_channel, ngf, kernel_size=4, stride=2, padding=1),
                                          nn.BatchNorm2d(ngf),
                                          nn.ReLU(True),

                                          nn.Conv2d(ngf, ngf * 2, kernel_size=4, stride=2, padding=1),
                                          nn.BatchNorm2d(ngf * 2),
                                          nn.ReLU(True),

                                          nn.Conv2d(ngf * 2, ngf * 4, kernel_size=4, stride=2, padding=1),
                                          nn.BatchNorm2d(ngf * 4),
                                          nn.ReLU(True),

                                          nn.Conv2d(ngf * 4, ngf * 8, kernel_size=4, stride=2, padding=1),
                                          nn.BatchNorm2d(ngf * 8),
                                          nn.ReLU(True),

                                          nn.Conv2d(ngf * 8, h_size, kernel_size=4, stride=2, padding=1),
                                          nn.BatchNorm2d(h_size),
                                          nn.Tanh()
                                          ])


        self.h_size =h_size*2*2

        self.mu_c_layer = nn.Linear(self.h_size, c_size)
        # self.log_sigma_c_layer = nn.Linear(self.h_size, c_size)
        self.mu_m_layer = nn.Linear(self.h_size, m_size*frames_num)
        # self.log_sigma_m_layer = nn.Linear(self.h_size, m_size)

    def forward(self, input):
        out = self.Encoder_net(input)

        out = out.view(-1,self.h_size)
        mu_c = self.mu_c_layer(out)
        # log_sigma_c = self.log_sigma_c_layer(out)
        mu_m = self.mu_m_layer(out)
        # log_sigma_m = self.log_sigma_m_layer(out)

        return mu_c, mu_m


class Decoder(nn.Module):
    def __init__(self, input_channel, ngf, output_channel):
        super(Decoder, self).__init__()
        self.Decoder_net = nn.Sequential(*[

            nn.ConvTranspose2d(input_channel, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, output_channel, 4, 2, 1, bias=False),
            nn.Tanh()])

    def forward(self, input):
        out = self.Decoder_net(input)
        return out


class MoCo_Vae(nn.Module):
    def __init__(self, input_channel, h_size, c_size, m_size, ngf, frames_num, batch_size,use_cuda = True):

        super(MoCo_Vae, self).__init__()
        self.batch_size = batch_size
        self.Zcm_size = c_size + m_size
        self.m_size = m_size
        self.c_size = c_size
        self.frames_num = frames_num
        self.Encoder = Encoder(input_channel * frames_num, h_size, c_size, m_size, ngf,frames_num)

        self.recurrent = nn.GRUCell(m_size, m_size)

        self.Decoder = Decoder(c_size + m_size, ngf, input_channel)

        self.l1_loss = nn.L1Loss()
        self.Mseloss = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=0.00001)

        self.use_cuda = use_cuda
        if use_cuda:
            self.T = torch.cuda
            self.cuda()

    def sampling(self, mu, log_sigma):
        epsilon = torch.from_numpy(np.random.normal(0, 1, size=log_sigma.size())).float()

        epsilon = Variable(epsilon, requires_grad=False)
        if self.use_cuda:
            epsilon = epsilon.cuda()
        return mu + torch.exp(log_sigma / 2.) * epsilon

    def forward_with_given(self,Z_m,Z_c):

        #  Z_c shape (batch_size, len of Z_content,1,1)
        out = []
        # print("this z_m")
        for z_m in Z_m:
            z_m = z_m.unsqueeze(2).unsqueeze(3)
            # print(z_m)
            # print(z_m.shape)
            # print(z_m)
            #  Z_m shape (batch_size, len of Z_motion,1,1)

            z = torch.cat((z_m, Z_c), dim=1)

            #  z shape (batch_size, len of Z_conten+Z_motion  (Zcm_size))
            out.append(self.Decoder(z))


        return  out


    def forward(self, input1,input2):



        self.mu_c1, self.mu_m1, = self.Encoder(input1)

        Z_c1 =  self.mu_c1

        h1 = self.get_gru_initial_state()
        # h shape (batch_size, len of Z_motion
        Z_m1 = []
        for i in range(self.frames_num):
            z_m1 = self.mu_m1[:,self.m_size*i:self.m_size*i+self.m_size]
            h1 = self.recurrent(z_m1,h1)
            Z_m1.append(h1)
        # Z_m shape (num of video, batch_size, len of Z_motion)
        # use for to (get batch_size, len of Z_motion)
        #  Z_c shape (batch_size, len of Z_content)

        # Z_c = Z_c.view(self.batch_size,self.c_size,1,1)
        Z_c1 = Z_c1.unsqueeze(2).unsqueeze(3)
        #  Z_c shape (batch_size, len of Z_content,1,1)
        out1 = []
        # print("this z_m")
        for z_m1 in Z_m1:
            z_m1 = z_m1.unsqueeze(2).unsqueeze(3)
            # print(z_m)
            # print(z_m.shape)
            # print(z_m)
            #  Z_m shape (batch_size, len of Z_motion,1,1)

            z1 = torch.cat((z_m1, Z_c1), dim=1)

            #  z shape (batch_size, len of Z_conten+Z_motion  (Zcm_size))
            out1.append(self.Decoder(z1))




        self.mu_c2, self.mu_m2, = self.Encoder(input2)

        Z_c2 = self.mu_c2

        h2 = self.get_gru_initial_state()
        # h shape (batch_size, len of Z_motion
        Z_m2 = []
        for i in range(self.frames_num):
            z_m2 = self.mu_m2[:, self.m_size * i:self.m_size * i + self.m_size]
            h2 = self.recurrent(z_m2, h2)
            Z_m2.append(h2)
        # Z_m shape (num of video, batch_size, len of Z_motion)
        # use for to (get batch_size, len of Z_motion)
        #  Z_c shape (batch_size, len of Z_content)

        # Z_c = Z_c.view(self.batch_size,self.c_size,1,1)
        Z_c2 = Z_c2.unsqueeze(2).unsqueeze(3)
        #  Z_c shape (batch_size, len of Z_content,1,1)
        out2 = []
        # print("this z_m")
        for z_m2 in Z_m2:
            z_m2 = z_m2.unsqueeze(2).unsqueeze(3)
            # print(z_m)
            # print(z_m.shape)
            # print(z_m)
            #  Z_m shape (batch_size, len of Z_motion,1,1)

            z2 = torch.cat((z_m2, Z_c2), dim=1)

            #  z shape (batch_size, len of Z_conten+Z_motion  (Zcm_size))
            out2.append(self.Decoder(z2))



        return out1,Z_c1,Z_m1,out2,Z_c2,Z_m2

    def compute_kl_loss(self):

        self.kl_loss_c = -0.5 * torch.mean(1.0 + self.log_sigma_c - self.mu_c ** 2. - torch.exp(self.log_sigma_c))
        self.kl_loss_m = -0.5 * torch.mean(1.0 + self.log_sigma_m - self.mu_m ** 2. - torch.exp(self.log_sigma_m))
        # kl_loss_c = torch.mean(kl_loss_c)
        # kl_loss_m = torch.mean(kl_loss_m)

        self.kl_loss = self.kl_loss_c + self.kl_loss_m
        return self.kl_loss

    def compute_image_loss(self, out,target):
        #input shape (batch, 3* frame_num,h,w)
        #out shape (frame_ num,batch, 3, h, w)

        # out_reshape = out[1]

        # for i in range(1,self.frames_num):
        #     out_reshape = torch.cat((out_reshape,out[i]),dim = 1)

        # now a shape (batch, 3* frame_num,h,w)
        image_loss = 0
        for i in range(self.frames_num):
            image_loss += self.l1_loss(out[i], target[:,i,:,:,:])


        return image_loss
    def com_mseloss_m(self,result,target):
        # target.requires_grad = False
        loss = 0
        for i in range(len(result)):
            a =target[i].clone()
            a = a.detach()
            loss += self.Mseloss(result[i],a)



        return loss

    def com_mseloss_c(self, result, target):
        a = target.clone()
        a = a.detach()

        loss = self.Mseloss(result, a)

        return loss

    def compute_loss(self,input1,input2):
        self.input1 = input1
        self.input2 = input2

        # Currently, we dont need Z_c Z_m
        def change_shape(input):
            new1 = []
            for i in range(self.frames_num):
                new1.append(input[:, i, :, :, :])
            return torch.cat((new1), dim=1)


        input1 = change_shape(input1)
        input2 = change_shape(input2)


        self.out1,self.Z_c1,self.Z_m1,self.out2,self.Z_c2,self.Z_m2 =  self(input1,input2)

        self.loss_im1 = self.compute_image_loss(self.out1,self.input1)*100

        self.loss_im2 = self.compute_image_loss(self.out2, self.input2)*100

        new_input1 = self.forward_with_given(self.Z_m1,self.Z_c2)
        self.com1m2c = new_input1


        new_input1 = torch.cat(new_input1,dim = 1)
        new_input2 = self.forward_with_given(self.Z_m2, self.Z_c1)
        self.com2m1c =new_input2
        new_input2 = torch.cat(new_input2, dim=1)

        _,Zc2hat, Zm1hat, _ ,Zc1hat, Zm2hat= self(new_input1, new_input2)





        self.loss_m1 = self.com_mseloss_m(Zm1hat,self.Z_m1)
        self.loss_m2 = self.com_mseloss_m(Zm2hat,self.Z_m2)
        self.loss_c1 = self.com_mseloss_c(Zc1hat,self.Z_c1)
        self.loss_c2 = self.com_mseloss_c(Zc2hat,self.Z_c2)

        self.loss = self.loss_im2+self.loss_im1+self.loss_m1+self.loss_m2+self.loss_c1+self.loss_c2


        return  self.loss


    def fit(self,input1,input2):
        self.train()

        self.optimizer.zero_grad()

        self.loss = self.compute_loss(input1,input2)

        self.loss.backward()


        self.optimizer.step()
        pass



    def get_gru_initial_state(self):
        return Variable(self.T.FloatTensor(self.batch_size, self.m_size).normal_())

    def get_current_visuals(self):
        visual_ret = OrderedDict()


        for i in range(self.frames_num):
            visual_ret['input1_'+str(i)] = self.input1[:,i,:,:,:]

        for i in range(self.frames_num):
            visual_ret['output1_' + str(i)] = self.out1[i]

        for i in range(self.frames_num):
            visual_ret['input2_' + str(i)] = self.input2[:, i, :, :, :]

        for i in range(self.frames_num):
            visual_ret['output2_' + str(i)] = self.out2[i]

        for i in range(self.frames_num):
            visual_ret['1m2c_' + str(i)] = self.com1m2c[i]

        for i in range(self.frames_num):
            visual_ret['2m1c' + str(i)] = self.com2m1c[i]


        return visual_ret

    def get_current_loss(self):
        # loss_names = ['loss_m1', 'loss_m1', 'loss_m1','loss_m1','loss_m1']
        #
        # error_kl_loss  =  OrderedDict()
        all_loss = OrderedDict()
        loss_im1 = OrderedDict()
        loss_im2 = OrderedDict()
        loss_m2 = OrderedDict()
        loss_c1 = OrderedDict()
        loss_c2 = OrderedDict()
        loss_m1 = OrderedDict()
        # error_kl_loss['kl_loss'] = float(getattr(self,'kl_loss'))
        together = OrderedDict()

        loss_m1['loss_m1'] = float(getattr(self,'loss_m1'))
        loss_m2['loss_m2'] = float(getattr(self, 'loss_m2'))
        loss_c1['loss_c1'] = float(getattr(self, 'loss_c1'))
        loss_c2['loss_c2'] = float(getattr(self, 'loss_c2'))
        loss_im1['loss_im1'] = float(getattr(self, 'loss_im1'))
        loss_im2['loss_im2'] = float(getattr(self, 'loss_im2'))
        all_loss['loss'] = float(getattr(self,'loss'))

        together['loss_m1'] = float(getattr(self, 'loss_m1'))
        together['loss_m2'] = float(getattr(self, 'loss_m2'))
        together['loss_c1'] = float(getattr(self, 'loss_c1'))
        together['loss_c2'] = float(getattr(self, 'loss_c2'))
        together['loss_im1'] = float(getattr(self, 'loss_im1'))/100
        together['loss_im2'] = float(getattr(self, 'loss_im2'))/100








        return [loss_m1,loss_m2,loss_c1,loss_c2,loss_im1,loss_im2,all_loss,together]
