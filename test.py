import numpy as np
import torch.utils.data as Data
import torch
from util.visualizer import Visualizer
import os
import hourglass
import cv2
from perceptual.filterbank import Steerable
import PIL.Image as Im
import matplotlib.pyplot as plt
from util.util import mkdir
def norm_localmotion(array):
    array = array/np.pi
    return array
def norm_heatmap(array):
    return array*2-1
def numpy2tensor(x,part=1000):
    x = x[0:part]
    x = torch.from_numpy(x)
    x = x.float()
    return x
cap = cv2.VideoCapture('data/finger.mp4')

def get_two_frame():

    ret1, frame1 = cap.read()
    ret2, frame2 = cap.read()

    return frame1,frame2

def get_location(f1,f2):
    f1 = Im.fromarray(f1).convert('L')
    f2 = Im.fromarray(f2).convert('L')
    f1 = f1.resize((256,256))
    f2 = f2.resize((256,256))
    f1 = np.array(f1)
    f2 = np.array(f2)
    csp = Steerable(2 + 1, 1)
    f1_csp = csp.buildSCFpyr(f1)
    f2_csp = csp.buildSCFpyr(f2)

    a = ((f2_csp[1][0].real) ** 2 + (f2_csp[1][0].imag) ** 2) ** 0.5
    f1phase = np.arctan2(f1_csp[1][0].imag, f1_csp[1][0].real)
    f2phase = np.arctan2(f2_csp[1][0].imag, f2_csp[1][0].real)



    return a**2*(f2phase-f1phase),f1phase,f2phase



def save_pic(array,path):
    array = (array-np.min(array))/(np.max(array)-np.min(array))
    array = array*255
    array = array.astype(np.uint8)
    pic = Im.fromarray(array)
    pic.save(path)



def start():
    model = hourglass.my_hg()

    model.load_state_dict(torch.load('save_model/model42'))
    for i in range(0,1000):
        frame1,frame2 = get_two_frame()
        location,f1p,f2p =get_location(frame1,frame2)

        pic1 = Im.fromarray(frame1)
        pic2 = Im.fromarray(frame2)
        pic1 = pic1.resize((256,256)).convert('L')
        pic2 = pic2.resize((256, 256)).convert('L')
        # pic1.show()
        # pic2.show()

        data = location[np.newaxis,np.newaxis,:,:]

        data = torch.from_numpy(data)
        data = data.float()
        data = data.cuda()

        out = model.forward(data)

        heatmap = out[0][0]



        def max_min_norm(array):
            return (array - torch.min(array) + 10 ** (-100) / (
                    torch.max(array) - torch.min(array)) + 10 ** (-100))



        heatmap = max_min_norm(heatmap)
        heat = heatmap.cpu().float().detach().numpy()
        cmap = plt.get_cmap('jet')
        rgba_img = cmap(heat) * 255
        rgba_img = rgba_img.astype(np.uint8)
        pic = Im.fromarray(rgba_img, "RGBA")
        pic = pic.convert('RGB')

        # pic.show()
        path = 'result/finger/%d'%i
        mkdir(path)
        pic1.convert('RGB')
        pic1.save(path+'/pic1.png')
        save_pic(f1p,path+'/pic1_phase.png')
        pic2.convert('RGB')
        pic2.save(path + '/pic2.png')
        save_pic(f2p, path + '/pic2_phase.png')
        pic.save(path + '/heatmap.png')

        # print('')













if __name__ == '__main__':
    start()
    pass

    import spacy









