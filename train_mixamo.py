import numpy as np
import torch.utils.data as Data
import torch
from util.visualizer import Visualizer
import source_hourglass
from util import util
from perceptual.filterbank import Steerable
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def get_location(f1, f2):
    f1 = np.array(f1)
    f2 = np.array(f2)
    csp = Steerable(2 + 3, 1)
    f1_csp = csp.buildSCFpyr(f1)
    f2_csp = csp.buildSCFpyr(f2)
    local_all = []
    for i in range(1, 4):
        a = ((f2_csp[i][0].real) ** 2 + (f2_csp[i][0].imag) ** 2) ** 0.5
        a = (a - np.min(a)) / (np.max(a) - np.min(a))
        f1phase = np.arctan2(f1_csp[i][0].imag, f1_csp[i][0].real)
        f2phase = np.arctan2(f2_csp[i][0].imag, f2_csp[i][0].real)

        local_all.append(a ** 2 * (f2phase - f1phase))

    return local_all


def get_location_one(f1, f2):
    f1 = f1.convert('L')
    f2 = f2.convert('L')
    f1 = np.array(f1)
    f2 = np.array(f2)
    csp = Steerable(2 + 1, 1)
    f1_csp = csp.buildSCFpyr(f1)
    f2_csp = csp.buildSCFpyr(f2)
    a = ((f2_csp[1][0].real) ** 2 + (f2_csp[1][0].imag) ** 2) ** 0.5
    a = (a - np.min(a)) / (np.max(a) - np.min(a))
    f1phase = np.arctan2(f1_csp[1][0].imag, f1_csp[1][0].real)
    f2phase = np.arctan2(f2_csp[1][0].imag, f2_csp[1][0].real)

    return [a ** 2 * (f2phase - f1phase), f1phase, f2phase]

def get_local_with_or(f1,f2,oritation=2,layer = 1):

    csp = Steerable(2 + layer, oritation)

    f1 = f1.convert('L')
    f2 = f2.convert('L')
    f1 = np.array(f1)
    f2 = np.array(f2)
    f1_csp = csp.buildSCFpyr(f1)
    f2_csp = csp.buildSCFpyr(f2)
    all_direction = []
    for d in range(0, oritation):
        a = ((f2_csp[1][d].real) ** 2 + (f2_csp[1][d].imag) ** 2) ** 0.5
        a = (a - np.min(a)) / (np.max(a) - np.min(a))
        f1phase = np.arctan2(f1_csp[1][d].imag, f1_csp[1][d].real)
        f2phase = np.arctan2(f2_csp[1][d].imag, f2_csp[1][d].real)
        res = a ** 2 * (f2phase - f1phase)
        all_direction.append(res)

    return np.array(all_direction)


def norm_localmotion(array):
    array = array / np.pi
    return array


def norm_heatmap(array):
    return array * 2 - 1


def numpy2tensor(x1, start, end):
    x = x1[start:end]
    x = torch.from_numpy(x)
    x = x.float()
    return x
def frameset2localset(frame_set,oritation=2,layer = 1):

    csp = Steerable(2 + layer, oritation)

    local_set = []
    for i in range(1,len(frame_set)):
        f1 = frame_set[i-1]
        f2 = frame_set[i]
        f1 = Im.fromarray(f1)
        f2 = Im.fromarray(f2)

        f1 = f1.convert('L')
        f2 = f2.convert('L')
        f1 = np.array(f1)
        f2 = np.array(f2)
        f1_csp = csp.buildSCFpyr(f1)
        f2_csp = csp.buildSCFpyr(f2)
        all_direction = []
        for d in range(0,oritation):

            a = ((f2_csp[1][d].real) ** 2 + (f2_csp[1][d].imag) ** 2) ** 0.5
            a = (a - np.min(a)) / (np.max(a) - np.min(a))
            f1phase = np.arctan2(f1_csp[1][d].imag, f1_csp[1][d].real)
            f2phase = np.arctan2(f2_csp[1][d].imag, f2_csp[1][d].real)
            res = a ** 2 * (f2phase - f1phase)
            all_direction.append(res)

        local_set.append(all_direction)

    local_set = np.array(local_set)

    local_set = np.fabs(local_set)
    local_set = local_set / (2 * np.pi)

    return local_set





pic_set = np.load('data/mixamo/frame.npy')

heat_set = np.load('data/mixamo/heat.npy')
heat_set = norm_heatmap(heat_set)
heat_set = heat_set.transpose((0, 1, 3, 2))
heat_set = heat_set.transpose((0,2,3,1))


#
# pic_set = pic_set[0:10]
# heat_set = heat_set[0:10]
print(pic_set.shape)

print(heat_set.shape)
# local_set = np.load('data/mixamo/loc.npy')
# local_set = local_set[:, np.newaxis, :, :]
# local_set = np.fabs(local_set)
# local_set = local_set / (2 * np.pi)
# local_set = local_set.transpose((0,3,1,2))

from imgaug import augmenters as iaa
import imgaug as ia

def get_data(start, end,epoch):


    seq = iaa.Sequential(iaa.Affine(rotate=90*((epoch/20)%5)))

    # seq = iaa.Sequential(
    #     [
    #         # apply the following augmenters to most images
    #         # iaa.Fliplr(0.5),  # horizontally flip 50% of all images
    #         # iaa.Flipud(0.2),  # vertically flip 20% of all images
    #         # crop images by -5% to 10% of their height/width
    #         sometimes(iaa.CropAndPad(
    #             percent=(-0.05, 0.1),
    #             pad_mode=ia.ALL,
    #             pad_cval=(0, 255)
    #         )),
    #         sometimes(iaa.Affine(
    #             scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
    #             # # scale images to 80-120% of their size, individually per axis
    #             # translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},  # translate by -20 to +20 percent (per axis)
    #             rotate=(-45, 45),  # rotate by -45 to +45 degrees
    #             shear=(-16, 16),  # shear by -16 to +16 degrees
    #             order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
    #             # cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
    #             # mode=ia.ALL  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
    #         )),
    #         ],
    #     random_order=True
    # )
    seq_det = seq.to_deterministic()

    new_p = seq_det.augment_images(pic_set)
    new_h = seq_det.augment_images(heat_set)

    new_p = pic_set
    new_h = heat_set

    local_set = frameset2localset(new_p)
    print(local_set.shape)
    new_h = new_h.transpose((0,3,1,2))
    x = numpy2tensor(local_set, start, end)
    z = numpy2tensor(new_p, start+1, end+1)
    y = numpy2tensor(new_h, start+1, end+1)
    # x = x.cuda()
    # y = y.cuda()
    data = Data.TensorDataset(x, y, z)
    loader = Data.DataLoader(
        data,
        batch_size=4,
        shuffle=True
    )

    return loader


def train():
    print_freq = 80
    plot_freq = 20
    display_freq = 20
    data_size = 400

    save_name = 'save_model/mixamo/twodirection__90_no_aug_model'
    # x = np.load('data/shapes/shape0.npy')
    batchSize = 8
    model = source_hourglass.my_hg(2,1)
    # model = torch.nn.DataParallel(model, [0, 1])
    # model.load_state_dict(torch.load('save_model/mixamo/90_aug_model60'))

    all_epoch = 10000000000000
    visualizer = Visualizer()
    total_steps = 0

    for epoch in range(0, all_epoch):

        epoch_iter = 0
        if epoch%20==0:
            loader = get_data(0, 5,epoch)

        for i, data in enumerate(loader):
            epoch_iter += batchSize
            total_steps += batchSize
            model.fit(data[0], data[1], data[2])
            if total_steps % display_freq == 0:
                # save_result = total_steps % update_html_freq == 0
                pass
                vispic = model.get_current_visuals()

                visualizer.display_current_results(vispic, epoch, False)

            if total_steps % plot_freq == 0:
                pass
                losses = model.get_current_loss()
                visualizer.new_plot_current_errors(epoch, float(epoch_iter) / data_size, losses)
            if total_steps % print_freq == 0:
                print(losses)
                pass
        if epoch % 10 == 0:
            torch.save(model.state_dict(), save_name + str(epoch))

import PIL.Image as Im


def test(start, end):

    model = source_hourglass.my_hg(2,1)
    ppp = 'twodirection__90_aug_model300'
    data_p = 'data/mixamo/man_punching_left60'
    data_path = data_p+'/frame'
    save_path = data_p+'/%s'%ppp
    util.mkdir(save_path)
    model.load_state_dict(torch.load('save_model/mixamo/%s'%ppp))




    for index in range(start, end):
        frame1 = Im.open(data_path+'/%04d.png'%(index+1))
        frame2 = Im.open(data_path + '/%04d.png' % index)
        # frame1 = frame1.rotate(40)
        # frame2 = frame2.rotate(40)
        local = get_local_with_or(frame1, frame2)

        # local = local[np.newaxis,np.newaxis,:,:]
        # for local in local_all:
        #     local = ((local-np.min(local))/(np.max(local)-np.min(local)))*255
        #
        #     local = local.astype(np.uint8)
        #
        #     local_pic = Im.fromarray(local)
        #
        #     local_pic.show()
        #

        local = np.fabs(local)
        local = local / (2 * np.pi)
        for jjj in range(len(local)):
            local_pic = local[jjj]
            local_pic = (local_pic - np.min(local_pic)) / (np.max(local_pic) - np.min(local_pic))

            local_pic = local_pic * 255
            local_pic = local_pic.astype(np.uint8)
            lo_p = Im.fromarray(local_pic)
            lo_p = lo_p.convert('RGB')
            lo_p.save(save_path + '/%d_local_%d.png' % (index,jjj))

        local = local[np.newaxis, :,:, :]
        data = torch.from_numpy(local)
        data = data.float()

        data = data.cuda()

        out = model.forward(data)

        heatmap = out[1].detach()
        heatmap = heatmap / 2 + 0.5
        joints = util.hmp2pose(heatmap, 1)
        pic = np.array(frame1)
        # pic = util.draw_limbs_on_image(joints, pic)
        import cv2
        for i in joints:
            if i!=[]:
                i = i[0]
                x= int(i[0])
                y = int(i[1])
                cv2.circle(pic, center=(x, y), radius=5,thickness=cv2.FILLED, color=(0, 0, 255))

        pic = Im.fromarray(pic)

        pic.save(save_path+'/%d_index.png' % index)


        pass




if __name__ == '__main__':
    # get_humam_data.start()
    # randomshuf()
    test(1,25)
    # train()
    # test(4000,4100)

