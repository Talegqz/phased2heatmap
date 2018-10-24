import numpy as np
import torch.utils.data as Data
import torch
from util.visualizer import Visualizer
import os
import source_hourglass
from util import util
from perceptual.filterbank import Steerable
import get_humam_data
from collections import OrderedDict


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


pic_set = np.load('data/mingyi/frame.npy')

heat_set = np.load('data/mingyi/clear_heatmap.npy')
heat_set = norm_heatmap(heat_set)

heat_set = heat_set.transpose((0, 1, 3, 2))
local_set = np.load('data/mingyi/local.npy')
local_set = local_set[:, np.newaxis, :, :]
local_set = np.fabs(local_set)
local_set = local_set / (2 * np.pi)
# test_pic_set = spic_set[3600:4000]
# test_local = slocal_set[3600:4000]
# test_heat = sheat_set[3600:4000]
# local_set = local_set[0:3600]
# pic_set = pic_set[0:3600]
# heat_set = heat_set[0:3600]

# all = list(zip(pic_set, heat_set, local_set))


#
# def get_test_data():
#
#
#     tpic_set = np.array(test_pic_set)
#     theat_set = np.array(test_heat)
#     tlocal_set = np.array(test_local)
#
#     # pic_set = pic_set[:,np.newaxis,:,:]
#     # print(np.max(heat_set))
#     # print(np.min(heat_set))
#
#     # local_set = norm_localmotion(local_set)
#
#     #
#     # print(np.max(local_set))
#     # print(np.min(local_set))
#
#     # heat_set = heat_set[:, np.newaxis, :, :]
#     x = numpy2tensor(tlocal_set,0,400)
#     # pic_set = (pic_set / 255)*2-1
#     z = numpy2tensor(tpic_set,0,400)
#     y = numpy2tensor(theat_set, 0,400)
#     x = x.cuda()
#     y = y.cuda()
#     print(x.size())
#     print(y.size())
#     print(z.size())
#     data = Data.TensorDataset(x, y, z)
#
#     loader = Data.DataLoader(
#         data,
#         batch_size=8,
#         shuffle=True
#     )
#     return loader
# # loader_test = get_test_data()

def get_data(start, end):
    # global all
    # spic_set, sheat_set, slocal_set = zip(*all)

    # pic_set = pic_set[:,np.newaxis,:,:]
    # print(np.max(heat_set))
    # print(np.min(heat_set))

    # local_set = norm_localmotion(local_set)

    #
    # print(np.max(local_set))
    # print(np.min(local_set))

    # heat_set = heat_set[:, np.newaxis, :, :]
    x = numpy2tensor(local_set, start, end)
    # pic_set = (pic_set / 255)*2-1
    z = numpy2tensor(pic_set, start, end)
    y = numpy2tensor(heat_set, start, end)
    x = x.cuda()
    y = y.cuda()

    data = Data.TensorDataset(x, y, z)

    loader = Data.DataLoader(
        data,
        batch_size=8,
        shuffle=True
    )

    return loader


def train():
    print_freq = 80
    plot_freq = 20
    display_freq = 80
    data_size = 4000
    # update_html_freq = 10
    # 00000000
    part = 450
    save_name = 'save_model/1018/model'
    # x = np.load('data/shapes/shape0.npy')

    batchSize = 8

    model = source_hourglass.my_hg()
    torch.nn.DataParallel(model, [0, 1])
    # model.load_state_dict(torch.load('save_model/1016/model_phaseandhg55'))

    all_epoch = 100000000000000
    visualizer = Visualizer()
    total_steps = 0

    for epoch in range(0, all_epoch):
        epoch_iter = 0
        # randomshuf()
        for start in range(0, 3600, 600):
            loader = get_data(start, start + 600)
            print(start)

            for i, data in enumerate(loader):
                epoch_iter += batchSize
                total_steps += batchSize
                model.fit(data[0], data[1], data[2])
                if total_steps % display_freq == 0:
                    # save_result = total_steps % update_html_freq == 0
                    pass
                    vispic = model.get_current_visuals()

                    visualizer.display_current_results(vispic, epoch, False)
                    # print(vispic[])

                if total_steps % plot_freq == 0:
                    pass
                    losses = model.get_current_loss()
                    # loss = OrderedDict()
                    # loss['test_loss'] = start_test(model)
                    # losses.append(loss)
                    visualizer.new_plot_current_errors(epoch, float(epoch_iter) / data_size, losses)
                if total_steps % print_freq == 0:
                    print(losses)
                    pass
            del loader
        if epoch % 3 == 0:
            torch.save(model.state_dict(), save_name + str(epoch))


import PIL.Image as Im


def test(start, end):
    import cv2
    model = source_hourglass.my_hg()
    all = np.load('data/mingyi/frame.npy')
    model.load_state_dict(torch.load('save_model/1016/model_phaseandhg35'))
    for index in range(start, end):
        # frame1 = Im.open('test_data/0042/00000%d.jpg'%index).convert('L')
        # frame2 = Im.open('test_data/0042/00000%d.jpg' % (index+1)).convert('L')
        # pic  =   Im.open('test_data/0042/00000%d.jpg' % (index+1))

        # frame1 = Im.open('human36m/s1/discussion/frame/%d.png' % index).convert('L')
        # frame2 = Im.open('human36m/s1/discussion/frame/%d.png' % (index + 1)).convert('L')
        # pic =    Im.open('human36m/s1/discussion/frame/%d.png' % (index + 1))
        #
        frame1 = all[index]
        frame1 = np.array(Im.fromarray(frame1).convert('L'))
        frame2 = all[index + 1]
        frame2 = np.array(Im.fromarray(frame2).convert('L'))

        pic = all[index + 1]

        pic = np.array(pic)
        frame1 = np.array(frame1)
        frame2 = np.array(frame2)

        frame1 = cv2.resize(frame1, (256, 256))
        frame2 = cv2.resize(frame2, (256, 256))
        pic = cv2.resize(pic, (256, 256))

        local_all = get_location_one(frame1, frame2)

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
        local = local_all[0]
        local = local[np.newaxis, np.newaxis, :, :]
        data = torch.from_numpy(local)
        data = data.float()

        data = data.cuda()

        out = model.forward(data)

        heatmap = out[1].detach()
        heatmap = heatmap / 2 + 0.5
        joints = util.hmp2pose(heatmap, 18)

        pic = util.draw_limbs_on_image(joints, pic)

        pic = Im.fromarray(pic)

        pic.save('result/mingyi/%d_index.png' % index)
        print('')
        pass


# def start_test(model):
#     loss = 0
#     for i, data in enumerate(loader_test):
#         loss += model.test(data[0],data[1])
#
#     return loss/(i+1)
#
#
#     pass


def randomshuf():
    global all

    np.random.shuffle(all)

    # pic_set,heat_set,local_set = zip(*all)
    # # print(pic_set.shape)
    # np.save('data/mingyi/shuffle_frame.npy',pic_set)
    # np.save('data/mingyi/shuffle_heat.npy', heat_set)
    # np.save('data/mingyi/shuffle_local.npy', local_set)


if __name__ == '__main__':
    # get_humam_data.start()
    # randomshuf()
    # test(400,500)
    train()
    # test(4000,4100)
