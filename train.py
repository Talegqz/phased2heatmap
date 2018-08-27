import numpy as np
import torch.utils.data as Data
import torch
from util.visualizer import Visualizer
import os
import hourglass
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

def start():

    print_freq = 800
    plot_freq = 100
    display_freq = 100
    data_size = 4000
    # update_html_freq = 1000000000
    part = 1200
    save_name = 'save_model/model'
    # x = np.load('data/shapes/shape0.npy')

    batchSize = 1

    pic_set = np.load('data/a1p1all_gray.npy')
    heat_set = np.load('data/a1p1all_deal_heat.npy')
    local_set  = np.load('data/a1p1_local.npy')
    pic_set = pic_set[:,np.newaxis,:,:]
    local_set = local_set[:,np.newaxis,:,:]

    # print(np.max(heat_set))
    # print(np.min(heat_set))

    # local_set = norm_localmotion(local_set)

    #
    # print(np.max(local_set))
    # print(np.min(local_set))

    heat_set = heat_set[:, np.newaxis, :, :]
    heat_set = norm_heatmap(heat_set)
    x = numpy2tensor(local_set,part)
    pic_set = (pic_set / 255)*2-1
    z = numpy2tensor(pic_set,part)
    y = numpy2tensor(heat_set,part)
    x = x.cuda()
    y = y.cuda()

    data = Data.TensorDataset(x,y,z)

    loader = Data.DataLoader(
        data,
        batch_size=1,
        shuffle=True
    )

    model = hourglass.my_hg()

    # model.load_state_dict(torch.load('save_model/model6'))

    all_epoch = 100000000000000
    visualizer = Visualizer()
    total_steps = 0



    for epoch in range(all_epoch):
        epoch_iter = 0

        for i ,data in enumerate(loader):
            epoch_iter += batchSize
            total_steps += batchSize
            model.fit(data[0],data[1],data[2])



            if total_steps % display_freq == 0:
                # save_result = total_steps % update_html_freq == 0
                pass
                vispic = model.get_current_visuals()
                visualizer.display_current_results(vispic, epoch, False)
                # print(vispic[])

            if total_steps % plot_freq == 0:
                pass
                losses = model.get_current_loss()
                visualizer.new_plot_current_errors(epoch, float(epoch_iter) / data_size, losses)


            if total_steps % print_freq == 0:
                print(losses)
                pass
        if epoch % 3 == 0:
            torch.save(model.state_dict(), save_name +str(epoch))




if __name__ == '__main__':
    start()












