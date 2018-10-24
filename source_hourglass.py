import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import torch.optim as optim
import util.util as util
# from .preresnet import BasicBlock, Bottleneck


__all__ = ['HourglassNet', 'hg']


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=True)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 2, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class Hourglass(nn.Module):
    def __init__(self, block, num_blocks, planes, depth):
        super(Hourglass, self).__init__()
        self.depth = depth
        self.block = block
        self.upsample = nn.Upsample(scale_factor=2)
        self.hg = self._make_hour_glass(block, num_blocks, planes, depth)

    def _make_residual(self, block, num_blocks, planes):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(planes * block.expansion, planes))
        return nn.Sequential(*layers)

    def _make_hour_glass(self, block, num_blocks, planes, depth):
        hg = []
        for i in range(depth):
            res = []
            for j in range(3):
                res.append(self._make_residual(block, num_blocks, planes))
            if i == 0:
                res.append(self._make_residual(block, num_blocks, planes))
            hg.append(nn.ModuleList(res))
        return nn.ModuleList(hg)

    def _hour_glass_forward(self, n, x):
        up1 = self.hg[n - 1][0](x)
        low1 = F.max_pool2d(x, 2, stride=2)
        low1 = self.hg[n - 1][1](low1)

        if n > 1:
            low2 = self._hour_glass_forward(n - 1, low1)
        else:
            low2 = self.hg[n - 1][3](low1)
        low3 = self.hg[n - 1][2](low2)
        up2 = self.upsample(low3)
        out = up1 + up2
        return out

    def forward(self, x):
        return self._hour_glass_forward(self.depth, x)


class HourglassNet(nn.Module):
    '''Hourglass model from Newell et al ECCV 2016'''

    def __init__(self, block,input_channel, num_stacks=2, num_blocks=4, num_classes=1):
        super(HourglassNet, self).__init__()

        self.inplanes = 64
        self.num_feats = 128
        self.num_stacks = num_stacks
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(input_channel, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=True)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_residual(block, self.inplanes, 1)
        self.layer2 = self._make_residual(block, self.inplanes, 1)
        self.layer3 = self._make_residual(block, self.num_feats, 1)
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.upsamping = nn.Upsample(scale_factor=4)
        # build hourglass modules
        ch = self.num_feats * block.expansion
        hg, res, fc, score, fc_, score_ = [], [], [], [], [], []
        for i in range(num_stacks):
            hg.append(Hourglass(block, num_blocks, self.num_feats, 4))
            res.append(self._make_residual(block, self.num_feats, num_blocks))
            fc.append(self._make_fc(ch, ch))
            score.append(nn.Conv2d(ch, num_classes, kernel_size=1, bias=True))
            if i < num_stacks - 1:
                fc_.append(nn.Conv2d(ch, ch, kernel_size=1, bias=True))
                score_.append(nn.Conv2d(num_classes, ch, kernel_size=1, bias=True))
        self.hg = nn.ModuleList(hg)
        self.res = nn.ModuleList(res)
        self.fc = nn.ModuleList(fc)
        self.score = nn.ModuleList(score)
        self.fc_ = nn.ModuleList(fc_)
        self.score_ = nn.ModuleList(score_)

        self.optimizer = optim.Adam(self.parameters())
        self.loss_fn = nn.MSELoss()
        self.cuda()
    def _make_residual(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=True),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_fc(self, inplanes, outplanes):
        bn = nn.BatchNorm2d(inplanes)
        conv = nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=True)
        return nn.Sequential(
            conv,
            bn,
            self.relu,
        )

    def forward(self, x):
        out = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.maxpool(x)
        x = self.layer2(x)
        x = self.layer3(x)

        for i in range(self.num_stacks):
            y = self.hg[i](x)
            y = self.res[i](y)
            y = self.fc[i](y)
            score = self.score[i](y)
            out.append(score)
            if i < self.num_stacks - 1:
                fc_ = self.fc_[i](y)
                score_ = self.score_[i](score)
                x = x + fc_ + score_

        for i in range(len(out)):
            out[i] = self.upsamping(out[i])
        return out

    def get_current_loss(self):

        loss = OrderedDict()
        loss['loss'] = float(getattr(self, 'loss'))

        return [loss]

    def test(self, data,heatmap):

        self.eval()
        k = self(data)
        loss = self.loss_fn(k[-1], heatmap)
        return loss

    def fit(self, data, heatmap, frame):
        self.train()
        data = data.cuda()
        heatmap = heatmap.cuda()
        self.input = data
        self.heatmap = heatmap
        self.image = frame
        self.result = self(data)

        loss = self.loss_fn(self.result[-1], heatmap)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.loss = loss

        return loss




    def get_current_visuals(self):
        import numpy as np
        import torch
        import cv2


        visual_ret = OrderedDict()

        this_pic = self.image.cpu().detach().numpy()
        this_pic = this_pic[0].copy()
        # this_pic = this_pic.transpose((1,2,0))
        # this_pic = (this_pic+1)/2*255
        this_pic = this_pic.astype(np.uint8)
        this_pic = np.array(this_pic)
        import PIL.Image as Image
        ppp = Image.fromarray(this_pic)
        this_pic = np.array(ppp)
        # ppp.show()
        pred = this_pic.copy()

        source_joint = util.hmp2pose(self.heatmap.detach()*2+2,  self.num_classes )
        pred_heat_map = util.hmp2pose(self.result[-1].detach()*2+2,  self.num_classes )

        for i in source_joint:
            if i!=[]:
                i = i[0]
                x= int(i[0])
                y = int(i[1])
                cv2.circle(this_pic, center=(x, y), radius=5,thickness=cv2.FILLED, color=(0, 0, 255))
        for i in pred_heat_map:
            if i != []:
                i = i[0]
                x = int(i[0])
                y = int(i[1])
                cv2.circle(pred, center=(x, y), radius=5,thickness=cv2.FILLED, color=(0, 0, 255))

        # pred = util.draw_limbs_on_image(pred_heat_map,pred)
        # this_pic = util.draw_limbs_on_image(source_joint, this_pic)
        def tran(array):
            array = array.transpose((2, 0, 1))
            array = array[np.newaxis,:,:,:]
            array = array/255*2-1
            return array

        pred = tran(pred)
        this_pic = tran(this_pic)


        visual_ret['groundtruth'] = torch.from_numpy(this_pic)
        visual_ret['prediction'] = torch.from_numpy(pred)

        dir1 = self.input[:, 0, :, :].unsqueeze(1)
        dir2 = self.input[:, 1, :, :].unsqueeze(1)

        visual_ret['local_1'] = (dir1 -torch.min(dir1))/(torch.max(dir1)-torch.min(dir1))
        visual_ret['local_2'] = (dir2- torch.min(dir2)) / (torch.max( dir2) - torch.min(dir2))
        return visual_ret


def hg(**kwargs):
    model = HourglassNet(Bottleneck, num_stacks=kwargs['num_stacks'], num_blocks=kwargs['num_blocks'],
                         num_classes=kwargs['num_classes'])
    return model


def my_hg(input_channel,joint_num):
    model = HourglassNet(Bottleneck,input_channel=input_channel,num_classes=joint_num)
    return model
