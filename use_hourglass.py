import source_hourglass
import numpy as np
import torch
import util.util as util

test = torch.Tensor(1,3,256,256).fill_(0.0)

model = source_hourglass.my_hg()

out = model.forward(test)

print(out.shape)


# def tensor2pic(array):
#     array = array.cpu().float().detach().numpy()
#     import PIL.Image as Im
#     a = np.array(array)
#     a = a[0][0]
#
#     # a = np.transpose(a,[1,2,0])
#     print(np.max(a))
#     print(np.min(a))
#
#
#     a = a*255
#     a = a.astype(np.uint8)
#     pic = Im.fromarray(a)
#
#     return pic
#
#
# joint_list = [[50,50],[20,10]]
#
# guass = util.get_gaussian_kernel(kernel_size=41,sigma=1)
#
# pic = util.joint2heatmap(joint_list,guass,100,100,20)
#
# new = tensor2pic(pic)
# #
# # for i in pic.numpy():
# #     print(i)
# a = pic.numpy()
# new.show()
# print(pic)
#
#
# print()
