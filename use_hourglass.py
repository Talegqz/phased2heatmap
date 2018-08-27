import hourglass
import numpy as np
import torch


test = np.arange(0,10*3*256*256).reshape(10,3,256,256)*1.0

test = torch.from_numpy(test)
test = test.float()

model = hourglass.my_hg()

out = model.forward(test)

print(out[0].shape)



