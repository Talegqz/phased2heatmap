import cdflib
import cv2
import numpy as np
import os

img = np.zeros(shape=(1000, 1000, 3))
img.fill(255)
# cdf_file = cdflib.CDF(r'D:\gqzinstalll\gqzdownload\human36m\Poses_D3_Positions_S1\S1\MyPoseFeatures\D3_Positions\Greeting.cdf')
cdf_file = cdflib.CDF('human36m/s1/TOF/Directions.cdf')

# print(cdf_file.zVariables)
z = cdf_file.varget("NameOfVariable", startrec = 0, endrec = 150)
info = cdf_file.cdf_info()
# print(in)
# k = os.path.split('./data/Directions.2d.cdf')
# print(k)
# t = os.path.splitext('./data/Directions.2d.cdf')
# print(x)
# print(t)
x= cdf_file.varget('Index')
x2= cdf_file.varget('Indicator')
print(len(x2[x2>0]))
x3= cdf_file.varget('RangeFrames')
x4= cdf_file.varget('IntensityFrames')
# x = x.reshape(1, 64, 1612)
# x_frame0 = x[0, :, 0]
print(' ')
# for i in range(32):
#     _x = int(x_frame0[2*i])
#     _y = int(x_frame0[2*i+1])
#
#     cv2.circle(img, center=(_x, _y), radius=3, color=(0,0,255))
#
# cv2.imwrite('./test.png', img)




print(x)
# y = cdf_file.attinq()
#
#
#
# print(cdf_file.cdf_info())
# # print(cdf_file.attrs)
# print('')
# #
# # for i in cdf_file:
#     print(i)