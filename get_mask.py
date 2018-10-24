import cv2
import cdflib
from human36m.xml import xml
import xmltodict
import numpy as np
def get_father(joints_bvh):
    father = [0]*50
    for i in range(0,32):
        this_joint_bvh = joints_bvh[i]
        children = this_joint_bvh['children']
        if children is not None:
            if children[0]=='[':
                children = children[1:-1].split(' ')

                for ch in children:
                    father[int(ch)] = i

            else:
                father[int(children)] =i
    return father
    # pass
# human36m/s1/Videos/Directions.55011271.mp4
aaa = xmltodict.parse(xml)
joints_bvh = aaa['skel_angles']['tree']['item']
father = get_father(joints_bvh)
cap = cv2.VideoCapture('human36m/s1/Videos/Directions.55011271.mp4')
cdf_file = cdflib.CDF('human36m/s1/D2_Positions/Directions.55011271.cdf')

d2_position = cdf_file.varget(0).reshape(64,-1)
d2_position = d2_position.transpose((1,0))
cdf_file = cdflib.CDF('human36m/s1/D3_Angles/Directions.cdf')
angle = cdf_file.varget(0).reshape(78,-1)
angle = angle.transpose((1,0))

a = 1


def save_pic():
    pass

ret1, last_frame = cap.read()

def get_local_motion(f1,f2):

    pass


threshold = 10**(-4)

while True:
    ret1, frame = cap.read()
    if not ret1:
        break

    local_motion = get_local_motion(last_frame,frame)

    this_2d_position = d2_position[a]
    last_2d_position = d2_position[a-1]
    cv2.imwrite('human36m/s1/directions0/%d.png'%a,frame)
    this_angle = angle[a]
    last_angle = angle[a-1]
    flag = [0 for qqq in range(32)]
    for i in range(0,32):
        this_joint_bvh = joints_bvh[i]
        rot_index = this_joint_bvh['rotInd']
        if rot_index is not None:
            rot_index = [int(q) -1 for q in (rot_index[1:-1].split(' '))]
            last_rot = [last_angle[r] for r in rot_index]
            this_rot = [this_angle[r] for r in rot_index]
            last_rot=  np.array(last_rot)
            this_rot = np.array(this_rot)
            deal = np.linalg.norm((last_rot-this_rot))
            if deal> threshold:
                flag[i] = 1
                _x = int(this_2d_position[i*2])
                _y = int(this_2d_position[i * 2+1])
                cv2.circle(frame,center=(_x,_y),radius=3,color=(0,0,255))
            # print(deal)
    cv2.imwrite('human36m/s1/directions_mask/%d.png' % a, frame)
    last_frame = frame
    a += 1
print(a)