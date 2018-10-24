#Discussion 1.55011271

#  from video get frame
#  get frame delta as input
#  get joint position
#
import cv2
import cdflib
from human36m.xml import xml
import xmltodict
import numpy as np
import util.util as util
import PIL.Image as Im
import matplotlib.pyplot as plt
from perceptual.filterbank import Steerable

we_need = {'0':'hip'
,'1':'rightupleg'
,'2':'rightleg'
,'3':'rightfoot'
,'6':'leftupleg'
,'7':'leftleg'
,'8':'leftfoot'
,'13':'neck'
,'14':'head'
,'17':'leftarm'
,'18':'leftforearm'
,'19':'lefthand'
,'25':'rightarm'
,'26':'rightforearm'
,'27':'righthand'}

joint_num = len(we_need)
aaa = xmltodict.parse(xml)
joints_bvh = aaa['skel_angles']['tree']['item']
# print(joints_bvh)
cap = cv2.VideoCapture('human36m/s1/Videos/Discussion 1.55011271.mp4')
cdf_file = cdflib.CDF('human36m/s1/D2_Positions/Discussion 1.55011271.cdf')
d2_position = cdf_file.varget(0).reshape(64,-1)
d2_position = d2_position.transpose((1,0))
cdf_file = cdflib.CDF('human36m/s1/D3_Angles/Discussion.cdf')
angle = cdf_file.varget(0).reshape(78,-1)
angle = angle.transpose((1,0))


def get_location(f1,f2):
    f1 = f1.convert('L')
    f2 = f2.convert('L')
    f1 = np.array(f1)
    f2 = np.array(f2)
    csp = Steerable(2 + 1, 1)
    f1_csp = csp.buildSCFpyr(f1)
    f2_csp = csp.buildSCFpyr(f2)
    a = ((f2_csp[1][0].real) ** 2 + (f2_csp[1][0].imag) ** 2) ** 0.5
    a = (a-np.min(a))/(np.max(a)-np.min(a))
    f1phase = np.arctan2(f1_csp[1][0].imag, f1_csp[1][0].real)
    f2phase = np.arctan2(f2_csp[1][0].imag, f2_csp[1][0].real)

    return a**2*(f2phase-f1phase),f1phase,f2phase


def get_two_frame():
    pass

def cal_phase_delta():
    pass

def get_joint_position():
    pass


def get_heat_map():
    pass

def frame2gray(f):
    frame_pic = Im.fromarray(f)
    frame_pic = frame_pic.convert('L')
    return np.array(frame_pic)

def draw_heat(a):

    a = np.sum(a,axis=0)

    cmap = plt.get_cmap('jet')
    rgba_img = cmap(a) * 255
    rgba_img = rgba_img.astype(np.uint8)
    pic = Im.fromarray(rgba_img, "RGBA")
    pic = pic.convert('RGB')

    return pic
def save_gray_pic(array,path):
    array = (array-np.min(array))/(np.max(array)-np.min(array))
    array = array*255
    array = array.astype(np.uint8)
    pic = Im.fromarray(array)
    pic.save(path)

def start():
    ret1, last_frame = cap.read()
    a = 1
    threshold = 10 ** (-4)
    path = 'human36m/s1/discussion/'
    frame_path = path+'frame/'
    util.mkdir(frame_path)
    width = 512
    height = 512
    frame_size = [last_frame.shape[0],last_frame.shape[1]]
    last_frame = cv2.resize(last_frame,(width,height))
    gauss_kernel = util.get_gaussian_kernel(25,3)
    all_frame = []
    all_local_motion = []
    all_heat_map = []
    all_joint_heat_map = []

    print(last_frame.shape)


    while True:
        ret1, frame = cap.read()
        if not ret1:
            break
        if a==3000:
            break;
        if a%100 == 0:
            print(a)
        frame = cv2.resize(frame, (width, height))
        this_2d_position = d2_position[a]
        last_2d_position = d2_position[a - 1]
        cv2.imwrite(frame_path+'%d.png' % a, frame)
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        all_frame.append(frame)
        this_angle = angle[a]
        last_angle = angle[a - 1]
        joints_list = []
        all_list = []
        frame_gray = frame2gray(frame)
        last_frame_gray = frame2gray(last_frame)
        local_motion,frame_phase,_ = get_location(last_frame_gray,frame_gray)

        save_gray_pic(frame_phase,frame_path+'%d_gray.png'% a)


        for i in we_need.keys():
            i  = int(i)
            this_joint_bvh = joints_bvh[i]
            rot_index = this_joint_bvh['rotInd']
            if rot_index is not None:
                rot_index = [int(q) - 1 for q in (rot_index[1:-1].split(' '))]
                last_rot = [last_angle[r] for r in rot_index]
                this_rot = [this_angle[r] for r in rot_index]
                last_rot = np.array(last_rot)
                this_rot = np.array(this_rot)
                deal = np.linalg.norm((last_rot - this_rot))
                _x = int(this_2d_position[i * 2]/frame_size[0]*width)
                _y = int(this_2d_position[i * 2 + 1]/frame_size[1]*height)

                joint = []
                joint.append(_y)
                joint.append(_x)
                if deal > threshold:
                    joints_list.append(joint)
                else:
                    joints_list.append([])
                all_list.append(joint)

                    # cv2.circle(frame, center=(_x, _y), radius=3, color=(0, 0, 255))
                # print(deal)
        # cv2.imwrite('human36m/s1/directions_mask/%d.png' % a, frame)
        heat_map = util.joint2heatmap(joints_list,joint_num,gauss_kernel,width,height,12)
        h = draw_heat(heat_map)

        all_local_motion.append(local_motion)
        all_heat_map.append(heat_map)

        joint_heat_map = util.joint2heatmap(all_list,joint_num,gauss_kernel,width,height,12)
        h_all = draw_heat(joint_heat_map)
        h_all.save(frame_path + '%d_all_heatmap.png' % a)
        all_joint_heat_map.append(joint_heat_map)
        h.save(frame_path+'%d_move_heatmap.png'% a)
        last_frame = frame
        a += 1
        if a%500==0:
            all_frame = np.array(all_frame)
            all_local_motion = np.array(all_local_motion)
            all_joint_heat_map = np.array(all_joint_heat_map)
            all_heat_map = np.array(all_heat_map)
            print(all_frame.shape)
            print(all_local_motion.shape)
            print(all_heat_map.shape)
            print(all_joint_heat_map.shape)
            np.save(path+'frame%d.npy'%a,all_frame)
            np.save(path + 'local_motion%d.npy'%a, all_local_motion)
            np.save(path + 'move_heat_map%d.npy'%a, all_heat_map)
            np.save(path + 'all_joint_heat_map%d.npy'%a, all_joint_heat_map)
            all_frame = []
            all_local_motion = []
            all_joint_heat_map = []
            all_heat_map = []

    pass
def look():
    num = np.load("human36m/s1/discussion/frame500.npy")


    a = num[0]
    pic = Im.fromarray(a)

    pic.show()

def heatmap2guass():

    heatmap = np.load('data/mingyi/heat.npy')
    gauss_kernel = util.get_gaussian_kernel(25, 3)
    print(heatmap.shape)
    new = []
    a = 0
    for heat in heatmap:
        a+=1
        if a%100==0:
            print(a)
        joints_list = util.hmp2pose_by_numpy(heat,18)
        heat_map = util.joint2heatmap(joints_list, 18, gauss_kernel.cuda(), 256, 256, 12)
        new.append(heat_map)

    new = np.array(new)

    print(new.shape)

    np.save('data/mingyi/clear_heatmap.npy',new)




import json
def deal_data_form_mixamo():
    import cv2
    cv2.paste

    data_path = 'data/mixamo'

    _,dirs,__  = util.walk_dir(data_path)
    all_frame = []
    all_heat = []
    all_loc = []
    gauss_kernel = util.get_gaussian_kernel(25, 3)
    for dir_data in dirs:
        frame_path = data_path+'/'+dir_data +'/frame'
        _,_,frame_list = util.walk_dir(frame_path)
        joint_path = data_path+'/'+dir_data+'/joint'

        _,_,joint_list =  util.walk_dir(joint_path)
        print(frame_path)
        last_frame = Im.open(frame_path+'/'+frame_list[0])
        for i in range(1,len(frame_list)):
            frame = Im.open(frame_path+'/'+frame_list[i])
            joint_dict = json.load(open(joint_path+'/'+joint_list[i]))
            joint_position = joint_dict[joint_dict['move']]
            local,_,_ = get_location(frame,last_frame)
            heat_map = util.joint2heatmap([joint_position], 1, gauss_kernel.cuda(), 512, 512, 12)
            all_heat.append(heat_map)
            all_frame.append(np.array(frame))
            all_loc.append(local)


    all_heat = np.array(all_heat)
    all_loc = np.array(all_loc)
    all_frame = np.array(all_frame)

    print(all_loc.shape)
    print(all_heat.shape)
    print(all_frame.shape)

    np.save('data/mixamo/frame.npy',all_frame)
    np.save('data/mixamo/heat.npy',all_heat)
    np.save('data/mixamo/loc.npy',all_loc)





            # print()
            # print(joint_list[i])




if __name__ == '__main__':
    # look()
    # start()

    # heatmap2guass()

    deal_data_form_mixamo()
    pass




