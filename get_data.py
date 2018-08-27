import numpy as np
import util.util as tool
import os
import PIL.Image as Im
from perceptual.filterbank import Steerable
limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10],
           [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17],
           [1, 16], [16, 18]]

datapath = r'D:\database\action1-person1-white'


save_path = 'data/a1p1'
tool.mkdir(save_path)
def dir_file_numb(path):
    for root, dirs, files in os.walk(path):
        # print(root) #当前目录路径
        # print(dirs) #当前路径下所有子目录
        # print(files) #当前路径下所有非目录子文件
        return len(files)

def heatmap2limbs(the_heat_map):
    pose = tool.hmp2pose_by_numpy(the_heat_map)
    limbs = tool.pose2limb(pose)
    return  limbs


# def reset_heatmap(heatmap, list):
#     ha
def get_gray_heatmap():
    all_deal_heatmap = []
    all_gray = []
    for i in range(1,dir_file_numb(datapath+'/heatmap')):
        thisimage = Im.open(datapath+'/frame/%d.png'%i)
        the_heat_map = np.load(datapath+'/heatmap/%d.npy'%i)
        last_heat = np.load(datapath+'/heatmap/%d.npy'%(i-1))
        limbs = heatmap2limbs(the_heat_map)
        last_limbs = heatmap2limbs(last_heat)
        dislist = tool.distance_limb(limbs,last_limbs)
        # print(len(limbs))
        # limbs  lastlimbs

        joint = []

        for j in range(0,len(dislist)-1):
            if dislist[j]==None:
                for p in limbSeq[j]:
                    if p not in joint:
                        joint.append(p)
            elif dislist[j]>0:
                for p in limbSeq[j]:
                    if p not in joint:
                        joint.append(p)

        the_heat_map =the_heat_map.transpose((2,0,1))
        deal_heat = the_heat_map[0]-the_heat_map[0]

        # print(joint)
        for jo in joint:
            deal_heat += the_heat_map[jo-1]
        if len(joint)>0:
            deal_heat = deal_heat/len(joint)
        else:
            print(i)
        gray_image = thisimage.convert('L')
        gray_image_numpy = np.array(gray_image)
        # gray_image.show()
        np.save(save_path+'/%d_gray.npy'%i,gray_image_numpy)
        thisimage.save(save_path+'/%d.png'%i)
        np.save(save_path+'/%d_deal_heat.npy'%i,deal_heat)
        # print(deal_heat.shape)
        all_deal_heatmap.append(deal_heat)
        all_gray.append(gray_image_numpy)

        #
        # count = 0
        # for dis in dislist:
        #     if dis==0:
        #         count+=1
        # if (count>1):
        #     print(i,count)



        #
        # for j in limbs:
        #     if len(j)<2:
        #         print(i)
        # print(dislist)
        # print(limbs)
        if i%100==0:
            # for j in limbs:
            #     print(len(j))
            print(i)
        # if len(limbs)!=12:
        #     print(i,limbs)
        # print('')


    all_deal_heatmap = np.array(all_deal_heatmap)
    all_gray = np.array(all_gray)

    print(all_deal_heatmap[1:].shape)
    print(all_gray[1:].shape)
    np.save(save_path+'all_deal_heat.npy',all_deal_heatmap[1:])
    np.save(save_path+'all_gray.npy',all_gray[1:])
def get_frame(i):
    return np.load('data/a1p1/%d_deal_heat.npy'%i)

def get_local_motion():
    AandPandlocalmotion = []
    csp = Steerable(2 + 1, 1)
    lenlen = dir_file_numb('data/a1p1')
    # lenlen = 10
    for i in range(2,1518 ):
        thiscsp = dict()
        thiscsp['A'] = []
        thiscsp['P'] = []
        thiscsp['L'] = []
        pic = get_frame(i)
        pic_csp = csp.buildSCFpyr(pic)
        basiccsp = csp.buildSCFpyr(get_frame(i-1))
        for pyrmaid in range(1, 1 + 1):
            nbandsA = []
            nbandsP = []
            nbandsL = []
            for nband in range(0, 1):
                a = ((pic_csp[pyrmaid][nband].real) ** 2 + (pic_csp[pyrmaid][nband].imag) ** 2) ** 0.5

                phase = np.arctan2(pic_csp[pyrmaid][nband].imag, pic_csp[pyrmaid][nband].real)
                basicphase = np.arctan2(basiccsp[pyrmaid][nband].imag, basiccsp[pyrmaid][nband].real)
                p = phase - basicphase

                nbandsA.append(a)
                nbandsP.append(p)
                lllllllll = a ** 2 * p
                # print(lllllllll)
                nbandsL.append(lllllllll)
            nbandsP = np.array(nbandsP)
            nbandsL = np.array(nbandsL)
            nbandsA = np.array(nbandsA)

            thiscsp['A'].append(nbandsA)
            thiscsp['P'].append(nbandsP)
            thiscsp['L'].append(nbandsL)
        AandPandlocalmotion.append(thiscsp)
    return AandPandlocalmotion


def change_demension(APL):
    allbands_local = []
    a = []
    p = []
    for py in range(0, 1):
        local = []
        la = []
        lp = []
        for i in range(len(APL)):
            local.append(APL[i]['L'][py])
            la.append(APL[i]['A'][py])
            lp.append(APL[i]['P'][py])
        allbands_local.append(np.array(local))
        a.append(np.array(la))
        p.append(np.array(lp))
    return allbands_local, a, p


if __name__ == '__main__':

    # get_gray_heatmap()
    APL = get_local_motion()
    allbands_local,_,_ = change_demension(APL)
    allbands_local =allbands_local[0]
    allbands_local = allbands_local[:,0,:,:]
    print(allbands_local.shape)
    np.save('data/a1p1_local',allbands_local)
    local = np.load('data/a1p1_local.npy')
    heat = np.load('data/a1p1all_deal_heat.npy')
    print(np.max(heat))
    print(' ')

