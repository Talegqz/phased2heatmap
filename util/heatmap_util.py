# -*- coding:utf-8 -*-
# The utils for openpose files

import cv2
import math
import json
import torch
import numpy as np
import torch.nn.functional as functional

from torch.autograd import Variable


def load_json(json_path):
    with open(json_path, 'r') as f:
        json_file = json.load(f)
    f.close()
    return json_file


def resize(json_file, source_width, target_width):
    scale = target_width / source_width
    for people_index, people in enumerate(json_file['people']):
        for key in people:
            for item_index in list(filter(lambda x: x%3 != 2, range(len(people[key])))):
                json_file['people'][people_index][key][item_index] = min(json_file['people'][people_index][key][item_index]*scale, target_width-1)
    return json_file


def get_features(pose_array):
    ankles_y = pose_array[:, [3*11+1, 3*14+1]]
    ankle_max = (ankles_y.flatten()[ankles_y.flatten() > 10]).max()
    ankle_average = np.average((ankles_y.flatten()[ankles_y.flatten() > 10]))

    ankle_max_set = np.max(ankles_y[ankles_y[:, 1] > 10], axis=1)
    ankle_max_set_min = ankle_max_set[ankle_max_set > 10].min()
    ankle_min_set = ankle_max_set[(ankle_max_set-ankle_max_set_min) < 10]
    ankle_min = ankle_min_set.max()

    head_y = pose_array[:, [3*0+1]]
    middle_y = pose_array[:, [3*8+1]]
    foot_y = np.average(ankles_y, axis=1)
    height_far = 0
    height_close = 0
    for i in range(len(head_y)):
        if middle_y[i] != 0 and head_y[i] != 0:
            height_far = max(height_far, middle_y[i] - head_y[i])
        if middle_y[i] != 0 and foot_y[i] != 0:
            height_close = max(height_close, foot_y[i] - middle_y[i])

    return ankle_max, ankle_min, ankle_average, height_far, height_close


def get_global_normalize(source_jsons, target_jsons):
    source_pose = []
    target_pose = []
    for item in source_jsons:
        if len(item['people']) > 0:
            source_pose.append(item['people'][0]['pose_keypoints_2d'])
    for item in target_jsons:
        if len(item['people']) > 0:
            target_pose.append(item['people'][0]['pose_keypoints_2d'])
    source_pose = np.array(source_pose)
    target_pose = np.array(target_pose)
    source_max, source_min, source_average, source_height_far, source_height_close = get_features(source_pose)
    target_max, target_min, target_average, target_height_far, target_height_close = get_features(target_pose)

    translation = source_average - target_average

    alpha = (source_average - source_min) / (source_max - source_min)
    scale = (1-alpha)*(source_height_far/target_height_far) + alpha*(source_height_close/target_height_close)

    return translation, scale[0], target_average


# center = []
def apply_global_normalize_to_pose(pose, center, translation, scale):

    for people_index, people in enumerate(pose[0]['people']):
        for key in people:
            for item_index in list(filter(lambda x: x % 3 == 0, range(len(people[key])))):
                pose[0]['people'][people_index][key][item_index] = (pose[0]['people'][people_index][key][item_index] - center[0])*scale + center[0]
            for item_index in list(filter(lambda x: x % 3 == 1, range(len(people[key])))):
                pose[0]['people'][people_index][key][item_index] = (pose[0]['people'][people_index][key][item_index] - center[1]) * scale + center[1] + translation

    return pose


def apply_global_normalize_to_image(image, center, translation, scale):
    width = image.shape[0]
    translation = translation / scale
    image_resize = cv2.resize(image.copy(), dsize=(int(width*scale), int(width*scale)))
    translation_abs = int(abs(translation))
    image_padding = np.zeros(shape=(int(width*scale+2*translation_abs), int(width*scale), 3))
    for i in range(3):
        image_padding[:, :, i] = np.pad(image_resize[:, :, i], ((translation_abs, translation_abs), (0, 0)),  mode='constant', constant_values=0)
    if translation < 0:
        return image_padding[int(center[1]*scale-(center[1])+3*translation_abs):int(center[1]*scale+(width-center[1])+3*translation_abs), int((image_padding.shape[1]-width)/2):int((image_padding.shape[1]-width)/2 + width)]
    else:
        top = max(int(center[1]*scale-(center[1])-translation), 0)
        left = max(int((image_padding.shape[1]-width)/2), 0)
        return image_padding[top:top+width, left:left + width]

# a = np.zeros(shape=(200, 200, 3))
# a[90:110, 90:110, :] = 255

# a = cv2.imread('/mnt/dataset/motions7/dance2-person1/skeleton-original/0_rendered.png')
# cv2.imwrite('original.png', a)
# cv2.imwrite('test-0.png', apply_global_normalize_to_image(a, [100, 100], 20, 3))
# cv2.imwrite('test-1.png', apply_global_normalize_to_image(a, [100, 100], 0, 2))
# cv2.imwrite('test-2.png', apply_global_normalize_to_image(a, [100, 100], -20, 3))
# cv2.imwrite('test--10.png', apply_global_normalize_to_image(a, [800, 540], -16, 1.06))


def get_gaussian_kernel_all(Tensor, opt):
    kernel_small = get_gaussian_kernel(Tensor, kernel_size=8, sigma=2)
    kernel_large = get_gaussian_kernel(Tensor, kernel_size=80, sigma=16)
    gaussian_kernel = Tensor(1, 11*opt.person_num, 80, 80)
    for person_index in range(opt.person_num):
        for i in range(0, 5):
            gaussian_kernel[:, person_index*11 + i, 36:44, 36:44] = kernel_small
        for i in range(5, 11):
            gaussian_kernel[:, person_index*11 + i, :, :] = kernel_large
    return gaussian_kernel


def get_gaussian_kernel(Tensor, kernel_size=10, sigma=40):
    x_cord = Tensor(np.arange(kernel_size))
    x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1)

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.
    gaussian_kernel = torch.exp(-torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance))
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    return gaussian_kernel


def json2heatmap_in_batch_conv3d(json_file, gaussian_kernel_small, gaussian_kernel, Tensor, opt):
    channels = Tensor(opt.batchSize, 1, opt.input_nc, opt.loadSize_h, opt.loadSize).fill_(0)
    conv_results = []
    for people_index, people in enumerate(json_file['people'][0:2]):
        init_index = people_index * 11
        for batch_index in range(opt.batchSize):
            for face_point in (people['face_keypoints_2d'][3*i:3*i+3] for i in range(70)):
                channels[batch_index, 0, init_index + 0, int(face_point[1][batch_index]), int(face_point[0][batch_index])] = 1
            for left_hand_point in (people['hand_left_keypoints_2d'][3*i:3*i+3] for i in range(21)):
                channels[batch_index, 0, init_index + 1, int(left_hand_point[1][batch_index]), int(left_hand_point[0][batch_index])] = 1
            for right_hand_point in (people['hand_right_keypoints_2d'][3*i:3*i+3] for i in range(21)):
                channels[batch_index, 0, init_index + 2, int(right_hand_point[1][batch_index]), int(right_hand_point[0][batch_index])] = 1
            for left_foot_point in (people['pose_keypoints_2d'][3*i:3*i+3] for i in [22, 23, 24]):
                channels[batch_index, 0, init_index + 3, int(left_foot_point[1][batch_index]), int(left_foot_point[0][batch_index])] = 1
            for right_foot_point in (people['pose_keypoints_2d'][3*i:3*i+3] for i in [19, 20, 21]):
                channels[batch_index, 0, init_index + 4, int(right_foot_point[1][batch_index]), int(right_foot_point[0][batch_index])] = 1
            body_list = people['pose_keypoints_2d']
            # parts = [[0, 15, 16, 18], [0, 1, 2, 5, 8, 9, 12], [2, 3, 4], [5, 6, 7], [9, 10, 11], [12, 13, 14]]
            # for part_index, part in enumerate(parts):
            #     for joint in part:
            #         channels[batch_index, 0, 5 + part_index + init_index, int(body_list[joint * 3 + 1][batch_index]), int(body_list[joint * 3][batch_index])] = 1
            # heads = [[0, 15], [0, 16], [17, 15], [16, 18]]
            # for head_limb in heads:
            #     channels[batch_index, 0, 5, int(body_list[head_limb[0] * 3 + 1][batch_index]), int(body_list[head_limb[0] * 3][batch_index])] = 1
            #     channels[batch_index, 0, 5, int(body_list[head_limb[1] * 3 + 1][batch_index]), int(body_list[head_limb[1] * 3][batch_index])] = 1
            # limbs = [[0, 1], [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [8, 12], [9, 10], [12, 13], [10, 11], [13, 14], [2, 9], [5, 12]]
            # for index, limb in enumerate(limbs):
            #     channels[batch_index, 0, 6 + index, int(body_list[limb[0] * 3 + 1][batch_index]), int(body_list[limb[0] * 3][batch_index])] = 1
            #     channels[batch_index, 0, 6 + index, int(body_list[limb[1] * 3 + 1][batch_index]), int(body_list[limb[1] * 3][batch_index])] = 1
            # conv1 = functional.conv3d(Variable(channels[:, :, 0:6]), Variable(gaussian_kernel_small), padding=(0, 4, 4)).data
            # conv2 = functional.conv3d(Variable(channels[:, :, 6:22]), Variable(gaussian_kernel), padding=(0, 40, 40)).data
        conv_results.append(functional.conv3d(Variable(channels[:, :, init_index + 0: init_index + 6]), Variable(gaussian_kernel_small), padding=(0, 4, 4)).data)
        conv_results.append(functional.conv3d(Variable(channels[:, :, init_index + 6: init_index + 11]), Variable(gaussian_kernel), padding=(0, 40, 40)).data)
        # conv_result = torch.cat((conv1, conv2), dim=2)
        # conv_results.append(conv_result)
    conv_results = torch.cat(conv_results, dim=2)
    return conv_results[:,0,:,:,:]


def json2heatmap_in_batch_conv2d(json_file, gaussian_kernel_small, gaussian_kernel_large, Tensor, opt):
    conv_results = []
    for batch_index in range(opt.batchSize):
        conv_batch = []
        channels = Tensor(opt.input_nc, 1, opt.loadSize_h, opt.loadSize).fill_(0)
        for people_index, people in enumerate(json_file['people'][0:opt.person_num]):
            init_index = people_index * 11
            for face_point in (people['face_keypoints_2d'][3*i:3*i+3] for i in range(70)):
                if int(face_point[1][batch_index]) != 0 or int(face_point[0][batch_index]) != 0:
                    channels[init_index + 0, 0,  int(face_point[1][batch_index]), int(face_point[0][batch_index])] = 1
            for left_hand_point in (people['hand_left_keypoints_2d'][3*i:3*i+3] for i in range(21)):
                if int(left_hand_point[1][batch_index]) != 0 or int(left_hand_point[0][batch_index]) != 0:
                    channels[init_index + 1, 0, int(left_hand_point[1][batch_index]), int(left_hand_point[0][batch_index])] = 1
            for right_hand_point in (people['hand_right_keypoints_2d'][3*i:3*i+3] for i in range(21)):
                if int(right_hand_point[1][batch_index]) != 0 or int(right_hand_point[0][batch_index]) != 0:
                    channels[init_index + 2, 0, int(right_hand_point[1][batch_index]), int(right_hand_point[0][batch_index])] = 1
            for left_foot_point in (people['pose_keypoints_2d'][3*i:3*i+3] for i in [22, 23, 24]):
                if int(left_foot_point[1][batch_index]) != 0 or int(left_foot_point[0][batch_index]) != 0:
                    channels[init_index + 3, 0, int(left_foot_point[1][batch_index]), int(left_foot_point[0][batch_index])] = 1
            for right_foot_point in (people['pose_keypoints_2d'][3*i:3*i+3] for i in [19, 20, 21]):
                if int(right_foot_point[1][batch_index]) != 0 or int(right_foot_point[0][batch_index]) != 0:
                    channels[init_index + 4, 0, int(right_foot_point[1][batch_index]), int(right_foot_point[0][batch_index])] = 1
            body_list = people['pose_keypoints_2d']
            parts = [[0, 15, 16, 18], [0, 1, 2, 5, 8, 9, 12], [2, 3, 4], [5, 6, 7], [9, 10, 11], [12, 13, 14]]
            for part_index, part in enumerate(parts):
                for joint in part:
                    if int(body_list[joint * 3 + 1][batch_index]) != 0 or int(body_list[joint * 3][batch_index]) != 0:
                        channels[5 + part_index + init_index, 0, int(body_list[joint * 3 + 1][batch_index]), int(body_list[joint * 3][batch_index])] = 1
        list_split_small = []
        list_split_large = []
        for i in range(opt.person_num):
            list_split_small += [11 * i, 11 * i + 1, 11 * i + 2, 11 * i + 3, 11 * i + 4]
            list_split_large += [11 * i + 5, 11 * i + 6, 11 * i + 7, 11 * i + 8, 11 * i + 9, 11 * i + 10]
        conv1 = functional.conv2d(Variable(channels[list_split_small]), Variable(gaussian_kernel_small), padding=4).data
        conv2 = functional.conv2d(Variable(channels[list_split_large]), Variable(gaussian_kernel_large), padding=40).data
        for people_index in range(opt.person_num):
            conv_batch.append(conv1[people_index*5: people_index*5+5, 0].unsqueeze(0))
            conv_batch.append(conv2[people_index*6: people_index*6+6, 0].unsqueeze(0))
        conv_results.append(torch.cat(conv_batch, dim=1))
    conv_result = torch.cat(conv_results, dim=0)
    return conv_result


def json2heatmap(json_file, width, gaussian_kernel, gaussian_kernel_small, Tensor):
    import time
    time1 = time.time()
    for people in json_file['people']:
        channels =Tensor(11, 1, width, width).fill_(0)
        for face_point in (people['face_keypoints_2d'][3*i:3*i+3] for i in range(70)):
            channels[0, 0,  int(face_point[1]), int(face_point[0])] = 1
        for left_hand_point in (people['hand_left_keypoints_2d'][3*i:3*i+3] for i in range(21)):
            channels[1, 0, int(left_hand_point[1]), int(left_hand_point[0])] = 1
        for right_hand_point in (people['hand_right_keypoints_2d'][3*i:3*i+3] for i in range(21)):
            channels[2, 0, int(right_hand_point[1]), int(right_hand_point[0])] = 1
        for left_foot_point in (people['pose_keypoints_2d'][3*i:3*i+3] for i in [22, 23, 24]):
            channels[3, 0, int(left_foot_point[1]), int(left_foot_point[0])] = 1
        for right_foot_point in (people['pose_keypoints_2d'][3*i:3*i+3] for i in [19, 20, 21]):
            channels[4, 0, int(right_foot_point[1]), int(right_foot_point[0])] = 1
        body_list = people['pose_keypoints_2d']
        parts = [[0, 15, 16, 18], [0, 1, 2, 5, 8, 9, 12], [2, 3, 4], [5, 6, 7], [9, 10, 11], [12, 13, 14]]
        for part_index, part in enumerate(parts):
            for joint in part:
                channels[5 + part_index, :, int(body_list[joint * 3 + 1]), int(body_list[joint * 3])] = 1
        conv1 = functional.conv2d(Variable(channels[0:6]), Variable(gaussian_kernel_small), padding=5).data
        conv2 = functional.conv2d(Variable(channels[6:11]), Variable(gaussian_kernel), padding=50).data

        # heads = [[0, 15], [0, 16], [17, 15], [16, 18]]
        # for head_limb in heads:
        #     channels[5, 0, int(body_list[head_limb[0]*3 + 1]), int(body_list[head_limb[0]*3])] = 1
        #     channels[5, 0, int(body_list[head_limb[1]*3 + 1]), int(body_list[head_limb[1]*3])] = 1
        # limbs = [[0, 1], [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [8, 12], [9, 10], [12, 13], [10, 11], [13, 14], [2, 9], [5, 12]]
        # for index, limb in enumerate(limbs):
        #     channels[6 + index, 0, int(body_list[limb[0] * 3 + 1]), int(body_list[limb[0] * 3])] = 1
        #     channels[6 + index, 0, int(body_list[limb[1] * 3 + 1]), int(body_list[limb[1] * 3])] = 1
        # conv1 = functional.conv2d(Variable(channels[0:6]), Variable(gaussian_kernel_small), padding=5).data
        # conv2 = functional.conv2d(Variable(channels[6:22]), Variable(gaussian_kernel), padding=50).data
        conv_result = torch.cat((conv1, conv2), dim=0)

    return conv_result[:, 0, :, :]