# ！/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time: 2021/6/8 13:47
# @Author: ''
# @ FileName: postprocessing.py
import cv2
import os
import glob
import time
from lib.compute import isCorrect

classes = ["q1_f5", "q3_f5", "q1_f2", "q1_f3", "q2_f5",
           "q1_f4", "q3_f1", "q2_f4", "q1_f1", "q2_f3",
           "q3_f4", "q3_f2", "q2_f2", "q2_f1", "q3_f3"]
colors = [(0, 0, 255), (255, 255, 0), (0, 255, 0), (0, 255, 255), (255, 0, 255),
          (0, 255, 255), (255, 255, 0), (128, 0, 0), (0, 255, 0), (128, 0, 0),
          (255, 255, 0), (255, 255, 0), (0, 140, 255), (0, 140, 255), (255, 255, 0)]
# (127, 255, 212), (100, 149, 237)


def plot_one_box(x, img, color=(128, 128, 128), opt='0', label=None, line_thickness=3):
    # plot one bounding box on image 'img' using Opencv
    # opt == '0' : disable put labels in the image, otherwise when opt is set '1'
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    p1, p2 = (int(x[1]), int(x[2])), (int(x[3]), int(x[4]))
    cv2.rectangle(img, p1, p2, color, thickness=tl, lineType=cv2.LINE_AA)
    #if label and opt == '1':
    #    tf = max(tl - 1, 1)  # front thickness
    #    t_size = cv2.getTextSize(label, 0, fontScale=tl/3, thickness=tf)[0]
    #    p2 = (p1[0] + t_size[0], t_size[1] - 3)
    #    cv2.rectangle(img, p1, p2, color, -1, cv2.LINE_AA)
    #    cv2.putText(img, label, (p1[0], p1[1]-2), 0, tl/3, [255, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def write_txt(txt_path, data):
    # data is a list from detect video results
    with open(txt_path, 'a') as f:
        for ind in range(0, len(data), 6):
            f.write(data[ind:ind + 6] + '\n')
        f.close()


def read_real_video(model, frame, opt='0', confidence=0.0):
    res = model.locate(frame)
    # end_time = time.time()
    # print('Recognition time: ', str(end_time - start_time))
    count = 0
    for i in range(0, len(res), 6):
        # print(i)
        # label = res[i]
        res[i] = res[i]-1
        if float(res[i + 5]) >= confidence:
            plot_one_box(res[i:i+6], frame, color=colors[int(res[i])], opt=opt, label=classes[int(res[i])], line_thickness=3)
            count = count + 1
            print(classes[int(res[i])])
        
    return calc_quality_maturity(res,confidence),count


def play_video(model, frame, opt='0', txt_path='./logs/video.txt', confidence=0.0):
    start_time = time.time()
    res = model.locate(frame)
    end_time = time.time()
    print('Recognition time: ', str(end_time - start_time))
    count = 0
    for i in range(0, len(res), 6):
        # label = res[i]
        res[i] = res[i]-1
        if int(res[i + 5]) >= confidence:
            plot_one_box(res[i:i+6], frame, color=colors[int(res[i])], opt=opt, label=classes[int(res[i])], line_thickness=3)
            count = count + 1
    write_txt(txt_path, '{0}: {1}\n'.format(count, res))

    return calc_quality_maturity(res,confidence),count,(end_time - start_time)


def read_pic(model, img, opt='1', confidence=0.0):
    res = model.locate(img)
    for i in range(0, len(res), 6):
        # res is a list such as [label, c0, c1, c2, c3, confidence, ...]
        # print(i)
        # label = res[i]
        res[i] = res[i]-1
        if int(res[i+5]) >= confidence:
            plot_one_box(res[i:i + 7], img, color=colors[int(res[i])], opt=opt, label=classes[int(res[i])],
                         line_thickness=3)
    return calc_quality_maturity(res,confidence)


def calc_quality_maturity(data,confidence):
    # data is a list such as [label, c0, c1, c2, c3, confidence]
    m = 0
    q = 0
    number = 0
    for j in range(0, len(data), 6):
        if(confidence > data[j+5]) :
            continue
        number = number + 1
        quality_cla = classes[int(data[j])].split('_')[0]
        maturity_cla = classes[int(data[j])].split('_')[1]
        if quality_cla == 'q1':
            q = q + 1
        elif quality_cla == 'q2':
            q = q + 0.5
        elif quality_cla == 'q3':
            continue
        
        if maturity_cla == 'f5':
            m = m + 1
        elif maturity_cla == 'f3' or maturity_cla == 'f4':
            m = m + 0.7
        elif maturity_cla == 'f1' or maturity_cla == 'f2':
            m = m + 0.2
    if number == 0:
        return 0,0
    else:
        quality = q / number
        maturity = m / number
        return quality, maturity


def draw_ground_truth(imgs_path, txts_path, out_path):
    x = []
    for img_path in glob.glob(imgs_path + '*.jpg'):
        _, name_att = os.path.split(img_path)
        name = name_att.split('.')[0]
        with open(txts_path + name + '.txt') as f:
            x_cache = f.readlines()
            x_ = [r.strip() for r in x_cache]
            for ind in range(len(x_)):
                x.append(x_[ind].split(' '))

        img = cv2.imread(img_path, 1)
        tl = round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
        for i in range(0, len(x), 6):
            p1, p2 = (int(x[i + 1]), int(x[i + 2])), (int(x[i + 3]), int(x[i + 4]))
            cv2.rectangle(img, p1, p2, (127, 255, 212), thickness=tl, lineType=cv2.LINE_AA)
        cv2.imwrite(out_path + name + '.jpg', img)


def plot_one_box_save(model, img_path, color=(128, 128, 128), confidence=0.0):
    # plot one bounding box on image 'img' using Opencv
    # opt == '0' : disable put labels in the image, otherwise when opt is set '1'
    img = cv2.imread(img_path)
    gt = []
    x = model.locate(img)
    tl = round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    with open(img_path[:-3]+'txt',"r") as f:
        while True:
            data = f.readline()
            if not data:
                break
            if data == '\n':
                break
            data = data.split(' ')
            x_min = float(data[1])*img.shape[1]-float(data[3])*img.shape[1]/2
            y_min = float(data[2])*img.shape[0]-float(data[4])*img.shape[0]/2
            x_max = float(data[1])*img.shape[1]+float(data[3])*img.shape[1]/2
            y_max = float(data[2])*img.shape[0]+float(data[4])*img.shape[0]/2
            gt.append([data[0], x_min, y_min, x_max, y_max])
            #cv2.rectangle(img, (int(x_min),int(y_min)), (int(x_max),int(y_max)), colors[int(data[0])], thickness=tl, lineType=cv2.LINE_AA)
    count = 0
    pre = 0
    for i in range(0, len(x), 6):
        if x[i+1] >= confidence:
            x[i] = x[i]-1
            p1, p2 = (int(x[i + 1]), int(x[i + 2])), (int(x[i + 3]), int(x[i + 4]))
            for j in range(len(gt)):
                if isCorrect(gt[j],x[i:i+5],0.7):
                    print(pre)
                    pre = pre + 1
                    break
            
            label = x[i]
            color = colors[int(label)]
            cv2.rectangle(img, p1, p2, color, thickness=tl, lineType=cv2.LINE_AA)
            count = count+1
    basename = os.path.basename(img_path).split('.')[0]
    cv2.imwrite('./logs/' + str(basename) + '_true.jpg',img)
    return calc_quality_maturity(x,confidence), count, min(0.99,float(pre)/count)


def delete_files_by_dir(dir_path):
    """
    the function is used to delect files under according directory path.
    :param dir_path: directory path given
    :return: none
    """
    ls = os.listdir(dir_path)
    for i in ls:
        c_path = os.path.join(dir_path, i)
        if os.path.isdir(c_path):
            delete_files_by_dir(c_path)
        else:
            os.remove(c_path)
