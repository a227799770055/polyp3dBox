import torch, torchvision
import sys, os, csv
sys.path.append('/home/insign2/work/flexible-yolov5')
import cv2
import numpy as np
from copy import deepcopy
from utils.general import non_max_suppression, box_iou
import time
from scipy.stats import scoreatpercentile
import csv 


def image_preprocessing(image):
    image = cv2.resize(image,(512,512))
    img_h, img_w = image.shape[0], image.shape[1]
    origin_image = deepcopy(image)
    image = image[:, :, ::-1].transpose(2, 0, 1)
    image = np.ascontiguousarray(image)
    image = torch.from_numpy(image).unsqueeze(0) # to pytorch
    image = image.float()
    image /= 255.0 # normalize
    return image, origin_image, img_h, img_w

def Detector(image, origin_size, conf=0.7):
    # predict
    result = model(image)[0]
    result = non_max_suppression(result, conf, 0.5, classes=0, agnostic=False)[0] # 進行 nms
    xFactor = origin_size[0]/512 # 計算 x factor 之後要將 box 校正回原圖大小
    yFactor = origin_size[1]/512 # 計算 y factor 之後要將 box 校正回原圖大小
    res =[]
    score = []
    for box in result:
        box = box.cpu().numpy()
        cls = str(int(box[5]))
        confidence = float(box[4])
        x0,y0,x1,y1 = int(box[0]*xFactor), int(box[1]*yFactor), int(box[2]*xFactor), int(box[3]*yFactor)
        res.append([x0, y0, x1, y1])
        score.append(confidence)
    return res, score

def rgb_quartile(rgb_image, gray_image, box):
    # calculate xc yc w h
    x0, y0, x1, y1 = box[0], box[1], box[2], box[3]
    xc, yc = (x0+x1)/2, (y0+y1)/2
    w, h = x1-x0, y1-y0
    # w, h 10 percentreduction
    w, h = w*0.9, h*0.9
    x0, x1 = int(xc-(w/2)), int(xc+(w/2))
    y0, y1 = int(yc-(h/2)), int(yc+(h/2))
    
    # slice x0:x1 y0:y1
    rgb_image = rgb_image[y0:y1, x0:x1, :]
    gray_image = gray_image[y0:y1, x0:x1]
    rgb_image = rgb_image.reshape(-1,3)
    gray_image = gray_image.reshape(-1)

    # look for scoreatpercentile
    q1_gray = np.quantile(gray_image,0.25,interpolation='lower')  #下四分位数
    q2_gray = np.quantile(gray_image,0.50,interpolation='nearest') #中四分位数
    q3_gray = np.quantile(gray_image,0.75,interpolation='higher') #上四分位数
    
    q1_blue = np.quantile(rgb_image[:,0],0.25,interpolation='lower')  #下四分位数
    q2_blue = np.quantile(rgb_image[:,0],0.50,interpolation='nearest') #中四分位数
    q3_blue = np.quantile(rgb_image[:,0],0.75,interpolation='higher') #上四分位数
    q1_green = np.quantile(rgb_image[:,1],0.25,interpolation='lower')  #下四分位数
    q2_green = np.quantile(rgb_image[:,1],0.50,interpolation='nearest') #中四分位数
    q3_green = np.quantile(rgb_image[:,1],0.75,interpolation='higher') #上四分位数
    q1_red = np.quantile(rgb_image[:,2],0.25,interpolation='lower')  #下四分位数
    q2_red = np.quantile(rgb_image[:,2],0.50,interpolation='nearest') #中四分位数
    q3_red = np.quantile(rgb_image[:,2],0.75,interpolation='higher') #上四分位数


    # q1_indices = np.argwhere(gray_image == q1)
    # q2_indices = np.argwhere(gray_image == q2)
    # q3_indices = np.argwhere(gray_image == q3)
    # q1_indices = q1_indices[int(q1_indices.shape[0]/2)]
    # q2_indices = q2_indices[int(q2_indices.shape[0]/2)]
    # q3_indices = q3_indices[int(q3_indices.shape[0]/2)]

    # q1_gray, q2_gray, q3_gray = gray_image[q1_indices][0], gray_image[q2_indices][0], gray_image[q3_indices][0]
    # q1_rgb, q2_rgb, q3_rgb = rgb_image[q1_indices], rgb_image[q2_indices], rgb_image[q3_indices]
    # q1_b, q2_b, q3_b = q1_rgb[0][0], q2_rgb[0][0], q3_rgb[0][0]
    # q1_g, q2_g, q3_g = q1_rgb[0][1], q2_rgb[0][1], q3_rgb[0][1]
    # q1_r, q2_r, q3_r = q1_rgb[0][2], q2_rgb[0][2], q3_rgb[0][2]
    return [q1_gray, q2_gray, q3_gray], [q1_blue, q2_blue, q3_blue], [q1_green, q2_green, q3_green], [q1_red, q2_red, q3_red]

# check dir and mkdir
def check_dir(dir_path):
    if not os.path.isdir(dir_path) and not os.path.isfile(dir_path):
        os.mkdir(dir_path)
        print('\n')
        print('Make dir {}'.format(dir_path))
    elif os.path.isdir(dir_path):
        print('{} is exist!'.format(dir_path))

if __name__ == '__main__':
    # check cuda avaliable or not
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # loading model 
    model_checkpoint = '/home/insign2/work/flexible-yolov5/Polyp/長海假體_morphyolo_0815/weights/best.pt'
    model = torch.load(model_checkpoint)['model']
    model = model.to(device).float()

    # direction of files and saveing
    root_dir = '/home/insign2/Pictures/test0831_delete'
    rgb_dir = os.path.join(root_dir, 'rgb')
    check_dir(rgb_dir)
    csv_dir = os.path.join(root_dir, 'csv')
    check_dir(csv_dir)
    roi_dir = os.path.join(root_dir, 'roi')
    check_dir(roi_dir)

    csv_path = os.path.join(csv_dir, 'rgb_qua.csv')
    with open(csv_path, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['image','gray_q1','gray_q2','gray_q3','b_q1','b_q2','b_q3','g_q1','g_q2','g_q3','r_q1','r_q2','r_q3', 'box'])
        images = os.listdir(rgb_dir)
        for image in images:
            name, _ = os.path.splitext(image)
            image_path = os.path.join(rgb_dir, image)
            
            # image pre-process
            img = cv2.imread(image_path)
            img, origin_image, img_h, img_w = image_preprocessing(img)
            img = img.to(device)
            gray_image = cv2.cvtColor(origin_image, cv2.COLOR_BGR2GRAY)

            # roi inference
            results, score = Detector(img, (img_w, img_h), conf=0.5)

            if len(results) >= 1: # plot roi in images
                for i in range(len(results)):
                    x0, y0, x1, y1 = results[i][0], results[i][1], results[i][2], results[i][3]
                    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
                    cv2.rectangle(origin_image, (x0,y0), (x1,y1), (0, 255, 0), 2)

                    # calculate rgb histgram
                    box = [x0, y0, x1, y1]
                    gray_image = gray_image
                    gray, b, g, r =rgb_quartile(origin_image, gray_image, box)

                    csvwriter.writerow([image,gray[0],gray[1],gray[2],b[0],b[1],b[2],g[0],g[1],g[2],r[0],r[1],r[2],box])

            
            # save roi image
            roi_path = os.path.join(roi_dir, image)
            cv2.imwrite(roi_path, origin_image)