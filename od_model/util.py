import os
root_dir = os.getcwd()

import cv2
import sys
sys.path.append(os.path.join(root_dir,"od_model"))
sys.path.append(os.path.join(root_dir,"od_model", "utils"))
import torch
import torchvision
import numpy as np
from copy import deepcopy
from od_model.utils.general import non_max_suppression, bbox_iou
import time


class OD_Model:
    def __init__(self, checkpoint, device):
        print("Start to initalize model")
        self.od_model = torch.load(checkpoint)['model']
        self.od_model = self.od_model.to(device).float()
        self.device = device
    def image_preprocessing(self, image):
        img_h, img_w = image.shape[0], image.shape[1]
        # image = letterbox(image, new_shape=640)[0]
        image = cv2.resize(image,(640,640))
        origin_image = deepcopy(image)
        image = image[:, :, ::-1].transpose(2, 0, 1)
        image = np.ascontiguousarray(image)
        image = torch.from_numpy(image).unsqueeze(0) # to pytorch
        image = image.float()
        image /= 255.0 # normalize
        return image, origin_image, img_h, img_w

    def __call__(self, image_path, conf=0.3):
        img, origin_image, img_h, img_w = self.image_preprocessing(image_path)
        origin_image = cv2.resize(origin_image, (img_w,img_h))
        img = img.to(self.device)
        result = self.od_model(img)[0]
        result = non_max_suppression(result, conf, 0.5, classes=0, agnostic=False)[0] # 進行 nms
        xFactor = img_w/640 # 計算 x factor 之後要將 box 校正回原圖大小
        yFactor = img_h/640 # 計算 y factor 之後要將 box 校正回原圖大小
        res =[]
        score = []
        for box in result:
            box = box.cpu().numpy()
            cls = str(int(box[5]))
            confidence = float(box[4])
            x0,y0,x1,y1 = int(box[0]*xFactor), int(box[1]*yFactor), int(box[2]*xFactor), int(box[3]*yFactor) # 將 bbox 的位置調整回去原圖的位置
            res.append([x0, y0, x1, y1])
            score.append(round(confidence,2))
        result = {"res": res,
                    "score": score,
                    "origin_image": origin_image}
        return result
    
device = "cuda" if torch.cuda.is_available() else "cpu"
# model = OD_Model("{}/od_model/best.pt".format(root_dir), device)
model = OD_Model("od_model/polyp_20240321/weights/best.pt", device)
