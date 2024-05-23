import torch, torchvision
import sys, os, csv
sys.path.append('/home/insign2/work/flexible-yolov5')
import cv2
import numpy as np
from copy import deepcopy
from utils.general import non_max_suppression, box_iou
import time
def image_preprocessing(image):
    img_h, img_w = image.shape[0], image.shape[1]
    # image = letterbox(image, new_shape=640)[0]
    image = cv2.resize(image,(512,512))
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
        # res.append([x0, y0, x1, y1, score])
        res.append([x0, y0, x1, y1])
        score.append(confidence)
    return res, score

if __name__ == '__main__':
    # check cuda avaliable or not
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # loading model 
    model_checkpoint = '/home/insign2/work/flexible-yolov5/Polyp/長海假體_morphyolo_0815/weights/best.pt'
    model = torch.load(model_checkpoint)['model']
    model = model.to(device).float()

    polyp_dir = '/media/insign2/TOSHIBA EXT/長海深度實驗/raw/raw'
    son_dir = os.listdir(polyp_dir)
    for son in son_dir:
        images = os.listdir(os.path.join(polyp_dir, son))
        for image in images:
            try:
                img_path = os.path.join(polyp_dir, son, image)
                print(img_path)
                img = cv2.imread(img_path)
                img, origin_image, img_h, img_w = image_preprocessing(img)
                origin_image = cv2.resize(origin_image, (img_w,img_h))
                img = img.to(device)
                # start to detect
                s = time.time()
                results, score = Detector(img, (img_w, img_h), conf=0.5)
                e = time.time()
                
                if len(results) != 0:
                    with open(polyp_dir+'{}.txt'.format(image.split('.')[0]), 'w') as f:
                        
                        for box in results:
                            box0, box1 = (box[0], box[1]), (box[2], box[3])
                            f.write("{} {} {} {} {}\n".format(0, box[0], box[1], box[2], box[3]))
                            cv2.rectangle(origin_image, box0, box1, (0, 255, 0), 2)
                name = image.split('.')[0]+'_box.png'
                det = os.path.join(polyp_dir, name)
                cv2.imwrite(det, origin_image)
            except:
                pass