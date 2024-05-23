import torch, torchvision
import sys, os, csv
sys.path.append('/home/insign2/work/flexible-yolov5')
import cv2
import numpy as np
from copy import deepcopy
from utils.general import non_max_suppression, box_iou
from datetime import datetime


def image_preprocessing(image):
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

def Detector(image, origin_size, conf=0.5):
    # predict
    result = model(image)[0]
    result = non_max_suppression(result, conf, 0.5, classes=0, agnostic=False)[0] # 進行 nms
    xFactor = origin_size[0]/640 # 計算 x factor 之後要將 box 校正回原圖大小
    yFactor = origin_size[1]/640 # 計算 y factor 之後要將 box 校正回原圖大小
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

def read_label(label_path, img_w, img_h):
    # 讀取 label
    labels =[]
    with open(label_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            tmp = []
            line = line.split(' ')
            for i in range(len(line)):
                if i == 0:
                    tmp.append(str(int(line[i])))
                else:
                    tmp.append(float(line[i]))
            labels.append(tmp)
     # convert xywh to xyxy
    for i in range(len((labels))):
        label = labels[i]
        x,y,w,h = label[1]*img_w, label[2]*img_h, label[3]*img_w, label[4]*img_h
        x0, x1 = x-(w/2), x+(w/2)
        y0, y1 = y-(h/2), y+(h/2)
        labels[i] = [int(x0), int(y0), int(x1), int(y1)]
    return labels

def result_csv_write(images, label_dir, save_dir, img_dir, conf=0.5):
    with open(os.path.join(save_dir, 'conf{}_result.csv'.format(conf)),'w', newline='') as csvfile:
        print(os.path.join(save_dir, 'conf{}_result.csv'.format(conf)))
        writer = csv.writer(csvfile)
        writer.writerow(['file_name', 'x0', 'y0', 'x1', 'y1', 'score', 'iou', 'type'])
        cf_matrix = {'TP':0, 'TN':0, 'FP':0, 'FN':0}
        for i in images:
            id = i.split('.')[0]
            img_path = os.path.join(img_dir, i)
            label_path = os.path.join(label_dir, id+'.txt')

            img = cv2.imread(img_path)
            image, origin_image, img_h, img_w = image_preprocessing(img)
            origin_image = cv2.resize(origin_image, (img_w,img_h))
            image = image.to(device)
            
            # 進行 predict
            results, score = Detector(image, (img_w, img_h), conf)
            # 讀取 label
            labels = read_label(label_path, img_w, img_h)
            # 判定 results 和 labels 是否為0
            # 有 detect 也有 gt 
            if len(labels)!=0 and len(results)!=0:
                labels, results = torch.tensor(labels), torch.tensor(results)
                # 計算 iou
                iou = box_iou(labels, results).numpy()[0]
                for item in zip(results, score, iou):
                    gt = labels[0].numpy()
                    if item[2] >=0.5:
                        box = item[0].numpy()
                        writer.writerow([i, box[0],box[1],box[2],box[3],item[1],item[2],'TP'])
                        cv2.rectangle(origin_image,(box[0],box[1]),(box[2],box[3]), (0,255,0), 2)
                        cf_matrix['TP'] += 1
                    elif item[2] >0.1 and item[2]<0.5:
                        box = item[0].numpy()
                        writer.writerow([i, box[0],box[1],box[2],box[3],item[1],item[2],'FP'])
                        cv2.rectangle(origin_image,(box[0],box[1]),(box[2],box[3]), (255,0,0), 2)
                        cf_matrix['FP'] += 1
                    cv2.rectangle(origin_image,(gt[0],gt[1]),(gt[2],gt[3]), (0,0,255), 4)
                cv2.imwrite(os.path.join(save_dir,'TP_{}.png'.format(id)), origin_image)
            # 沒有 detect 有 gt 
            elif len(labels)!=0 and len(results)==0:
                writer.writerow([i, 0,0,0,0,0,0,'FN'])
                box = labels[0]
                cv2.rectangle(origin_image,(box[0],box[1]),(box[2],box[3]), (0,0,255), 4)
                cf_matrix['FN'] += 1
                cv2.imwrite(os.path.join(save_dir,'FN_{}.png'.format(id)), origin_image)
            # 有 detect 沒有 gt 
            elif len(labels)==0 and len(results)!=0:
                for item in zip(results, score):
                    box = item[0]
                    writer.writerow([i, box[0],box[1],box[2],box[3],item[1],0,'FP'])
                    cv2.rectangle(origin_image,(box[0],box[1]),(box[2],box[3]), (255,0,0), 2)
                    cf_matrix['FP'] += 1
                cv2.imwrite(os.path.join(save_dir,'FP_{}.png'.format(id)), origin_image)
            # 沒有 detect 也沒有 gt 
            else:
                writer.writerow([i, 0,0,0,0,0,0,'TN'])
                cf_matrix['TN'] += 1
    return cf_matrix

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # checkpoint_list = ['polyp_morphyoloNew_效能驗證_0816', 'polyp_yolo_效能驗證_0620', 'polyp_morresnet18_效能驗證_0814', 'polyp_resnet_效能驗證_0808']
    checkpoint_list = ['polyp_morphDeep_效能驗證_0904'] 
    save_dir = '/home/insign2/work/Poly_img/效能驗證資料集/test/0828'
    
    for checkpoint in checkpoint_list:

        print('Start to test checkpoint: {}'.format(checkpoint))

        # loading model
        model_checkpoint = './Polyp/{}/weights/best.pt'.format(checkpoint)
        model = torch.load(model_checkpoint)['model']
        model = model.to(device).float()
        
        # Positive Dataset
        p_img_dir = '/home/insign2/work/Poly_img/效能驗證資料集/test/polyp_rgb'
        p_label_dir = '/home/insign2/work/Poly_img/效能驗證資料集/test/polyp_labels'
        p_save_dir = os.path.join(save_dir, '{}_pos'.format(checkpoint))
        # Negative Dataset
        np_img_dir = '/home/insign2/work/Poly_img/效能驗證資料集/test/nonpolyp_rgb'
        np_label_dir = '/home/insign2/work/Poly_img/效能驗證資料集/test/nonpolyp_labels'
        np_save_dir = os.path.join(save_dir, '{}_neg'.format(checkpoint))
        
        if not os.path.isdir(p_save_dir):
            os.mkdir(p_save_dir)
        if not os.path.isdir(np_save_dir):
            os.mkdir(np_save_dir)    

        p_images = os.listdir(p_img_dir)
        np_images = os.listdir(np_img_dir)
        # 設定目前資料即是 positive 或是 negative
        cf_matrix_types = ['Positive', 'Negative']

        # 設定要測試的 confidence threshold
        conf_list = np.arange(0.01, 1, 0.003)
        # conf_list = [0.214]
        
        with open(os.path.join(save_dir, '{}_conMatrix.csv'.format(checkpoint)), 'w') as file:
            writer = csv.writer(file)
            writer.writerow(['TP', 'TN', 'FP', 'FN', 'Sensitive',  'Specificy', 'TP rate', 'conf'])

            for conf in conf_list:
                print('----------')
                print('conf = {}'.format(conf), )
                for cf_matrix_type in cf_matrix_types:    
                    if cf_matrix_type == 'Positive':
                        cf_matrix = result_csv_write(p_images, p_label_dir, p_save_dir, p_img_dir, conf)
                        cf_matrix['TP'] = len(p_images) - cf_matrix['FN']
                        cf_matrix['Sensitive'] = cf_matrix['TP']/(cf_matrix['TP']+cf_matrix['FN'])
                        cf_matrix['TP rate'] = cf_matrix['TP']/(cf_matrix['TP']+cf_matrix['FP'])
                        writer.writerow([cf_matrix['TP'], cf_matrix['TN'], cf_matrix['FP'], cf_matrix['FN'], cf_matrix['Sensitive'], '0', cf_matrix['TP rate'], conf])
                        print(cf_matrix)
                    elif cf_matrix_type == 'Negative':
                        cf_matrix = result_csv_write(np_images, np_label_dir, np_save_dir, np_img_dir, conf)
                        cf_matrix['Specificy'] = cf_matrix['TN']/len(np_images)
                        writer.writerow([cf_matrix['TP'], cf_matrix['TN'], cf_matrix['FP'], cf_matrix['FN'], '0', cf_matrix['Specificy'], '0'])
                        print(cf_matrix)
    print('Finish', datetime.now())