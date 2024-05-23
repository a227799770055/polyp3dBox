from find_black_point import find_black_point
from od_model import model
import cv2
import numpy as np
import os

def quadrantDetermine(box, hole):
    slop_rate = (hole[1]-box[1])/(hole[0]-box[0])

    if box[0] > hole[0] and box[1] > hole[1]:
        return "leftTop",slop_rate
    elif box[0] > hole[0] and box[1] < hole[1]:
        return "leftDown",slop_rate
    elif box[0] < hole[0] and box[1] > hole[1]:
        return "rightTop",slop_rate
    else:
        return "rightDown",slop_rate

def point3DBox(box, quadrant, slop_rate):
    x0, y0, x1, y1 = box[0], box[1], box[2], box[3]
    diagonal_leng = np.sqrt((box[2]-box[0])**2 + (box[3]-box[1])**2)
    base_leng = diagonal_leng * 0.1
    base_x = int(np.sqrt((base_leng**2)/(1+(slop_rate**2))))
    base_y = int((base_x*slop_rate))

    if quadrant == "leftTop":
        p1 = (x0-base_x, y1-base_y)
        p2 = (x0-base_x, y0-base_y)
        p3 = (x1-base_x, y0-base_y)
    elif quadrant == "rightTop":
        p1 = (x0+base_x, y0-base_y)
        p2 = (x1+base_x, y0-base_y)
        p3 = (x1+base_x, y1-base_y)
    elif quadrant == "leftDown":
        p1 = (x1-base_x, y1+base_y)
        p2 = (x0-base_x, y1+base_y)
        p3 = (x0-base_x, y0+base_y)
    else:
        p1 = (x1+base_x, y0+base_y)
        p2 = (x1+base_x, y1+base_y)
        p3 = (x0+base_x, y1+base_y)

    return (p1, p2, p3)

def gray_point3DBox(box, quadrant, slop_rate):
    x0, y0, x1, y1 = box[0], box[1], box[2], box[3]
    diagonal_leng = np.sqrt((box[2]-box[0])**2 + (box[3]-box[1])**2)
    base_leng = diagonal_leng * 0.05
    base_x = int(np.sqrt((base_leng**2)/(1+(slop_rate**2))))
    base_y = int((base_x*slop_rate))

    if quadrant == "leftTop":
        p1 = (x0-base_x, y1-base_y)
        p2 = (x0-base_x, y0-base_y)
        p3 = (x1-base_x, y0-base_y)
    elif quadrant == "rightTop":
        p1 = (x0+base_x, y0-base_y)
        p2 = (x1+base_x, y0-base_y)
        p3 = (x1+base_x, y1-base_y)
    elif quadrant == "leftDown":
        p1 = (x1-base_x, y1+base_y)
        p2 = (x0-base_x, y1+base_y)
        p3 = (x0-base_x, y0+base_y)
    else:
        p1 = (x1+base_x, y0+base_y)
        p2 = (x1+base_x, y1+base_y)
        p3 = (x0+base_x, y1+base_y)

    return (p1, p2, p3)

def plot3DBox(img, box, threedPoints, quadrant):
    x0, y0, x1, y1 = box[0], box[1], box[2], box[3]
    if quadrant == "leftTop":
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 3)
        cv2.circle(img, threedPoints[0], 3, (0, 255, 0), -1) 
        cv2.circle(img, threedPoints[1], 3, (0, 255, 0), -1) 
        cv2.circle(img, threedPoints[2], 3, (0, 255, 0), -1)
        cv2.line(img, threedPoints[0], (x0,y1), (0, 255, 0), 3)
        cv2.line(img, threedPoints[1], (x0,y0), (0, 255, 0), 3)
        cv2.line(img, threedPoints[2], (x1,y0), (0, 255, 0), 3)
    elif quadrant == "rightTop":
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 3)
        cv2.circle(img, threedPoints[0], 3, (0, 255, 0), -1)
        cv2.circle(img, threedPoints[1], 3, (0, 255, 0), -1) 
        cv2.circle(img, threedPoints[2], 3, (0, 255, 0), -1)
        cv2.line(img, threedPoints[0], (x0,y0), (0, 255, 0), 3)
        cv2.line(img, threedPoints[1], (x1,y0), (0, 255, 0), 3)
        cv2.line(img, threedPoints[2], (x1,y1), (0, 255, 0), 3)
    elif quadrant == "leftDown":
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 3)
        cv2.circle(img, threedPoints[0], 3, (0, 255, 0), -1)
        cv2.circle(img, threedPoints[1], 3, (0, 255, 0), -1) 
        cv2.circle(img, threedPoints[2], 3, (0, 255, 0), -1)
        cv2.line(img, threedPoints[0], (x1,y1), (0, 255, 0), 3)
        cv2.line(img, threedPoints[1], (x0,y1), (0, 255, 0), 3)
        cv2.line(img, threedPoints[2], (x0,y0), (0, 255, 0), 3)
    else:
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 3)
        cv2.circle(img, threedPoints[0], 3, (0, 255, 0), -1)
        cv2.circle(img, threedPoints[1], 3, (0, 255, 0), -1) 
        cv2.circle(img, threedPoints[2], 3, (0, 255, 0), -1)
        cv2.line(img, threedPoints[0], (x1,y0), (0, 255, 0), 3)
        cv2.line(img, threedPoints[1], (x1,y1), (0, 255, 0), 3)
        cv2.line(img, threedPoints[2], (x0,y1), (0, 255, 0), 3)
    cv2.line(img, threedPoints[0], threedPoints[1], (0, 255, 0), 3)
    cv2.line(img, threedPoints[1], threedPoints[2], (0, 255, 0), 3)

if __name__ == '__main__':

    img_dir = "data"
    images = os.listdir(img_dir)
    for image in images:
        img = cv2.imread(os.path.join(img_dir, image))

        black_coor, gray_value = find_black_point(img)
        cv2.circle(img, black_coor, 3, (0, 255, 0), -1)

        boxs = model(img)
        box = boxs['res'][0]
        cent = (int((box[2]+box[0])/2), int((box[3]+box[1])/2))

        quadrant,slop_rate = quadrantDetermine(cent, black_coor)

        threedPoints = point3DBox(box, quadrant, slop_rate)

        if gray_value < 100:
            threedPoints = point3DBox(box, quadrant, slop_rate)
        else:
            threedPoints =  gray_point3DBox(box, quadrant, slop_rate)
        plot3DBox(img, box, threedPoints, quadrant)
        cv2.imwrite("./res/{}_{}_{}".format(gray_value, quadrant, image), img)