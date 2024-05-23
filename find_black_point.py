import cv2
import numpy as np

def f_b_p(img,shape,num_p=5,offset=(0,0)):
    x_step = int(shape[1]/(num_p+1))
    start_x = offset[0]+x_step
    y_step = int(shape[0]/(num_p+1))
    start_y = offset[1]+y_step
    dict = {}
    for i in range(num_p):
        for j in range(num_p):
            x = start_x + i * x_step
            y = start_y + j * y_step
            dict[(x,y)]=img[y,x]
    filtered_sorted_dict = {k:v for k,v in sorted(dict.items(), key=lambda x: x[1]) if v > 5}
    return filtered_sorted_dict

def find_black_point(img,num_p=5,is_draw=False):
    g_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    dict1 = f_b_p(g_img,g_img.shape,num_p=num_p)
    first_coord = next(iter(dict1))
    stepx,stepy = int(img.shape[1]/num_p),int(img.shape[0]/num_p)
    offset_x,offset_y = first_coord[0]-int(stepx),first_coord[1]-int(stepy)
    need_shape = (stepy*2,stepx*2)
    dict2 = f_b_p(g_img,need_shape,num_p=num_p,offset=(offset_x,offset_y))
    [loc for  loc in dict2.keys()]
    ans_coord= next(iter(dict2))
    if is_draw:
        circle_size = 5
        print(dict1)
        for coord in dict1:
            cv2.circle(img,coord,circle_size,(0,255,255),0)
        print(dict2)
        for coord in dict2:
            cv2.circle(img,coord,circle_size,(0,255,0),0)
        cv2.circle(img,ans_coord,circle_size,(0,0,255),0)
    gray_value = (g_img[ans_coord[1], ans_coord[0]])
    return ans_coord, gray_value

def bp_bgr_quartiles(img, num_p=5):
    g_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    dict1 = f_b_p(g_img,g_img.shape,num_p=num_p)
    
    # find dict1 bgr quartiles
    b,g,r = [], [], []
    res_full = []
    for d in dict1:
        bgr = img[d[1],d[0]]
        b.append(bgr[0])
        g.append(bgr[1])
        r.append(bgr[2])
    for color in [b,g,r]:
        res_full.append(np.percentile(color, 25)/255)
        res_full.append(np.percentile(color, 50)/255)
        res_full.append(np.percentile(color, 75)/255)

    first_coord = next(iter(dict1))
    stepx,stepy = int(img.shape[1]/num_p),int(img.shape[0]/num_p)
    offset_x,offset_y = first_coord[0]-int(stepx),first_coord[1]-int(stepy)
    need_shape = (stepy*2,stepx*2)
    dict2 = f_b_p(g_img,need_shape,num_p=num_p,offset=(offset_x,offset_y))
    xcyc = list(dict2.keys())[0]
    coord = [loc for  loc in dict2.keys()]
    b,g,r = [], [], []
    res = []
    for c in coord:
        bgr = img[c[1], c[0]]
        b.append(bgr[0])
        g.append(bgr[1])
        r.append(bgr[2])
    for color in [b,g,r]:
        # unnormalize
        res.append(np.percentile(color, 25)/255)
        res.append(np.percentile(color, 50)/255)
        res.append(np.percentile(color, 75)/255)
        # res.append(np.percentile(color, 25))
        # res.append(np.percentile(color, 50))
        # res.append(np.percentile(color, 75))
    return res, xcyc, res_full

if __name__=="__main__":
    image = cv2.imread( "images/123.png")
    quartiles, xcyc = bp_bgr_quartiles(image,num_p=5)
    # print(quartiles)
    # ans_coord = find_black_point(image,num_p=5,is_draw=True)
    # print(f"ans_coord={ans_coord}")
    # cv2.imshow("output",image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()