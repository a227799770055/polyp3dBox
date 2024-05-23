import os 
import csv
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt

def plot_mAP(df, mAP):
    # 畫圖
    x, y = df['recall'].values, df['precision'].values
    x = np.append(x,x[-1])
    y = np.append(y,0)
    plt.scatter(x, y)
    plt.plot(x, y, '-')
    plt.title('Average Precision')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim(0,1.05)
    plt.ylim(0,1.05)
    # 保存图像
    plt.savefig('mAP.png')

def cal_precision(sorted_data, total_gt):
    df = pd.DataFrame(sorted_data,columns=['id', 'x0', 'y0', 'x1', 'y1', 'conf', 'iou', 'type'])
    df = df.assign(precision=' ', recall='')
    
    total_TP = 0
    total_TPFP = 0
    for index in range(len(df)):
        row = df.iloc[index]
        # 計算 precision = tp / tp+fp
        # 計算 recall = tp / total_gt
        # 如果 type 是 tp    
        if row['type'] == 'TP':
            total_TP+=1
            total_TPFP+=1
            df.at[index, 'precision'] = total_TP/total_TPFP
            df.at[index, 'recall'] = total_TP/total_gt
        # 如果 type 是 fp    
        elif row['type'] == 'FP':
            total_TPFP+=1
            df.at[index, 'precision'] = total_TP/total_TPFP
            df.at[index, 'recall'] = total_TP/total_gt
    return df

def cal_mAP(x,y):
    mAP = 0
    for i,item in enumerate(zip(x,y)):
        if i==0:
            mAP += x[i]*np.max(y[i:])
        else:
            mAP += (x[i]-x[i-1])*np.max(y[i:])
    mAP = round(mAP*100,2)
    return mAP

if __name__ == '__main__':
    # 開啟 CSV 檔案
    with open('/home/insign2/work/Poly_img/效能驗證資料集/test/yolo_result/results/conf0.5_result.csv', newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        data = list(reader)
    # 對資料進行排序
    sort_column_index = 5
    sorted_data = sorted(data, key=lambda x: x[sort_column_index], reverse=True)
    # 儲存到新的 csv
    with open('test_new.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(sorted_data)
    # 使用 pd 讀取資料
    df = cal_precision(sorted_data, total_gt=1259)
    # 作圖
    
    x, y = df['recall'].values, df['precision'].values
    mAP = cal_mAP(x,y)
    print('mAP ={}'.format(mAP))
    plot_mAP(df, mAP)
    print(df)
    df.to_csv("mAP.csv")

