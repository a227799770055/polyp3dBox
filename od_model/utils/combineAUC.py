import os 
import csv
import matplotlib.pyplot as plt

if __name__ == '__main__':
    root_dir = '/home/insign2/work/Poly_img/效能驗證資料集/test/res/'
    checkpoint_list = ['polyp_morphyoloNew_效能驗證_0816', 'polyp_yolo_效能驗證_0620', 'polyp_morphDeep_效能驗證_0904']
    auc_collect = []
    for i, check in enumerate(checkpoint_list):
        auc_collect.append(os.path.join(root_dir, 'confusion_matrix_sorting_{}.csv'.format(check)))

    color = ['red', 'blue', 'black', 'orange']
    label = ['DHNN auc:0.97 FLOPs:205', 'CSPDarknet53 auc:0.94 FLOPs:290', 'morphology auc:0.85 FLOPs:91']

    plt.figure()
    
    for z in zip(auc_collect,color,label):
        path = z[0]
        c = z[1]
        l = z[2]
        sensitive = [1]
        specificy = [1]
        with open(path, 'r') as file:
            reader = csv.reader(file)
            next(reader)
            for i, row in enumerate(reader):
                sen = float(row[5])
                spe = float(row[6])
                sensitive.append(sen)
                specificy.append(spe)
        plt.plot(specificy, sensitive, color=c, label=l)
    
    
    # plt.axhline(y=0.904, color='green', linestyle='--')
    # plt.axvline(x=0.092, color='green', linestyle='--')
    plt.title("AUC")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    
    
    left, bottom, width, height = 0.45, 0.4, 0.35, 0.35
    ax2 = plt.axes([left, bottom, width, height])
    for z in zip(auc_collect,color,label):
        path = z[0]
        c = z[1]
        l = z[2]
        sensitive = []
        specificy = []
        with open(path, 'r') as file:
            reader = csv.reader(file)
            next(reader)
            for i, row in enumerate(reader):
                sen = float(row[5])
                spe = float(row[6])
                if sen > 0.6 and spe < 0.2 and spe > 0.01:
                    sensitive.append(sen)
                    specificy.append(spe)
        ax2.plot(specificy, sensitive, color=c, label=l)
    
    plt.savefig(os.path.join('./', 'AUC_combine.png'))
    plt.show()