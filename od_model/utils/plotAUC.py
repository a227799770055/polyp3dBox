import os 
import csv
import matplotlib.pyplot as plt

def save_arrays_to_csv(arrays, headers, filename):
    rows = zip(*arrays)

    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)  # 寫入標題行
        writer.writerows(rows)


if __name__ == '__main__':

    checkpoint_list = ['polyp_morphDeep_效能驗證_0904'] 
    for checkpoint in checkpoint_list:
        csv_path = '/home/insign2/work/Poly_img/效能驗證資料集/test/res/{}_conMatrix.csv'.format(checkpoint)
        sensitive = [1]
        specificy = [1]
        TP, TN, FP, FN, conf= [0], [0], [0], [0], [0]
        with open(csv_path, 'r') as file:
            reader = csv.reader(file)
            next(reader)
            for i, row in enumerate(reader):
                if i%2 == 0:
                    s = int(row[0])/(int(row[0])+int(row[3]))
                    sensitive.append(s)
                    TP.append(row[0])
                    FN.append(row[3])
                    conf.append(row[7])
                else:
                    p = int(row[2])/(int(row[1])+int(row[2]))
                    specificy.append(p)
                    TN.append(row[1])
                    FP.append(row[2])

        # 儲存 csv
        header = ['conf', 'TP', 'TN', 'FP', 'FN', 'sensitive', 'specificy']
        arrays = [conf, TP, TN, FP, FN, sensitive, specificy]
        filename = '/home/insign2/work/Poly_img/效能驗證資料集/test/res/confusion_matrix_sorting_{}.csv'.format(checkpoint)
        save_arrays_to_csv(arrays, header, filename)

        # calculate AUC
        auc = 0
        for i in range(len(sensitive)-1):
            a1 = (specificy[i]-specificy[i+1])*sensitive[i+1]
            a2 = (specificy[i]-specificy[i+1])*(sensitive[i]-sensitive[i+1])/2
            auc += a1
            auc += a2
        print('AUC={}'.format(auc))

        # scatter plot
        plt.figure()
        plt.scatter(specificy, sensitive)
        plt.title("{} AUC".format(checkpoint))
        plt.xlabel("Specificy")
        plt.ylabel("Sensitive")
        plt.text(0.8, 0, 'AUC={}'.format(round(auc,3)))
        plt.savefig('/home/insign2/work/Poly_img/效能驗證資料集/test/res/{}.png'.format(checkpoint))
        plt.show()