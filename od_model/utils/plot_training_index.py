import os
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    file_path = "/home/insign2/Documents/results.txt"
    column_names = ['Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'total', 'targets', 'img_size',  "P", "R", "mAP@.5", "mAP@.5-.95", 'b', 'o', 'c']
    
    df = pd.read_csv(file_path, delimiter='\t', names=column_names)  # Change the delimiter if needed
    print(df.head())

    epoch = [i for i in range(0,300)]
    metrics = ["total", "P", "R"]
    metrics_name = ["Loss", "Precision", "Recall"]
    
    for name, metric in zip(metrics_name,metrics):
        loss = df[metric]
        plt.plot(epoch, loss, marker='o', linestyle='-', color='blue')
        # Customize the plot (labels, title, legend, etc.)
        plt.xlabel('Epochs', fontsize=14)
        plt.ylabel('{}'.format(name), fontsize=14)
        plt.title('Training {}'.format(name), fontsize=18)
        plt.legend()
        plt.grid(True)
        # Display the plot
        plt.savefig("{}.png".format(name))
        plt.show()