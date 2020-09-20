import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

currentDir = os.getcwd().replace("src", "")
training_dataset_path = currentDir + "datasets/data_train.npz"
test_dataset_path = currentDir + "datasets/data_test.npz"

# def neighbourhood_denoise(image):
#     for blankPixel in image:



if __name__ == "__main__":
    dataset_train = np.load(training_dataset_path)
    images = dataset_train['data']
    plt.imshow(images[65])
    plt.show()
        

    
