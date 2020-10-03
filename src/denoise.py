import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def averageNN(image, i_, j_, radius):
    sumPixel = 0
    nPixel = 0
    for i in range(i_ - radius, i_ + radius + 1):
        for j in range(j_ - radius, j_ + radius + 1):
            if ((i >= 0) and (j >= 0)):
                if((i < image.shape[0]) and (j < image.shape[1])):
                        if((i != i_) and  (j != j_)): # remove the blank pixel
                            if ((image[i][j] != np.zeros(3)).all()): #remove blank neighboor
                                sumPixel += image[i][j]
                                nPixel += 1
    if nPixel != 0:
        return sumPixel/(nPixel)
    else:
        return sumPixel

def fill_blanks_averageNN(image, radius):
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if (image[i][j] == np.zeros(3)).all():
                new_pixel = averageNN(image, i, j, radius)
                image[i][j] = new_pixel
    return image
    
def fill_blanks_averageNN_batch(dataset, radius):
    new_dataset = []
    for i in tqdm(range(dataset.shape[0])):
        new_dataset.append(fill_blanks_averageNN(dataset[i], radius)/255.0)
    return np.asarray(new_dataset)

def save_fill_blank_averageNN_batch(filename, dataset, radius):
    images = fill_blanks_averageNN_batch(dataset['data'], radius)
    np.savez(filename, images, dataset['labels'], dataset['metadata'])



if __name__ == "__main__":





