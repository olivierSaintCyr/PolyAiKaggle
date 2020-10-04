

import tensorflow as tf

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
tf.compat.v1.Session(config=config)

import numpy as np
import os

import tensorflow.keras as keras
import time
import tensorboard
import matplotlib.pyplot as plt

currentDir = os.getcwd().replace("src", "")
dataset_path = currentDir + "datasets/"
training_dataset_path = dataset_path + "data_train.npz"
test_dataset_path = dataset_path + "data_test.npz"

def to_oneHot(labels):
    a = labels.astype(int)
    b = np.zeros((a.size, a.max()+1))
    b[np.arange(a.size),a] = 1
    return b

def get_class(modelOutput): #takes the max probability index of a row and turns it into a one hot vector
    b = np.zeros(modelOutput.shape)
    b[np.arange(b.shape[0]), np.argmax(a, axis=1)] = 1
    return b

class convNeuralNet:
    def __init__(self):
        self.model = self.modelInit()

    def modelInit(self):
        model = keras.models.Sequential()
        
        
        model.add(keras.layers.Conv2D(80, (3,3), input_shape = (32,32,3), activation='relu'))
        model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
        
        model.add(keras.layers.SpatialDropout2D(0.45))

        model.add(keras.layers.Conv2D(64, (3,3), activation='relu'))
        model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
        model.add(keras.layers.SpatialDropout2D(0.4))

        model.add(keras.layers.Conv2D(64, (3,3),  activation='relu'))
        model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))


        model.add(keras.layers.Flatten())
        model.add(keras.layers.BatchNormalization())
        
        
        model.add(keras.layers.Dense(200, activation = 'relu'))
        model.add(keras.layers.Dropout(0.4))

        model.add(keras.layers.Dense(200, activation = 'relu'))
        
        model.add(keras.layers.Dense(20, activation = 'softmax'))

        model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
    
    def modelLoad(self, path):
        self.model = keras.models.load_model(path)
    
    def train(self, x, y, nEpochs):
        self.model.fit(x, y, epochs = nEpochs)
    
    def predict(self, data_x):
        return self.model.predict(data_x)

    def ajustDropout(self, new_rates):
        if(len(new_rates)==3):
            self.model.layers[2].rate = new_rates[0]
            self.model.layers[5].rate = new_rates[1]
            self.model.layers[11].rate = new_rates[2]

    

if __name__ == "__main__":
    dataset = np.load(dataset_path + "denoisedTrain.npz")
    test = np.load(dataset_path + "denoisedTest.npz")
    
    images = dataset['arr_0']
    labels = dataset['arr_1']
    
    imagesTrain = images[0:45000]
    imageValidation = images[-4999:]
    
    lablesOneHot = to_oneHot(labels)
    lablesTrain = lablesOneHot[0:45000]
    lablesValidation = lablesOneHot[-4999:]
    
    predictImages = test['arr_0']
    
    print(imagesTrain.shape)
    
    modelConv = convNeuralNet()
    train_dataset = tf.data.Dataset.from_tensor_slices((imagesTrain, lablesTrain))
    
    I = 464
    modelConv.modelLoad(currentDir + "/superBigmodel5BackUP/" + str(I))

    # modelConv.modelLoad(currentDir + "/superBigmodel5Best/")
    # y = modelConv.model.predict(imageValidation)
    # m = tf.keras.metrics.CategoricalAccuracy()
    # m.update_state(lablesValidation, y)
    # m.result().numpy()
    # print(" accuracy : ",m.result().numpy())
    
    accuracy = []
    bestAccuracy = 0.5421084
    
    new_rates = [0.38, 0.36, 0.28]
    modelConv.ajustDropout(new_rates)
    
    for i in range(50):
        modelConv.train(imagesTrain, lablesTrain, nEpochs = 3)
        modelConv.ajustDropout([0, 0, 0])
        y = modelConv.model.predict(imageValidation)
        m = tf.keras.metrics.CategoricalAccuracy()
        m.update_state(lablesValidation, y)
        accuracy.append(m.result().numpy())
       
        print("i: ", i+I, " accuracy : ",accuracy[-1])
        modelConv.ajustDropout(new_rates)
        if((i +1)% 3==0):
            modelConv.model.save(currentDir + "/superBigmodel5BackUP/" + str(i+I))
        if(accuracy[-1]>bestAccuracy):
            print("BEST ACCURACY")
            modelConv.model.save(currentDir + "/superBigmodel5Best/")
            bestAccuracy = accuracy[-1]
    
    plt.plot(accuracy)
    plt.show()

