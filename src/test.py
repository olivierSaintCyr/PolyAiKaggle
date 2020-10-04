import model
import exportCSV
import numpy as np
import os

currentDir = os.getcwd().replace("src", "")
dataset_path = currentDir + "datasets/"
training_dataset_path = dataset_path + "data_train.npz"
test_dataset_path = dataset_path + "data_test.npz"

def one_hot_to_lables(onehotData):
    return np.argmax(onehotData, axis = 1)


if __name__ == '__main__':
    convModel = model.convNeuralNet()
    convModel.modelLoad(currentDir + "/superBigmodel5Best")
    X = np.load(dataset_path + "denoisedTest.npz")
    
    prediction = np.array(convModel.predict(X['arr_0']))
    # prediction = np.array([[0,1,0,0],
    #                        [1,0,0,0],
    #                        [0,0,0,1],
    #                        [0,0,1,0]])

    i = np.arange(prediction.shape[0])

    labels = one_hot_to_lables(prediction)

    output = np.stack((i, labels), axis = 1)
    print(output)

    exportCSV.write_to_file(output.tolist(), outputFile = "prediction54.csv")
   
