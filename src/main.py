import os

# custom .py files
import exportCSV
#import model
import test
import train

currentDir = os.getcwd().replace("src", "")
dataset_path = currentDir + "datasets/"
training_dataset_path = dataset_path + "data_train.npz"
test_dataset_path = dataset_path + "data_test.npz"

# Verifies npz data file is in current directory
# Raises exeception otherwise
def verify_npz_file(file, path = dataset_path):
    path += file
    if not os.path.isfile(path):
        raise Exception("Verify file ", file, " is in current dirctory") 

def menu():
    print("-------------------")
    print("Select an option : ")
    print("1 - Denoise")
    print("2 - <other>")
    print("else - <exit>")
    print("-------------------")
    select = input()
    print("\n" * 80)
    if select == 1:
        dataset_train = np.load(training_dataset_path)
        dataset_test = np.load(test_dataset_path)
        #save_fill_blank_averageNN_batch("denoisedTrain.npz",dataset_train, 1)
        images = fill_blanks_averageNN_batch(dataset_test['data'], 1)
        np.savez("denoisedTest.npz", images)
    elif select == 2:
        pass
    else:
        print("-------------------")
        print("- - - Exiting - - -")
        print("-------------------")
        exit()

if __name__ == "__main__":

    verify_npz_file("denoisedTest.npz")
    
    while(True):
        menu()











    # Data of type [[0,6],[1,8],...[n,k]...] feeds into 'data'
    # where n is the image id and k is the images class
    #write_to_file(data)