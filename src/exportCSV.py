import csv

# Data of type [[0,6],[1,8],...[n,k]...] feeds into 'data'
# where n is the image id and k is the images class
def write_to_file(data, outputFile = "output.csv"):
    myFile = open(outputFile, 'w', newline='')
    with myFile:
        writer = csv.writer(myFile)
        data = [["id", "class"]] + data
        writer.writerows(data)
        print(data)
    print("Writing to ", outputFile," completed")