import csv
import numpy as np
import matplotlib.pyplot as plt

def load_series(filname,seires_index=1):
    try:
        with open(filname) as csvfile:
            csvreader=csv.reader(csvfile)
            data=[float(row[seires_index]) for row in csvreader if len(row)>0]
            normalized=(data-np.mean(data))/np.std(data)
        return  normalized
    except IOError:
        return  None

def split_data(data,percent_train=0.80):
    num_rows=len(data)
    train_data,test_data=[],[]
    for idx,row in enumerate(data):
        if idx<num_rows*percent_train:
            train_data.append(row)
        else:
            test_data.append(row)
    return  train_data,test_data




if __name__=="__main__":
    timeseries=load_series("data.csv")
    print(np.shape(timeseries))
    plt.figure()
    plt.plot(timeseries)
    plt.show()


