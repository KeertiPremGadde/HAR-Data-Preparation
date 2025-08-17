import os
from unittest import skip
import numpy as np
import csv
import pandas as pd
from scipy.fft import skip_backend
# import re
from sklearn.impute import SimpleImputer

# Replace Nan values in data
def replaceNan (data):
    # Use median as strategy to replace nan values 
    imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    # Fit data excluding the first column (timestamp)
    imputer.fit(data[:, 1:])    
    data[:, 1:] = imputer.transform(data[:, 1:])

def load_datafile(filename, columns):
    if not os.path.exists(filename):
        print("file is not exits {}".format(filename))
        return

    df = pd.read_csv(filename, header = 0, names = columns)
    df=df.dropna()
    
    data_x = df[columns].to_numpy()
    nanCount = np.isnan(data_x).sum()
    #print(nanCount," Nan values")
    replaceNan(data_x)
    
    return data_x
recording = 1 #changefor the person number
columns_label = ['Label', 'Left_start', 'Left_end', 'Right_start', 'Right_end']
label = load_datafile(f"./Recording{str(recording)}/Star_Label_P{recording}_AllSessions.csv", columns_label)
#print(label)
f = []
count = 0
countskip = 0
for i in range(len(label)):
    if label[i][0] == 12.0:
        if i <=6:
            pass
        elif countskip == 0:
            with open(f"./Recording{str(recording)}/Star_Label_P{recording}_Session{count+1}.csv", 'w', newline="") as file:
                # using csv.writer method from CSV package
                write = csv.writer(file)
                write.writerow(columns_label)
                write.writerows(f)
                #print(f)
                f = []
            count = count+1
            #print(count)
            countskip = countskip + 1
    else:
        countskip = 0
        f.append(label[i])