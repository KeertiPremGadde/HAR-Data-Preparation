import os
import numpy as np
import sys

import pandas as pd
# import re
from sklearn.impute import SimpleImputer

import _pickle as cp

# Replace Nan values in data
def replaceNan (data):
    # Use median as strategy to replace nan values 
    imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    # Fit data excluding the first column (timestamp)
    imputer.fit(data[:, 1:])    
    data[:, 1:] = imputer.transform(data[:, 1:])



def process_WISDM_dataset(dataset_name, data, label):
    """Function defined as a pipeline to process individual OPPORTUNITY files

    :param data: numpy integer matrix
        Matrix containing data samples (rows) for every sensor channel (column)
    :param label: string, ['gestures' (default), 'locomotion']
        Type of activities to be recognized
    :return: numpy integer matrix, numy integer array
        Processed sensor data, segmented into features (x) and labels (y)
    """
    
    data_x = data[["x-axis","y-axis","z-axis"]].to_numpy()
    data_y = data['activity'].to_numpy()
    data_s = data['user'].to_numpy()
    
    print(data_x.shape, data_x.dtype)
    print(data_y.shape, data_y.dtype)
    print(data_s.shape, data_s.dtype)
    # Check if there are Nan values in the dataset
    nanCount = np.isnan(data_x).sum()
    print(nanCount," Nan values")

    # Taking care of missing data (Nan)
    print("Replacing Nan values ...")
    replaceNan(data_x)

    # Check again if there are Nan values in the dataset
    nanCount = np.isnan(data_x).sum()
    print(nanCount," X Nan values")

    # data_y = adjust_idx_labels(data_y, label, dataset_name)
    data_y = data_y.astype(int)
    
    print(data_x.shape, data_y.shape, np.unique(data_y), len(np.unique(data_y)), data_s.shape, np.unique(data_s), len(np.unique(data_s)))
    
    data_x = normalize_std(data_x)

    print(data_x.shape, data_y.shape, np.unique(data_y), len(np.unique(data_y)))

    for i in range(1,len(np.unique(data_s))+1 ):
        x = data_x[np.where(data_s==i)]
        y = data_y[np.where(data_s==i)]

        print(i, x.shape, y.shape, np.unique(y), len(np.unique(y)))
        file_name = 'subject_'+ str(i)+'.dat'
        saved_datafilename = 'data/WISDM/'+ file_name    
        with open(saved_datafilename,'wb') as f:
            cp.dump((x,y),f)
        
    return data_x, data_y


def load_datafile(filename, columns):
    if not os.path.exists(filename):
        print("file is not exits {}".format(filename))
        return

    df = pd.read_csv(filename, header = 0, names = columns)
    df=df.dropna()
    
    data_x = df[columns].to_numpy()
    nanCount = np.isnan(data_x).sum()
    print(nanCount," Nan values")
    replaceNan(data_x)
    
    return data_x

columns = ['Local_Time', 'M_x', 'M_y', 'M_z', 'A_x', 'A_y', 'A_z', 'G_x', 'G_y', 'G_z', 'Cap']

n_population = 1
n_session = 6 # 5 or 6 based on the number of sessions person did in the video
idx_list = list(range(n_population))
person = 1 #change as per person
input_path = f'./Recording{person}/'

for num in idx_list:
    #filename_left = input_path + 'Star_P' + str(num+1) + '_Left.csv'
    #filename_right = input_path + 'Star_P' + str(num+1) + '_Right.csv'
    filename_left = input_path + 'Star_P' + str(person) + '_Left.csv'
    filename_right = input_path + 'Star_P' + str(person) + '_Right.csv'
    data_left = load_datafile(filename_left, columns)
    data_right = load_datafile(filename_right, columns)

    columns_label = ['Label', 'Left_start', 'Left_end', 'Right_start', 'Right_end']
    error = 0
    for session_num in range(n_session):
        #filename_label = input_path + 'Star_Label_P' + str(num+1) + '_S' + str(session_num+1) + '.csv'
        filename_label = input_path + 'Star_Label_P' + str(person) + '_Session' + str(session_num+1) + '.csv'
        label = load_datafile(filename_label, columns_label)
        
        #print(len(label))
        idx_left = 0
        idx_right = 0
        merge_data = np.empty(shape=[0,21])
        try:
            for label_idx in range(len(label)):
                bStart_left = False
                bEnd_left = False
                bStart_right = False
                bEnd_right = False
                #print(label_idx)
                while(bEnd_left==False or bEnd_right==False):
                    
                    if data_left[idx_left][0]/1000 >= label[label_idx][1] and data_left[idx_left][0]/1000 <= label[label_idx][2]:
                        bStart_left = True
                        temp_data = data_left[idx_left][1:]
                        if bStart_right:
                            idx_left += 1
                    else:
                        if bStart_left:
                            if bEnd_left == False:
                                bEnd_left = True
                                idx_left+=1
                        else:
                            if label_idx > 0:
                                temp_data = data_left[idx_left][1:]

                            idx_left+=1
                    
                    if data_right[idx_right][0]/1000 >= label[label_idx][3] and data_right[idx_right][0]/1000 <= label[label_idx][4]:
                        bStart_right = True
                        if bStart_left:
                            if len(temp_data) == 10:
                                temp_data = np.concatenate((temp_data, data_right[idx_right][1:], label[label_idx][0]), None)
                                merge_data = np.vstack((merge_data, temp_data))
                            idx_right += 1
                    else:
                        if bStart_right:
                            if bEnd_right == False:
                                bEnd_right = True
                                idx_right+=1
                        else:
                            if label_idx > 0:
                                temp_data = np.concatenate((temp_data, data_right[idx_right][1:], 0), None)
                                merge_data = np.vstack((merge_data, temp_data))
                            idx_right+=1
                
            filename_session = input_path + 'Star_P' + str(person) +"_Session"+str(session_num+1) + '.npy'
            
            print(merge_data.shape)

            with open(filename_session, 'wb') as f:
                np.save(f, merge_data)
        except IndexError as error:
            # Output expected IndexErrors.
            print(f"{error} in session {session_num+1}")
        # with open(filename_session,'wb') as f:
            # cp.dump(merge_data,f)