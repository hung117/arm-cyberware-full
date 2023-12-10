import base64
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from matplotlib import colors
import matplotlib
import tensorflow as tf
import cv2
import keras_tuner
import keras
import pandas as pd
from tensorflow.keras import layers
from keras import metrics
from sklearn import preprocessing
import math  
import datetime
import tensorflow_decision_forests as tfdf
import os
from keras.utils import plot_model
import tensorflow_datasets as tfds
from keras.utils import plot_model
import pydot
import serial
import time
import difflib
from sklearn.metrics.pairwise import cosine_similarity
import io

# import graphviz
global plot_path
global w 
global h 
global dataDir
global dataFiles
global dataFiles_train 
global dataFiles_test 
global e

global data_train_nm
global dataset_train_ar
global dataset_test_ar
global dataset_val_ar

#/////////////==========================
dataDir = "/media/james/Datasets_Drive1/semg_for_basic_hand_movment_6/Database_1/"
dataFiles=['female_1.mat','female_2.mat','female_3.mat','male_1.mat','male_2.mat']
dataFiles_train = ['female_1.mat','male_1.mat','male_2.mat']
dataFiles_test = ['female_2.mat','female_3.mat']
# dataFiles=['female_1.mat']
e = 2.718281828459045
#/////////////==========================
def normalize_arr(arr,i):
    signal = arr.copy()
    signal = signal.tolist()
    signal.append(int(i/2))
    signal = np.array(signal)
    return signal

def get_channel_pair(chn1,chn2,i):
    paired_data = []
    label = int(i/2)
    chn1 =  list(np.delete(chn1,0,1))
    chn2 =  list(np.delete(chn2,0,1))

    for chunk1 in chn1:
        chunk2 = chn2[i].T
        chunk1 = chunk1.T

        while(len(chunk1)>0):
            row=[chunk1[0],chunk2[0],label]
            paired_data.append(row)

            chunk1 = list(chunk1)
            chunk2 = list(chunk2)
            chunk1.pop(0)
            chunk2.pop(0)

    return paired_data

def getData(dataFile):
    data = [] #processed and normalised with pose_idx
    data_path = dataDir+dataFile
    print(data_path)
    mat = scipy.io.loadmat(data_path)
    mat.pop("__header__")
    mat.pop("__version__")
    mat.pop("__globals__")
    i=0

    for channel in mat: 
    
        if(i%2==0):
            channel2 =  channel[:-1]
            channel2 += '2'
            sigs1 = mat[channel] 
            sigs2 = mat[channel2]
            sigs1_norm = []
            sigs2_norm = []

            for signal in sigs1:
                # signal = normalize_arr(signal,i)
                if(signal[0]<2):
                    sigs1_norm.append(signal)
                    sigs2_norm.append(signal)
            # for signal in sigs2:
            #     # signal = normalize_arr(signal,i)
            #     if(signal[0]<2):
            #         sigs2_norm.append(signal)

            if i==0:
                data = get_channel_pair(sigs1_norm,sigs2_norm,i)
            else:
                None
                data_lc = get_channel_pair(sigs1_norm,sigs2_norm,i)
                data += data_lc
        i+=1
    return data
# def getData(dataFiles):
#     data = [] #processed and normalised with pose_idx
#     for file in dataFiles:
#         data_path = dataDir+file
#         print(data_path)
#         mat = scipy.io.loadmat(data_path)
#         mat.pop("__header__")
#         mat.pop("__version__")
#         mat.pop("__globals__")
#         i=0

#         plot_interval = 100
#         plot_index = 0

#         for channel in mat: 
        
#             if(i%2==0):
#                 channel2 =  channel[:-1]
#                 channel2 += '2'
#                 sigs1 = mat[channel] 
#                 sigs2 = mat[channel2]
#                 sigs1_norm = []
#                 sigs2_norm = []

#                 for signal in sigs1:
#                     # signal = normalize_arr(signal,i)
#                     if(signal[0]<2):
#                         sigs1_norm.append(signal)
#                         sigs2_norm.append(signal)
#                 # for signal in sigs2:
#                 #     # signal = normalize_arr(signal,i)
#                 #     if(signal[0]<2):
#                 #         sigs2_norm.append(signal)

#                 if i==0:
#                     data = get_channel_pair(sigs1_norm,sigs2_norm,i)
#                 else:
#                     None
#                     data_lc = get_channel_pair(sigs1_norm,sigs2_norm,i)
#                     data += data_lc
#             i+=1
#     return data
def get_nm_data(dataFiles):
    data = getData(dataFiles)
    print(np.array(data).shape)
    print(data[-3])

    data_nm = np.copy(data)[:,:2]
    data_nm.shape
    nm_c1 =  preprocessing.normalize([data_nm.T[0]]).T.flatten()
    nm_c2 =  preprocessing.normalize([data_nm.T[1]]).T.flatten()
    df_data_nm = pd.DataFrame(np.vstack((nm_c1,nm_c2)).T, columns = ['channel1','channel2'])

    label_arr=[]

    for row in data:
        label_arr.append(int(row[2]))
    print(label_arr)

    data_nm = df_data_nm.assign(label=label_arr)
    return data_nm



#==========================================
def GetSpacedElements(array, numElems = 4):
    return array[np.round(np.linspace(0, len(array)-1, numElems)).astype(int)]

#////////////////////============
def display_signals(b):
    w = 20
    h = 20
    fig = plt.figure(figsize=(10, 10))
    columns = 1
    rows = 2

    for index in range(1,2+1):
        idx = 0
        fig.add_subplot(rows, columns, index)
        for signal in b:
            plt.plot(signal[index-1],label=str(idx))
            idx += 1
        plt.title('channel '+str(index))
        plt.legend(ncol=1)
        
            
    plt.show()
    fig = plt.figure(figsize=(20, 20))
    columns = 2
    rows = 3
    idx = 0

    for signal in b:
        fig.add_subplot(rows, columns, idx+1)
        plt.plot(signal[0],color='blue',label=str(0))
        plt.plot(signal[1],color='orange',label=str(1))
        idx += 1
        plt.title('pose '+str(idx-1))
        plt.legend(ncol=1)
            
    plt.show()


def reshape_data(n,df_data):
    # get n channel 1 and n channel 2 into 1 sample
    chn1 = df_data['channel1'].to_numpy()
    chn2 = df_data['channel2'].to_numpy()
    label = df_data['label'].to_numpy()
    i = 0
    newShape_data = []
    while i < chn1.size:
        #currently  the size is 540000, tack batch 20 signals, the loop will rn 27000 times
        reshape_data = np.concatenate((chn1[i:i+n], chn2[i:i+n]))
        tmp = reshape_data.tolist()
        tmp.append(int(label[i]))
        tmp = np.array(tmp)

        if(len(reshape_data)==n*2):
            newShape_data.append(tmp)
        i += n

    newShape_data = np.array(newShape_data)
    return newShape_data

def split_X(arr):
    arr = arr.T
    arr = arr[0:-1]
    arr = arr.T
    return arr
def split_y(arr):
    arr = arr.T
    arr = arr[-1]
    arr = arr.T.astype(np.uint8)
    return arr
def reshape_arr(arr):
    new_shape = []
    for row in arr:
        n_len =int(math.sqrt(len(row)))
        row = row.reshape(n_len,n_len)
        new_shape.append(row)
    return np.array(new_shape)
def reshape_arr_img_transfer(arr,n):
    new_shape = []
    for row in arr:
        n_len = len(row)
        # row =np.pad(row,(0,n-n_len),'symmetric')
        row =np.pad(row,(0,n-n_len),'constant',constant_values=10)
        row = row.reshape(n,n)
        new_shape.append(row)
    return np.array(new_shape)
def signal_to_3channelimg(arr):
    rgb_batch_ts =  np.repeat(arr[..., np.newaxis], 3, -1)
    rgb_batch_ts.shape
    return rgb_batch_ts
def reshape_for_conv2d(t,w,h):
    train_size = int(len(t) * len(t[0])* len(t[0][0]) / (w*h))
    reshaped_tuple = t.reshape(train_size, w,h, 1)
    return reshaped_tuple
def get_groups_of_data(dataFiles):# from multiple paths
    data_test_nm = pd.DataFrame({'A' : []})
    for dataFile in dataFiles:
        cur_data_test_nm = get_nm_data(dataFile)
        if(data_test_nm.empty):
            data_test_nm=cur_data_test_nm
        else:
            data_test_nm=pd.concat([data_test_nm,cur_data_test_nm])
    return data_test_nm
def loadData(img_w_h_len):
    # dataFiles_test[0]
    data_user_test0 = get_nm_data(dataFiles_test[0])
    data_user_test1 = get_nm_data(dataFiles_test[1])

    # data_train_nm = get_nm_data(dataFiles_train)
    # data_test_nm = get_nm_data(dataFiles_test)

    data_train_nm = get_groups_of_data(dataFiles_train)
    data_test_nm = get_groups_of_data(dataFiles_test)

    w = img_w_h_len
    h = img_w_h_len


    # dataset_train = data_train_nm
    dataset_test = data_test_nm

    val_test_mask = np.random.rand(len(data_train_nm)) < 0.5

    dataset_train = data_train_nm[val_test_mask]
    dataset_val = data_train_nm[~val_test_mask]

    n = int(img_w_h_len * img_w_h_len / 2)
    data_user_test0_ar = reshape_data(n,data_user_test0)
    data_user_test1_ar = reshape_data(n,data_user_test1)
    dataset_train_ar = reshape_data(n,dataset_train)
    dataset_test_ar = reshape_data(n,dataset_test)
    dataset_val_ar = reshape_data(n,dataset_val)

    return {'w':w,'h':h,'dataset_train_ar':dataset_train_ar,'dataset_test_ar':dataset_test_ar,'dataset_val_ar':dataset_val_ar,'data_train_nm':data_train_nm,'data_test_nm':data_test_nm,'data_user_test0':data_user_test0,'data_user_test1':data_user_test1,'data_user_test0_ar':data_user_test0_ar,'data_user_test1_ar':data_user_test1_ar}
global data_user_test0_ar
global data_user_test1_ar


def getX_y():
    X = split_X(dataset_train_ar)

    X_val = split_X(dataset_val_ar)
    y_val = split_y(dataset_val_ar)

    X_test = split_X(dataset_test_ar)
    y_test = split_y(dataset_test_ar)

    X_user0_test = split_X(data_user_test0_ar)
    y_user0_test = split_y(data_user_test0_ar)

    X_user1_test = split_X(data_user_test1_ar)
    y_user1_test = split_y(data_user_test1_ar)


    # OPTIONAL TURN X TO NxN
    X = reshape_arr(X)
    X_test = reshape_arr(X_test)
    X_val = reshape_arr(X_val)

    X_user0_test = reshape_arr(X_user0_test)
    X_user1_test = reshape_arr(X_user1_test)

    X = reshape_for_conv2d(X,w,h)
    X_test = reshape_for_conv2d(X_test,w,h)
    X_val = reshape_for_conv2d(X_val,w,h)

    X_user0_test = reshape_for_conv2d(X_user0_test,w,h)
    X_user1_test = reshape_for_conv2d(X_user1_test,w,h)

    return {'X':X,"X_val":X_val,'y_val':y_val,'X_test':X_test,'y_test':y_test,
            'X_user0_test':X_user0_test,'X_user1_test':X_user1_test,
            'y_user0_test':y_user0_test,'y_user1_test':y_user1_test}
#=====================================
def getPredSample(test_sample,X_test_data,y_test_data):
    test_X_sample_rate = np.array_split(X_test_data,len(y_test_data)/(test_sample))
    test_y_sample_rate = np.array_split(y_test_data,len(y_test_data)/(test_sample))
    np.array(test_X_sample_rate[1]).shape
    tempt_y = []
    for ys in test_y_sample_rate:
        tempt_y.append(ys[0])
    test_y_sample_rate = tempt_y
    test_y_sample_rate = np.array(test_y_sample_rate)
    test_y_sample_rate
    return {'test_X_sample_rate':test_X_sample_rate,'test_y_sample_rate':test_y_sample_rate}
def predFromMaxPred(X_test_rate,preds_label_c):
    ensemble_pred = []
    for i in np.arange(len(X_test_rate)):
        i=i
        arr = []
        for pred in preds_label_c:
            arr.append(pred[i])
        ensemble_pred.append(max_voting(arr))
    return max_voting(ensemble_pred)
def acquire_pred_for_each(pred_c):
    preds_label_c = []
    for preds in pred_c:
        pred_label = []
        for prediction in preds:
            pred_label.append(np.argmax(prediction))
        pred_label = np.array(pred_label)
        preds_label_c.append(pred_label)
    preds_label_c = np.array(preds_label_c)
    return preds_label_c
matplotlib.use('Agg')
def drawPlotAlongSample(X_test_rate):
    plt.ioff()

    XFlat = X_test_rate.flatten()
    sample = 50 # max 200 for clean code sake
    rate = sample *2
    XFlat = GetSpacedElements(XFlat,rate)
    XFlat1=XFlat[0:int(len(XFlat)/2)]
    XFlat2=XFlat[int(len(XFlat)/2):-1]
    global plot_path
    global cur_sample_idx

    # plot_path = '/home/james/Documents/cyber_arm_fin/arm-cyberware-full/app/assets/signal${cur_sample_idx}.png'
    plot_path = '/home/james/Documents/cyber_arm_fin/arm-cyberware-full/app/assets/signal.png'

    plt.plot(XFlat1,color='blue',label=str(0))
    plt.plot(XFlat2,color='orange',label=str(1))
    plt.title("sEmg signal")
    plt.legend(ncol=1)  
    # plt.savefig(plot_path)
    my_stringIObytes = io.BytesIO()
    plt.savefig(my_stringIObytes, format='jpg')
    my_stringIObytes.seek(0)
    my_base64_jpgData = base64.b64encode(my_stringIObytes.read()).decode()
    plt.pause(0.001)
    plt.clf()
    plt.close()
    # print(my_base64_jpgData)
    return my_base64_jpgData
#====================================
def max_voting(List):
    counter = 0
    num = List[0]
     
    for i in List:
        curr_frequency = List.count(i)
        if(curr_frequency> counter):
            counter = curr_frequency
            num = i

    return num

# BLUETOOTH var define
global port
global bluetooth
def BlueToothConnect():
    global bluetooth
    try:
        port="/dev/rfcomm0"
        bluetooth = serial.Serial(port=port,   baudrate=9600)
        # return {'port':port,'bluetooth':bluetooth}
    except:
        pass

# BLUETOOTH COM
def bluetoothCommand(pred):
    global bluetooth
    pose  = str(pred) + ";"
    try:
        bluetooth.write(bytes(pose,   'utf-8'))
    except:
        pass
    time.sleep(1.2)
    print(pose)
#====================================
global cur_sample_idx
cur_sample_idx = 0
global pred_rate
# pred_rate = []
global b_once
b_once = True
def predict_plot(cur_sample_idx,wanted_pose,test_user):
    # pose = 5+
    global b_once
    sample_idx_from=0
    sample_idx_to=0
    if(test_user==0):
        test_X_sample_rate=batch_testX_user0
    elif(test_user==1):
        test_X_sample_rate=batch_testX_user1
    else:
        test_X_sample_rate=batch_testX

    if(wanted_pose>5):
        lim_sample_idx=len(test_X_sample_rate)
        start_idx = 0
        b_once = True
    else:
        match wanted_pose:
            case 0:
                sample_idx_from = 000
                sample_idx_to = 111
            case 1:
                sample_idx_from = 113
                sample_idx_to = 222
            case 2:
                sample_idx_from = 222
                sample_idx_to = 334
            case 3:
                sample_idx_from = 334
                sample_idx_to = 446
            case 4:
                sample_idx_from = 447
                sample_idx_to = 558
            case 5:
                sample_idx_from = 559
                sample_idx_to = -1
        if(b_once):
            cur_sample_idx = sample_idx_from
            b_once=False
        lim_sample_idx = sample_idx_to
        start_idx = sample_idx_from

    
    if(cur_sample_idx > lim_sample_idx):
        cur_sample_idx = start_idx
    
    X_test_rate = test_X_sample_rate[cur_sample_idx]
    base64=drawPlotAlongSample(X_test_rate)
    if(base64!=None):
        plot_as_base64 = base64
    preds_1 = load_model_1.predict(X_test_rate)
    preds_2 = load_model_2.predict(X_test_rate)
    preds_3 = load_model_3.predict(X_test_rate)

    pred_c = []
    pred_c.append(preds_1)
    pred_c.append(preds_2)
    pred_c.append(preds_3)
    pred_c = np.array(pred_c)


    preds_label_c = []
    for preds in pred_c:
        pred_label = []
        for prediction in preds:
            pred_label.append(np.argmax(prediction))
        pred_label = np.array(pred_label)
        preds_label_c.append(pred_label)
    

    preds_label_c = acquire_pred_for_each(pred_c)
    
    pred = predFromMaxPred(X_test_rate,preds_label_c)
    bluetoothCommand(pred)
    # pred_rate.append(pred)
    pred_rate = np.array(pred)

    return {'pred':pred_rate,'base64':plot_as_base64}
def pose_by_selection(requested_pose):
    bluetoothCommand(requested_pose)

global y_user1_test
global X_user1_test
global y_user0_test
global X_user0_test
#====================================
def initClassifier():
    global dataset_train_ar
    global dataset_test_ar
    global dataset_val_ar

    global data_user_test0_ar
    global data_user_test1_ar

    global data_train_nm
    global data_train_nm
    global data_test_nm
    global w 
    global h 
    global X 
    global X_test 
    global X_val
    global y_test 
    global y_val 

    global y_user1_test
    global X_user1_test
    global y_user0_test
    global X_user0_test

    loadedData = loadData(img_w_h_len=40)
    w = loadedData['w']
    h = loadedData['h']
    dataset_train_ar = loadedData['dataset_train_ar']
    dataset_test_ar = loadedData['dataset_test_ar']
    dataset_val_ar = loadedData['dataset_val_ar']
    data_train_nm = loadedData['data_train_nm']
    data_test_nm = loadedData['data_test_nm']
    data_user_test0_ar = loadedData['data_user_test0_ar']
    data_user_test1_ar = loadedData['data_user_test1_ar']
    X_Y = getX_y()
    X = X_Y['X']
    X_test = X_Y['X_test']
    X_val = X_Y['X_val']
    y_test = X_Y['y_test']
    y_val = X_Y['y_val']

    X_user0_test = X_Y['X_user0_test']
    X_user1_test = X_Y['X_user1_test']
    y_user0_test = X_Y['y_user0_test']
    y_user1_test = X_Y['y_user1_test']

    
#====================================

initClassifier()
load_model_1 = keras.models.load_model('/home/james/Documents/cyber_arm_fin/arm-cyberware-full/NN_dev/models/tuned_model_loss_0.7_0.7_acc_new_10_11.keras')
load_model_2 = keras.models.load_model('/home/james/Documents/cyber_arm_fin/arm-cyberware-full/NN_dev/models/tuned_model_loss_5.0_0.1_acc_us_15_11.keras')
# load_model_3 = model
load_model_3 = keras.models.load_model('/home/james/Documents/cyber_arm_fin/arm-cyberware-full/NN_dev/models/tuned_model_loss_2_0.4_acc_new_9_11.keras')
load_model_4 = load_model_1
# load_model_4 = keras.models.load_model('./models/tuned_model_loss_2_0.4_acc_new_9_11.keras')
# load_model_3 = keras.models.load_model('./models/model_0.4_40l_1c_new.keras')


PredSample = getPredSample(test_sample=1,X_test_data=X_test,y_test_data=y_test)
batch_testX = PredSample['test_X_sample_rate']
batch_testy = PredSample['test_y_sample_rate']
PredSample = getPredSample(test_sample=1,X_test_data=X_user0_test,y_test_data=y_user0_test)
batch_testX_user0 = PredSample['test_X_sample_rate']
batch_testy_user0 = PredSample['test_y_sample_rate']
PredSample = getPredSample(test_sample=1,X_test_data=X_user1_test,y_test_data=y_user1_test)
batch_testX_user1 = PredSample['test_X_sample_rate']
batch_testy_user1 = PredSample['test_y_sample_rate']


# plot_path = "/home/james/Documents/cyber_arm_fin/arm-cyberware-full/app/assets/signal.png"

# t = predict_plot()