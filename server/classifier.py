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


def getData(dataFiles):
    data = [] #processed and normalised with pose_idx
    for file in dataFiles:
        data_path = dataDir+file
        print(data_path)
        mat = scipy.io.loadmat(data_path)
        mat.pop("__header__")
        mat.pop("__version__")
        mat.pop("__globals__")
        i=0

        plot_interval = 100
        plot_index = 0

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
def loadData(img_w_h_len):
    data_train_nm = get_nm_data(dataFiles_train)
    data_test_nm = get_nm_data(dataFiles_test)
    w = img_w_h_len
    h = img_w_h_len


    # dataset_train = data_train_nm
    dataset_test = data_test_nm

    val_test_mask = np.random.rand(len(data_train_nm)) < 0.5

    dataset_train = data_train_nm[val_test_mask]
    dataset_val = data_train_nm[~val_test_mask]

    n = int(img_w_h_len * img_w_h_len / 2)

    dataset_train_ar = reshape_data(n,dataset_train)
    dataset_test_ar = reshape_data(n,dataset_test)
    dataset_val_ar = reshape_data(n,dataset_val)
    return {'w':w,'h':h,'dataset_train_ar':dataset_train_ar,'dataset_test_ar':dataset_test_ar,'dataset_val_ar':dataset_val_ar,'data_train_nm':data_train_nm,'data_test_nm':data_test_nm}

def getX_y():
    X = split_X(dataset_train_ar)

    X_val = split_X(dataset_val_ar)
    y_val = split_y(dataset_val_ar)

    X_test = split_X(dataset_test_ar)
    y_test = split_y(dataset_test_ar)


    # OPTIONAL TURN X TO NxN
    X = reshape_arr(X)
    X_test = reshape_arr(X_test)
    X_val = reshape_arr(X_val)




    X = reshape_for_conv2d(X,w,h)
    X_test = reshape_for_conv2d(X_test,w,h)
    X_val = reshape_for_conv2d(X_val,w,h)
    return {'X':X,"X_val":X_val,'y_val':y_val,'X_test':X_test,'y_test':y_test}
#=====================================
def getPredSample(test_sample):
    test_X_sample_rate = np.array_split(X_test,len(y_test)/(test_sample))
    test_y_sample_rate = np.array_split(y_test,len(y_test)/(test_sample))
    print(len(test_X_sample_rate))
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
    print(my_base64_jpgData)
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
    port="/dev/rfcomm0"
    bluetooth = serial.Serial(port=port,   baudrate=9600)
    return {'port':port,'bluetooth':bluetooth}
# BLUETOOTH COM
def bluetoothCommand(pred):
    pose  = str(pred) + ";"
    # bluetooth.write(bytes(pose,   'utf-8'))
    time.sleep(1.2)
    print(pose)
#====================================
global cur_sample_idx
cur_sample_idx = 0
global pred_rate
# pred_rate = []
b = True
def predict_plot():
    # if(b):
    #     BlueToothConnect()
    #     b = False
    global cur_sample_idx
    global pred_rate
    plot_as_base64 = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAoAAAAHgCAYAAAA10dzkAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjguMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/SrBM8AAAACXBIWXMAAA9hAAAPYQGoP6dpAAAIyUlEQVR4nO3WMQEAIAzAMMC/5+ECjiYKenbPzCwAADLO7wAAAN4ygAAAMQYQACDGAAIAxBhAAIAYAwgAEGMAAQBiDCAAQIwBBACIMYAAADEGEAAgxgACAMQYQACAGAMIABBjAAEAYgwgAECMAQQAiDGAAAAxBhAAIMYAAgDEGEAAgBgDCAAQYwABAGIMIABAjAEEAIgxgAAAMQYQACDGAAIAxBhAAIAYAwgAEGMAAQBiDCAAQIwBBACIMYAAADEGEAAgxgACAMQYQACAGAMIABBjAAEAYgwgAECMAQQAiDGAAAAxBhAAIMYAAgDEGEAAgBgDCAAQYwABAGIMIABAjAEEAIgxgAAAMQYQACDGAAIAxBhAAIAYAwgAEGMAAQBiDCAAQIwBBACIMYAAADEGEAAgxgACAMQYQACAGAMIABBjAAEAYgwgAECMAQQAiDGAAAAxBhAAIMYAAgDEGEAAgBgDCAAQYwABAGIMIABAjAEEAIgxgAAAMQYQACDGAAIAxBhAAIAYAwgAEGMAAQBiDCAAQIwBBACIMYAAADEGEAAgxgACAMQYQACAGAMIABBjAAEAYgwgAECMAQQAiDGAAAAxBhAAIMYAAgDEGEAAgBgDCAAQYwABAGIMIABAjAEEAIgxgAAAMQYQACDGAAIAxBhAAIAYAwgAEGMAAQBiDCAAQIwBBACIMYAAADEGEAAgxgACAMQYQACAGAMIABBjAAEAYgwgAECMAQQAiDGAAAAxBhAAIMYAAgDEGEAAgBgDCAAQYwABAGIMIABAjAEEAIgxgAAAMQYQACDGAAIAxBhAAIAYAwgAEGMAAQBiDCAAQIwBBACIMYAAADEGEAAgxgACAMQYQACAGAMIABBjAAEAYgwgAECMAQQAiDGAAAAxBhAAIMYAAgDEGEAAgBgDCAAQYwABAGIMIABAjAEEAIgxgAAAMQYQACDGAAIAxBhAAIAYAwgAEGMAAQBiDCAAQIwBBACIMYAAADEGEAAgxgACAMQYQACAGAMIABBjAAEAYgwgAECMAQQAiDGAAAAxBhAAIMYAAgDEGEAAgBgDCAAQYwABAGIMIABAjAEEAIgxgAAAMQYQACDGAAIAxBhAAIAYAwgAEGMAAQBiDCAAQIwBBACIMYAAADEGEAAgxgACAMQYQACAGAMIABBjAAEAYgwgAECMAQQAiDGAAAAxBhAAIMYAAgDEGEAAgBgDCAAQYwABAGIMIABAjAEEAIgxgAAAMQYQACDGAAIAxBhAAIAYAwgAEGMAAQBiDCAAQIwBBACIMYAAADEGEAAgxgACAMQYQACAGAMIABBjAAEAYgwgAECMAQQAiDGAAAAxBhAAIMYAAgDEGEAAgBgDCAAQYwABAGIMIABAjAEEAIgxgAAAMQYQACDGAAIAxBhAAIAYAwgAEGMAAQBiDCAAQIwBBACIMYAAADEGEAAgxgACAMQYQACAGAMIABBjAAEAYgwgAECMAQQAiDGAAAAxBhAAIMYAAgDEGEAAgBgDCAAQYwABAGIMIABAjAEEAIgxgAAAMQYQACDGAAIAxBhAAIAYAwgAEGMAAQBiDCAAQIwBBACIMYAAADEGEAAgxgACAMQYQACAGAMIABBjAAEAYgwgAECMAQQAiDGAAAAxBhAAIMYAAgDEGEAAgBgDCAAQYwABAGIMIABAjAEEAIgxgAAAMQYQACDGAAIAxBhAAIAYAwgAEGMAAQBiDCAAQIwBBACIMYAAADEGEAAgxgACAMQYQACAGAMIABBjAAEAYgwgAECMAQQAiDGAAAAxBhAAIMYAAgDEGEAAgBgDCAAQYwABAGIMIABAjAEEAIgxgAAAMQYQACDGAAIAxBhAAIAYAwgAEGMAAQBiDCAAQIwBBACIMYAAADEGEAAgxgACAMQYQACAGAMIABBjAAEAYgwgAECMAQQAiDGAAAAxBhAAIMYAAgDEGEAAgBgDCAAQYwABAGIMIABAjAEEAIgxgAAAMQYQACDGAAIAxBhAAIAYAwgAEGMAAQBiDCAAQIwBBACIMYAAADEGEAAgxgACAMQYQACAGAMIABBjAAEAYgwgAECMAQQAiDGAAAAxBhAAIMYAAgDEGEAAgBgDCAAQYwABAGIMIABAjAEEAIgxgAAAMQYQACDGAAIAxBhAAIAYAwgAEGMAAQBiDCAAQIwBBACIMYAAADEGEAAgxgACAMQYQACAGAMIABBjAAEAYgwgAECMAQQAiDGAAAAxBhAA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" 
    if(cur_sample_idx <len(test_X_sample_rate)):
        # for X_test_rate in test_X_sample_rate:
        X_test_rate = test_X_sample_rate[cur_sample_idx]
        plot_as_base64=drawPlotAlongSample(X_test_rate)

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

        cur_sample_idx += 1
    else:
        cur_sample_idx = 0
    # return pred_rate[-1]
        # print(plot_as_base64)
    return {'pred':pred_rate,'base64':plot_as_base64}

# def predict_plot():
#     global pred_rate
#     for X_test_rate in test_X_sample_rate:
#         drawPlotAlongSample(X_test_rate)

#         preds_1 = load_model_1.predict(X_test_rate)
#         preds_2 = load_model_2.predict(X_test_rate)
#         preds_3 = load_model_3.predict(X_test_rate)

#         pred_c = []
#         pred_c.append(preds_1)
#         pred_c.append(preds_2)
#         pred_c.append(preds_3)
#         pred_c = np.array(pred_c)


#         preds_label_c = []
#         for preds in pred_c:
#             pred_label = []
#             for prediction in preds:
#                 pred_label.append(np.argmax(prediction))
#             pred_label = np.array(pred_label)
#             preds_label_c.append(pred_label)
        

#         preds_label_c = acquire_pred_for_each(pred_c)
        
#         pred = predFromMaxPred(X_test_rate,preds_label_c)
#         bluetoothCommand(pred)
#         pred_rate.append(pred)
#     pred_rate = np.array(pred_rate)

#====================================
def initClassifier():
    global dataset_train_ar
    global dataset_test_ar
    global dataset_val_ar
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
    loadedData = loadData(img_w_h_len=40)
    w = loadedData['w']
    h = loadedData['h']
    dataset_train_ar = loadedData['dataset_train_ar']
    dataset_test_ar = loadedData['dataset_test_ar']
    dataset_val_ar = loadedData['dataset_val_ar']
    data_train_nm = loadedData['data_train_nm']
    data_test_nm = loadedData['data_test_nm']

    X_Y = getX_y()
    X = X_Y['X']
    X_test = X_Y['X_test']
    X_val = X_Y['X_val']
    y_test = X_Y['y_test']
    y_val = X_Y['y_val']
#====================================
# loadedData = loadData(img_w_h_len=40)
# w = loadedData['w']
# h = loadedData['h']
# dataset_train_ar = loadedData['dataset_train_ar']
# dataset_test_ar = loadedData['dataset_test_ar']
# dataset_val_ar = loadedData['dataset_val_ar']
# data_train_nm = loadedData['data_train_nm']
# data_test_nm = loadedData['data_test_nm']

# X_Y = getX_y()
# X = X_Y['X']
# X_test = X_Y['X_test']
# X_val = X_Y['X_val']
# y_test = X_Y['y_test']
# y_val = X_Y['y_val']

initClassifier()
load_model_1 = keras.models.load_model('/home/james/Documents/cyber_arm_fin/arm-cyberware-full/NN_dev/models/tuned_model_loss_0.7_0.7_acc_new_10_11.keras')
load_model_2 = keras.models.load_model('/home/james/Documents/cyber_arm_fin/arm-cyberware-full/NN_dev/models/tuned_model_loss_5.0_0.1_acc_us_15_11.keras')
# load_model_3 = model
load_model_3 = keras.models.load_model('/home/james/Documents/cyber_arm_fin/arm-cyberware-full/NN_dev/models/tuned_model_loss_2_0.4_acc_new_9_11.keras')
load_model_4 = load_model_1
# load_model_4 = keras.models.load_model('./models/tuned_model_loss_2_0.4_acc_new_9_11.keras')
# load_model_3 = keras.models.load_model('./models/model_0.4_40l_1c_new.keras')


PredSample = getPredSample(test_sample=1)
test_X_sample_rate = PredSample['test_X_sample_rate']
test_y_sample_rate = PredSample['test_y_sample_rate']


# plot_path = "/home/james/Documents/cyber_arm_fin/arm-cyberware-full/app/assets/signal.png"

# t = predict_plot()