#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 10:18:31 2019

@author: eleftherios
"""
import keras
import numpy as np

from keras.models import Model
from keras.preprocessing import image
from keras.layers import Input
from matplotlib.pyplot import imshow
import glob
from sklearn import svm
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle
from sklearn.preprocessing import normalize,scale
from sklearn.metrics import roc_auc_score, roc_curve, auc
from pandas import read_table as rt
import pickle as pkl

#selection:
#database
database_name = "mias"
#model selection
model_name="nasnet"

if model_name == "inc3":
    from keras.applications.inception_v3 import InceptionV3
    from keras.applications.inception_v3 import preprocess_input
elif model_name == "vgg":
    from keras.applications.vgg19 import VGG19
    from keras.applications.vgg19 import preprocess_input
elif model_name == "incr2":
    from keras.applications.inception_resnet_v2 import InceptionResNetV2
    from keras.applications.inception_resnet_v2 import preprocess_input
elif model_name == "densenet":
    from keras.applications.densenet import DenseNet201
    from keras.applications.densenet import preprocess_input
elif model_name == "nasnet":
    from keras.applications.nasnet import NASNetLarge
    from keras.applications.nasnet import preprocess_input

#load pre-trained model
input_tensor = Input(shape=(725, 234, 3)) 

if model_name == "inc3":
    pretrained_model = InceptionV3(input_tensor=input_tensor, include_top = False, weights = "imagenet", pooling = 'avg')
elif model_name == "vgg":
    pretrained_model = VGG19(input_tensor=input_tensor, include_top = False, weights = "imagenet", pooling = 'avg') 
elif model_name == "incr2":
    pretrained_model = InceptionResNetV2(input_tensor=input_tensor, include_top = False, weights = "imagenet", pooling = 'avg')
elif model_name == "densenet":
    pretrained_model = DenseNet201(input_tensor=input_tensor, include_top = False, weights = "imagenet", pooling = 'avg')
elif model_name == "nasnet":
    pretrained_model = NASNetLarge(input_tensor=input_tensor, include_top = False, weights = "imagenet", pooling = 'avg')

model = Model(inputs = pretrained_model.input,
                     outputs = pretrained_model.output)

# i.e. freeze all convolutional InceptionV3 layers
for layer in pretrained_model.layers:
    layer.trainable = False

ds_dir = "dataset/"

if database_name == "ddsm":
    img_dir = "DDSM/"
#    lbl_path= "Main_MLO_Data.txt"
    lbl_path= "labels.txt"
    
    with open("dataset/DDSM/labels.txt", "rb") as file:
        labels_df=rt(file)
        
    labels_df=labels_df.drop(2287)# class 0
    labels_df=labels_df.drop(2288)# class 0
    
    labels = dict(labels_df.values)
    pids = np.array(list(labels_df['file_name'].values))
    label_values = np.array(list(labels_df['label'].values))
    
    #remove zero class and re-value other classes
    label=label_values.tolist()
    for index in range(0,len(label)):
        #for DDSM 0 is one false subject
        #1 can be zero
        #2 can be zero
        # 3 & 4 can be 1
        if label[index] == 1:
            label_values[index]=0
            labels[pids[index]]=0
        elif label[index] == 2:
            label_values[index]=0
            labels[pids[index]]=0
        elif label[index] == 3:
            label_values[index]=1
            labels[pids[index]]=1
        elif label[index] == 4:
            label_values[index]=1
            labels[pids[index]]=1
        else:
            print(label[index])
            print(pids[index])
            print(index)
elif database_name == "mias":
    with open("dataset/Breast/labels.pkl", "rb") as file:
        labels=pkl.load(file)
    pids = np.array(list(labels.keys()))
    label_values = np.array(list(labels.values()))
    #miniMIAS
    img_dir = "Breast/"
#    lbl_path = "miniMIAS labels.txt"
    
    #remove zero class and re-value other classes
    label=label_values.tolist()
    for index in range(0,len(label)):
        #for MIAS 0 is zero
        #1 can be zero
        #2 can be one
        if label[index] == 1:
            label_values[index]=0
            labels[pids[index]]=0
        elif label[index] == 2:
            label_values[index]=1
            labels[pids[index]]=1
#        else:#0 is zero
#            print(label[index])
#            print(pids[index])
#            print(index)      

print("MIN L: "+str(label_values.min()))
print("MAX L: "+str(label_values.max()))


#load images
deep_features=[]
for img_path in pids:
    if database_name == "ddsm":
        img = image.load_img(ds_dir+img_dir+img_path, target_size=(725, 234))
        x = image.img_to_array(img)
    elif database_name == "mias":
        x = np.load(ds_dir+img_dir+img_path+".npy")
        x = np.concatenate((x,x,x),axis=2)#color channel same img, requirement of transfer learning to have 3 color channels
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    #infer
    deep_features.append(model.predict(x))

deep_features_raw,img_labels=shuffle(deep_features,list(label_values))

#feature preprocessing
if model_name == "inc3":
    deep_features=np.array(deep_features_raw).reshape(-1,2048)#INc3
elif model_name == "vgg":
    deep_features=np.array(deep_features_raw).reshape(-1,512)#VGG19
elif model_name == "incr2":
    deep_features=np.array(deep_features_raw).reshape(-1,1536)#InceptionResNetV2 #[:512]
elif model_name == "densenet":
    deep_features=np.array(deep_features_raw).reshape(-1,1920)#DenseNet201
elif model_name == "nasnet":
    deep_features=np.array(deep_features_raw).reshape(-1,4032)#NASNetLarge
    
#feature normalization    
deep_features = scale(deep_features,with_mean=True,with_std=True)#the best
#deep_features = normalize(deep_features)#big no-no

#labels: list to npy 
img_labels=np.array(img_labels)

#classification
performance = []
roc_auc_avg = []
auc_avg = []
index=0
skfold = StratifiedKFold(n_splits=5, random_state=None, shuffle=True)
for train_index,test_index in skfold.split(deep_features,img_labels):
    if model_name == "incr2":
        X_train, X_test = deep_features[train_index][:,:512],deep_features[test_index][:,:512]#INC2
    else:
        X_train, X_test = deep_features[train_index],deep_features[test_index]
    y_train, y_test = img_labels[train_index], img_labels[test_index]
    
    clf = svm.SVC(kernel='rbf', gamma='scale', probability=True)#poly, linear, rbf
    clf.fit(X_train, y_train)  
#    clf.fit(X_train, y_train)  
    score = clf.score(X_test,y_test)
    predictions = clf.predict_proba(X_test)
    
    roc = roc_auc_score(y_test,predictions[:,1],average='weighted')        
    fpr, tpr, thresholds = roc_curve(y_test,predictions[:,1])#, pos_label=1)
    auc_score = auc(fpr, tpr)
            
    index+=1
    print("Acc of fold "+str(index)+": "+str(score*100)+"%")
    print("Roc of fold "+str(index)+": "+str(roc*100)+"%")
#    print("Auc of fold "+str(index)+": "+str(auc_score*100)+"%")
    
    performance.append(score)    
    roc_auc_avg.append(roc)
    auc_avg.append(auc_score)
    
#final avg metrics
fold_mean_acc = np.array(performance).mean()
fold_mean_roc = np.array(roc_auc_avg).mean()
fold_mean_auc = np.array(auc_avg).mean()
print("Average Acc: "+str(fold_mean_acc*100)+"%")
print("Average Roc: "+str(fold_mean_roc*100)+"%")
print("Average Auc: "+str(fold_mean_auc*100)+"%")