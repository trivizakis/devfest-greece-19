# -*- coding: utf-8 -*-
"""

@author: Lefteris Trivizakis
@github: github.com/trivizakis

"""

import keras.backend as K
import numpy as np

from keras.models import Model
from keras.layers import Input
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input

from sklearn import svm
from sklearn.utils import shuffle
from sklearn.preprocessing import scale
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report

from dataset import DataConverter

import warnings
warnings.filterwarnings("ignore")

#directories in root
ds_path = "dataset\\persona\\"

#load pre-trained model
input_shape = (256, 256, 3)
input_tensor = Input(input_shape) 

pretrained_model = VGG19(input_tensor=input_tensor, include_top = False, weights = "imagenet", pooling = 'avg')

# i.e. freeze all convolutional layers
for layer in pretrained_model.layers:
    layer.trainable = False
    
model = Model(inputs = pretrained_model.input,
                     outputs = pretrained_model.output)
hypes={}
hypes["dataset_dir"]=ds_path
hypes["input_shape"]=input_shape
hypes["image_normalization"]="-11"
#DataConverter(hypes).convert_png_to_npy()
pids = np.load(ds_path+"pids.npy")
labels = np.load(ds_path+"labels.npy")
class_names = ["aigis","morgana","takamaki","teddie"]
#load images
deep_features_raw=[]
for pid in pids:
    img = np.load(ds_path+pid+".npy")
    img = np.expand_dims(img, axis=0)
    
    #infer
    deep_features_raw.append(model.predict(img))

deep_features=np.array(deep_features_raw).reshape(-1,1920)

#feature normalization    
deep_features = scale(deep_features,with_mean=True,with_std=True)#the best
performance=[]
n_splits=5
sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.2, train_size=0.8)
for training_indeces, testing_indeces in sss.split(pids,labels):
    X_train, X_test = deep_features[training_indeces],deep_features[testing_indeces]
    y_train, y_test = labels[training_indeces], labels[testing_indeces]
    
    clf = svm.SVC(kernel='linear', gamma='scale', probability=True)#poly, linear, rbf
    clf.fit(X_train, y_train)
    predictions = clf.predict_proba(X_test)
#    y_pred = np.argmax(predictions)
#    cm=confusion_matrix(y_test,y_pred)
    
    performance.append(clf.score(X_test,y_test))
scores = np.stack(performance)
print("Acc: "+str(scores.mean()*100)+"%")
