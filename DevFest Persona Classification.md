

```python
# -*- coding: utf-8 -*-
"""

@author: Eleftherios Trivizakis
@github: github.com/trivizakis

"""
```

We import the basic libraries for the deep learning component.
We will utilize a pre-trained DenseNet201 as an "off-the-shelf" feature extractor.


```python
import keras.backend as K#K.clear_session()
import numpy as np

from keras.models import Model
from keras.layers import Input
from keras.applications.vgg19 import VGG19
```

We import scikit-learn libraries for the classification component of
this totaly unscientific experiment ;)
We will use:

-"StratifiedShuffleSplit" for a random hold-out split strategy (training &  testing set)

-"scale" for raw feature scaling with values between -1 to 1

-"Support Vector Machine" as a classifier


```python
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import scale
from sklearn import svm
```

"DataConverter" is a custom library for converting *.png images to *.npy


```python
from dataset import DataConverter
from pathlib import Path
```

Remove some terminal warnings for this experiment


```python
import warnings
warnings.filterwarnings("ignore")
```

Define the dataset directory


```python
ds_path = "dataset\\persona\\"
```

-Initialize the Input layer. Since we do not use the fully-connected part of the network
we could append any input size fits our problem

-Get the ImageNet acquired weights for DenseNet

-Freeze the convolutional layers, useful when you apply fine-tuning

-Initialize the deep feature extraction Model


```python
input_shape = (256, 256, 3)
input_tensor = Input(input_shape) 

pretrained_model = VGG19(input_tensor=input_tensor, include_top = False, weights = "imagenet", pooling = 'avg')

# i.e. freeze all convolutional layers
#for layer in pretrained_model.layers:
#    layer.trainable = False
    
model = Model(inputs = pretrained_model.input,
                     outputs = pretrained_model.output)
print("Model created.")
```

    Model created.
    

Initialize the parameters dictionary for
converting the png images to npy and then
import them as numpy arrays


```python
hypes={}
hypes["dataset_dir"]=ds_path
hypes["input_shape"]=input_shape
hypes["image_normalization"]="-11"

my_file = Path(hypes["dataset_dir"]+"pids.npy")
if not my_file.is_file():
    DataConverter(hypes).convert_png_to_npy()
    
print("npy files generated.")
pids = np.load(ds_path+"pids.npy")
labels = np.load(ds_path+"labels.npy")

class_names = ["aigis","morgana","takamaki","teddie"]
```

    npy files generated.
    

Extraction of the deep features


```python
deep_features_raw=[]
for pid in pids:
    img = np.load(ds_path+pid+".npy")
    img = np.expand_dims(img, axis=0)    
    
    deep_features_raw.append(model.predict(img))
```

Reshape feature maps to a feature vector


```python
deep_features=np.array(deep_features_raw).reshape(-1,512)
```

Feature vector normalization


```python
deep_features = scale(deep_features,with_mean=True,with_std=True)
print("Deep features extracted and scaled.")
```

    Deep features extracted and scaled.
    

-Hold-out Stratification

-Fitting the classier on the training set using a svm with linear kernel

-Calculate performance on testing set for each random split


```python
performance=[]
n_splits=5
sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.2, train_size=0.8)
for training_indeces, testing_indeces in sss.split(pids,labels):
    X_train, X_test = deep_features[training_indeces],deep_features[testing_indeces]
    y_train, y_test = labels[training_indeces], labels[testing_indeces]
    clf = svm.SVC(kernel='linear', gamma='scale', probability=True)#poly, linear, rbf
    clf.fit(X_train, y_train)
    performance.append(clf.score(X_test,y_test))
```

Calculate the mean accuracy


```python
scores = np.stack(performance)
print("Mean Acc: "+str(scores.mean()*100)+"%")
```

    Mean Acc: 96.92307692307693%
    


```python

```
