

```python
"""

@author: Eleftherios Trivizakis
@github: github.com/trivizakis

"""
```

Import custom libraries for data handling & deep learning


```python
from dataset import DataConverter
from data_augmentation import DataAugmentation
from data_generator import DataGenerator
from model import CustomModel as Model
import numpy as np
from utils import Utils
from sklearn.model_selection import StratifiedShuffleSplit
from pathlib import Path
import pickle as pkl
from keras import backend as K

import warnings
warnings.filterwarnings("ignore")
```

    Using TensorFlow backend.
    

Import dataset & labels


```python
hyperparameters = Utils.get_hypes()

my_file = Path(hyperparameters["dataset_dir"] + "labels.pkl")
if not my_file.is_file():
    DataConverter(hyperparameters).convert_png_to_npy()
    
with open(hyperparameters["dataset_dir"]+"labels.pkl", "rb") as file:
    labels = pkl.load(file)

pids = np.load(hyperparameters["dataset_dir"] + "pids.npy")
label_values = np.load(hyperparameters["dataset_dir"] + "labels.npy")
```

Initialization kfold or hold-out


```python
tst_split_index = 1
val_split_index = 1
#skf_tr_tst = StratifiedKFold(n_splits=hyperparameters["kfold"][0],shuffle=hyperparameters["shuffle"])
skf_tr_val = StratifiedShuffleSplit(n_splits=hyperparameters["kfold"][0], test_size=0.2, train_size=0.8)
values = np.array(list(labels.values()))
```

Import data, augmentation, model fitting 


```python
#for trval_index, tst_index in skf_tr_tst.split(pids,values):
convergence_pids = pids#[trval_index]
convergence_labels = values#[trval_index]
for tr_index, val_index in skf_tr_val.split(convergence_pids,convergence_labels):
    K.clear_session()
    
    #network version
    version = str(tst_split_index)+"."+str(val_split_index)
    hyperparameters["version"] = "network_version"+version+"\\"
    
    #make dirs        
    Utils.make_dirs(version,hyperparameters)        
    
    #save patient id per network version
    Utils.save_skf_pids(version,convergence_labels[tr_index],convergence_pids[val_index],convergence_pids[val_index],hyperparameters)#pids[tst_index],hyperparameters)        
    
    #offline data augmentation
    if hyperparameters["offline_augmentation"] == True and hyperparameters["data_augmentation"] == True:
        training_set, training_labels = DataConverter(hyperparameters).load_npy_from_hdd(convergence_pids[tr_index], labels)
        training_set, training_labels = DataAugmentation.apply_augmentation(training_set, training_labels,hyperparameters)
        pids_tr, labels_tr=DataConverter(hyperparameters).save_augmented_samples(training_set, training_labels,convergence_pids[tr_index],convergence_labels[tr_index])
    else:
        pids_tr = convergence_pids[tr_index]
        labels_tr = labels
            
    # Generators
    training_generator = DataGenerator(pids_tr, labels_tr, hyperparameters, training=True)
    validation_generator = DataGenerator(convergence_pids[val_index], labels, hyperparameters, training=False)
#    testing_generator = DataGenerator(pids[tst_index], labels, hyperparameters, training=False)
    
    #create network
    cnn = Model.get_model(hyperparameters)
    
    #fit network
    cnn = Model.train_model(cnn,hyperparameters,training_generator,validation_generator)
    Utils.save_hypes(hyperparameters["chkp_dir"]+hyperparameters["version"], "hypes"+version, hyperparameters)
    
    #Update version indeces
    val_split_index+=1
val_split_index=1
tst_split_index+=1
```

    WARNING: Logging before flag parsing goes to stderr.
    W0922 23:10:12.742325  1220 deprecation_wrapper.py:119] From C:\Users\Lefteris\Anaconda3\envs\deep_learning\lib\site-packages\keras\backend\tensorflow_backend.py:95: The name tf.reset_default_graph is deprecated. Please use tf.compat.v1.reset_default_graph instead.
    
    W0922 23:10:12.742325  1220 deprecation_wrapper.py:119] From C:\Users\Lefteris\Anaconda3\envs\deep_learning\lib\site-packages\keras\backend\tensorflow_backend.py:98: The name tf.placeholder_with_default is deprecated. Please use tf.compat.v1.placeholder_with_default instead.
    
    W0922 23:10:12.757950  1220 deprecation_wrapper.py:119] From C:\Users\Lefteris\Anaconda3\envs\deep_learning\lib\site-packages\keras\backend\tensorflow_backend.py:102: The name tf.get_default_graph is deprecated. Please use tf.compat.v1.get_default_graph instead.
    
    W0922 23:10:12.773574  1220 deprecation_wrapper.py:119] From C:\Users\Lefteris\Anaconda3\envs\deep_learning\lib\site-packages\keras\backend\tensorflow_backend.py:517: The name tf.placeholder is deprecated. Please use tf.compat.v1.placeholder instead.
    
    W0922 23:10:12.773574  1220 deprecation_wrapper.py:119] From C:\Users\Lefteris\Anaconda3\envs\deep_learning\lib\site-packages\keras\backend\tensorflow_backend.py:4138: The name tf.random_uniform is deprecated. Please use tf.random.uniform instead.
    
    W0922 23:10:15.441256  1220 deprecation_wrapper.py:119] From C:\Users\Lefteris\Anaconda3\envs\deep_learning\lib\site-packages\keras\backend\tensorflow_backend.py:1834: The name tf.nn.fused_batch_norm is deprecated. Please use tf.compat.v1.nn.fused_batch_norm instead.
    
    W0922 23:10:16.757139  1220 deprecation_wrapper.py:119] From C:\Users\Lefteris\Anaconda3\envs\deep_learning\lib\site-packages\keras\backend\tensorflow_backend.py:3980: The name tf.nn.avg_pool is deprecated. Please use tf.nn.avg_pool2d instead.
    
    W0922 23:10:16.819638  1220 deprecation.py:506] From C:\Users\Lefteris\Anaconda3\envs\deep_learning\lib\site-packages\keras\backend\tensorflow_backend.py:3445: calling dropout (from tensorflow.python.ops.nn_ops) with keep_prob is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use `rate` instead of `keep_prob`. Rate should be set to `rate = 1 - keep_prob`.
    W0922 23:10:17.054013  1220 deprecation_wrapper.py:119] From C:\Users\Lefteris\Anaconda3\envs\deep_learning\lib\site-packages\keras\optimizers.py:790: The name tf.train.Optimizer is deprecated. Please use tf.compat.v1.train.Optimizer instead.
    
    

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv0 (Conv2D)               (None, 150, 150, 64)      1792      
    _________________________________________________________________
    conv_bn0 (BatchNormalization (None, 150, 150, 64)      256       
    _________________________________________________________________
    conv0_out (Activation)       (None, 150, 150, 64)      0         
    _________________________________________________________________
    conv1 (Conv2D)               (None, 75, 75, 64)        36928     
    _________________________________________________________________
    conv_bn1 (BatchNormalization (None, 75, 75, 64)        256       
    _________________________________________________________________
    conv1_out (Activation)       (None, 75, 75, 64)        0         
    _________________________________________________________________
    conv2 (Conv2D)               (None, 75, 75, 128)       73856     
    _________________________________________________________________
    conv_bn2 (BatchNormalization (None, 75, 75, 128)       512       
    _________________________________________________________________
    conv2_out (Activation)       (None, 75, 75, 128)       0         
    _________________________________________________________________
    conv3 (Conv2D)               (None, 38, 38, 128)       147584    
    _________________________________________________________________
    conv_bn3 (BatchNormalization (None, 38, 38, 128)       512       
    _________________________________________________________________
    conv3_out (Activation)       (None, 38, 38, 128)       0         
    _________________________________________________________________
    conv4 (Conv2D)               (None, 19, 19, 256)       295168    
    _________________________________________________________________
    conv_bn4 (BatchNormalization (None, 19, 19, 256)       1024      
    _________________________________________________________________
    conv4_out (Activation)       (None, 19, 19, 256)       0         
    _________________________________________________________________
    conv5 (Conv2D)               (None, 10, 10, 512)       1180160   
    _________________________________________________________________
    conv_bn5 (BatchNormalization (None, 10, 10, 512)       2048      
    _________________________________________________________________
    conv5_out (Activation)       (None, 10, 10, 512)       0         
    _________________________________________________________________
    AvgPooling2D (AveragePooling (None, 5, 5, 512)         0         
    _________________________________________________________________
    flatten_layer (Flatten)      (None, 12800)             0         
    _________________________________________________________________
    fc0 (Dense)                  (None, 1024)              13108224  
    _________________________________________________________________
    drop0 (Dropout)              (None, 1024)              0         
    _________________________________________________________________
    fc_bn0 (BatchNormalization)  (None, 1024)              4096      
    _________________________________________________________________
    softmax_layer (Dense)        (None, 8)                 8200      
    =================================================================
    Total params: 14,860,616
    Trainable params: 14,856,264
    Non-trainable params: 4,352
    _________________________________________________________________
    None
    

    W0922 23:10:18.562630  1220 deprecation.py:323] From C:\Users\Lefteris\Anaconda3\envs\deep_learning\lib\site-packages\tensorflow\python\ops\math_grad.py:1250: add_dispatch_support.<locals>.wrapper (from tensorflow.python.ops.array_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use tf.where in 2.0, which has the same broadcast rule as np.where
    W0922 23:10:22.680984  1220 deprecation_wrapper.py:119] From C:\Users\Lefteris\Anaconda3\envs\deep_learning\lib\site-packages\keras\callbacks.py:850: The name tf.summary.merge_all is deprecated. Please use tf.compat.v1.summary.merge_all instead.
    
    W0922 23:10:22.680984  1220 deprecation_wrapper.py:119] From C:\Users\Lefteris\Anaconda3\envs\deep_learning\lib\site-packages\keras\callbacks.py:853: The name tf.summary.FileWriter is deprecated. Please use tf.compat.v1.summary.FileWriter instead.
    
    

    Epoch 1/2000
    133/133 [==============================] - 26s 192ms/step - loss: 1.8085 - acc: 0.6160 - val_loss: 2.2239 - val_acc: 0.5414
    Epoch 2/2000
    133/133 [==============================] - 21s 160ms/step - loss: 1.4435 - acc: 0.7556 - val_loss: 1.9042 - val_acc: 0.5838
    Epoch 3/2000
    133/133 [==============================] - 21s 160ms/step - loss: 1.3096 - acc: 0.8073 - val_loss: 1.8790 - val_acc: 0.6434
    Epoch 4/2000
    133/133 [==============================] - 21s 161ms/step - loss: 1.1774 - acc: 0.8687 - val_loss: 1.3014 - val_acc: 0.8081
    Epoch 5/2000
    133/133 [==============================] - 22s 163ms/step - loss: 1.0948 - acc: 0.9035 - val_loss: 1.4678 - val_acc: 0.7071
    Epoch 6/2000
    133/133 [==============================] - 21s 161ms/step - loss: 1.0173 - acc: 0.9368 - val_loss: 1.3695 - val_acc: 0.7444
    Epoch 7/2000
    133/133 [==============================] - 21s 160ms/step - loss: 0.9322 - acc: 0.9659 - val_loss: 1.1767 - val_acc: 0.8667
    Epoch 8/2000
    133/133 [==============================] - 22s 166ms/step - loss: 0.8848 - acc: 0.9779 - val_loss: 1.1575 - val_acc: 0.8636
    Epoch 9/2000
    133/133 [==============================] - 22s 162ms/step - loss: 0.8374 - acc: 0.9890 - val_loss: 1.2189 - val_acc: 0.8253
    Epoch 10/2000
    133/133 [==============================] - 22s 164ms/step - loss: 0.7965 - acc: 0.9930 - val_loss: 1.2227 - val_acc: 0.8343
    Epoch 11/2000
    133/133 [==============================] - 22s 162ms/step - loss: 0.7762 - acc: 0.9937 - val_loss: 1.2358 - val_acc: 0.8101
    Epoch 12/2000
    133/133 [==============================] - 21s 161ms/step - loss: 0.7548 - acc: 0.9952 - val_loss: 1.1884 - val_acc: 0.8333
    


```python

```
