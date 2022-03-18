import glob
import os
import numpy as np
import copy as copy
from tqdm import tqdm
import pickle
from sklearn import mixture
import tensorflow as tf
from sklearn.model_selection import train_test_split
import collections
from collections import Counter
import operator

from manager.feature_extractor import *

model_save_path = "./model"
RANDOM_SEED = 42
buff = collections.deque(maxlen=10)
nb_classe=8

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

#hmm_models
def mean_classes(C,classes_list):
    buff.append(C)
    # list = [buff.count(classe) for classe in classes_list]
    list_sorted =  Counter(buff)
    # max_key = max(list_sorted.iteritems(), key=operator.itemgetter(1))[0]
    max_key = max(list_sorted.items(), key=operator.itemgetter(1))[0]
    return max_key
    # if(classe_max/len(buff) > )

def arrange_data_for_FCNN(train_arrays):
    data = np.empty((0,42))
    for i in range(train_arrays.shape[0]): data=np.append(data,train_arrays[i], axis=0)
    return data#sum([train_arrays[i].shape[0] for i in range(train_arrays.shape[0])])

def train_model_FCNN(X_dataset, y_dataset):
    
    X_train, X_test, y_train, y_test = train_test_split(X_dataset, y_dataset, train_size=0.75, random_state=RANDOM_SEED)
    fcnn = fcnn_model()
    print(fcnn.summary())

    cp_callback = tf.keras.callbacks.ModelCheckpoint(model_save_path, verbose=1, save_weights_only=False)# Model checkpoint callback
    es_callback = tf.keras.callbacks.EarlyStopping(patience=20, verbose=1)# Callback for early stopping
        
    fcnn.compile(# Model compilation
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )    

    fcnn.fit(
        X_train, np.array(y_train),
        epochs=500,
        batch_size=32,
        validation_data=(X_test, np.array(y_test)),
        callbacks=[cp_callback, es_callback]
    )
    
    return fcnn

#     from sklearn.model_selection import RepeatedKFold, cross_val_score
# from tensorflow.keras.models import * 
# from tensorflow.keras.layers import * 
# from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

# def buildmodel():
#     model= Sequential([
#         Dense(10, activation="relu"),
#         Dense(5, activation="relu"),
#         Dense(1)
#     ])
#     model.compile(optimizer='adam', loss='mse', metrics=['mse'])
#     return(model)

# estimator= KerasRegressor(build_fn=buildmodel, epochs=100, batch_size=10, verbose=0)
# kfold= RepeatedKFold(n_splits=5, n_repeats=100)
# results= cross_val_score(estimator, x, y, cv=kfold, n_jobs=2)  # 2 cpus
# results.mean()  # Mean MSE


def predict_model_FCNN(fcnn, data_test):
    probs=fcnn.predict(np.array(data_test).reshape((1,42)))
    predict_classes=np.argmax(probs,axis=1)
    return predict_classes

def fcnn_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input((21 * 2, )),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(20, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(nb_classe, activation='softmax')
        ])
    return model
