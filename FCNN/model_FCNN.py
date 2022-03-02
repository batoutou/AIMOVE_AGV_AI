import glob
import os
import numpy as np
import copy as copy
from tqdm import tqdm
import pickle
from sklearn import mixture
import tensorflow as tf
from sklearn.model_selection import train_test_split

from manager.feature_extractor import *

model_save_path = "./model"
RANDOM_SEED = 42

nb_classe=2

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

#hmm_models

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
        epochs=100,
        batch_size=256,
        validation_data=(X_test, np.array(y_test)),
        callbacks=[cp_callback, es_callback]
    )
    return fcnn

def predict_model_FCNN(fcnn, data_test):
    #probs = fcnn.predict_proba(data_test)

    probs=fcnn.predict(data_test)
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
