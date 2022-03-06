import random
from DataBank import *
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.applications import ResNet101V2, ResNet152V2
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score
import gc
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
import pandas as pd
import time


def train_model(model_type, species_folder, augmentation_amount, seed, call_order, INPUT_SHAPE, 
    sample_size, save_results_folder, EPOCHS):

    # Get the training data
    db = DataBank(species_folder, 0.30, augmentation_amount, 'pow')
    db.set_seed(seed)
    X_train, Y_train, data_distribution, train_distribution, new_seed = db.get_data(sample_size,  'roar', call_order)

    if model_type == 'ResNet152V2':
        base_model = ResNet152V2(weights="imagenet",
            input_shape=INPUT_SHAPE,
            include_top=False)  
        
    if model_type == 'ResNet101V2':
        base_model = ResNet101V2(weights="imagenet",
            input_shape=INPUT_SHAPE,
            include_top=False)  
    
    # Freeze the base_model
    base_model.trainable = False

    # Create new model on top
    inputs = Input(shape=INPUT_SHAPE)

    # The base model contains batchnorm layers. We want to keep them in inference mode
    # when we unfreeze the base model for fine-tuning, so we make sure that the
    # base_model is running in inference mode here.
    x = base_model(inputs)

    x = Flatten()(x)
    outputs = Dense(2, activation='softmax')(x)
    model = Model(inputs, outputs)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(),metrics=['accuracy'])

    print (model.summary())

    filepath=save_results_folder+'/'+"{}-{}_NoFineTuned.hdf5".format(seed, model_type)
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    
    history = model.fit(X_train,Y_train, 
                        validation_split=0.2,epochs=EPOCHS, 
                        callbacks=callbacks_list)
    
    model = tf.keras.models.load_model(filepath)
    
    tf.keras.backend.clear_session()
    gc.collect()
    
    return model

