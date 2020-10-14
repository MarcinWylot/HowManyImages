
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle

import tensorflow
from tensorflow.keras import metrics
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

import models

print('GPU', tensorflow.config.experimental.list_physical_devices('GPU'))
physical_devices = tensorflow.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tensorflow.config.experimental.set_memory_growth(physical_devices[0], True)


epochs = 500
batch_size = 50
shape = (224, 224)


def pickel_logs(log_entry):
    log_file = 'logs.pickle'
    try:
        with open(log_file, 'rb') as f:
            logs = pickle.load(f)
    except:
        logs = []

    logs.append(log_entry)

    with open(log_file, 'wb') as f:
        pickle.dump(logs, f, pickle.HIGHEST_PROTOCOL)


def image_generator(preprocessing_function):
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocessing_function,
        rotation_range=40,
        width_shift_range=0.3,
        height_shift_range=0.4,
        shear_range=0.2,
        zoom_range=0.4,
        horizontal_flip=True
    )

    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocessing_function)

    return train_datagen, test_datagen


def prepate_dataset(train_size=100, val_size=250, test_size=600):
    data = []
    directory = '/tf/work/data/chest-xray-pneumonia/'
    for folder in sorted(os.listdir(directory)):
        for file in sorted(os.listdir(directory+folder)):
            data.append((folder, file, directory+folder+'/'+file))

    dataframe = pd.DataFrame(data, columns=['label', 'file', 'path'])

    if train_size+val_size+test_size >= len(dataframe):
        print('not enough data fot the train_size, val_size, test_size:',
              train_size, val_size, test_size)

    dataframe.loc[dataframe.label == 'NORMAL', 'label'] = '0_negative'
    dataframe.loc[dataframe.label == 'PNEUMONIA', 'label'] = '1_positive'

    df, test_df = train_test_split(
        dataframe, test_size=test_size, stratify=dataframe.label, random_state=9)
    train_df, val_df = train_test_split(
        df, train_size=train_size, test_size=val_size, stratify=df.label, random_state=9)

    return train_df, val_df, test_df


def compute_weithts(df):
    pos = len(df[df.label == '1_positive'])
    total = len(df)
    neg = total - pos
    weight_for_0 = (1 / neg)*(total)/2.0
    weight_for_1 = (1 / pos)*(total)/2.0

    class_weight = {0: weight_for_0, 1: weight_for_1}

    initial_bias = np.log([pos/neg])

    return initial_bias, class_weight


def train_val_test(train_dataframe, val_dataframe, test_dataframe, train_datagen, test_datagen):
    train_generator = train_datagen.flow_from_dataframe(
        train_dataframe,
        directory=None,
        x_col="path",
        y_col="label",
        batch_size=batch_size,
        class_mode='binary',
        target_size=shape
    )

    val_generator = test_datagen.flow_from_dataframe(
        val_dataframe,
        directory=None,
        x_col="path",
        y_col="label",
        batch_size=1,
        class_mode='binary',
        target_size=shape,
        shuffle=False
    )

    test_generator = test_datagen.flow_from_dataframe(
        test_dataframe,
        directory=None,
        x_col="path",
        y_col="label",
        batch_size=1,
        class_mode='binary',
        target_size=shape,
        shuffle=False
    )

    return train_generator, val_generator, test_generator


def make_model(model):
    # https://www.tensorflow.org/tutorials/structured_data/imbalanced_data
    METRICS = [
        metrics.TruePositives(name='tp'),
        metrics.FalsePositives(name='fp'),
        metrics.TrueNegatives(name='tn'),
        metrics.FalseNegatives(name='fn'),
        metrics.BinaryAccuracy(name='binary_accuracy'),
        metrics.Precision(name='precision'),
        metrics.Recall(name='recall'),
        metrics.AUC(name='auc'),
    ]

    optimizer = Adam(learning_rate=0.0001)

    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy', metrics=METRICS)

    return model


def train(model, train_generator, val_generator, test_generator, class_weight):
    history = model.fit_generator(
        generator=train_generator,
        validation_data=val_generator,
        steps_per_epoch=int(train_generator.samples/batch_size),
        epochs=epochs,
        validation_steps=int(val_generator.samples/1),
        verbose=1,
        callbacks=None,  # [es],
        class_weight=class_weight
    )

    evaluate = model.evaluate_generator(
        generator=test_generator, steps=int(test_generator.samples/1), verbose=0)

    return history, evaluate


def show_metrics(history, evaluate, train_size):
    print('-'*80)
    print('Model: ', history.model.name)
    print('Test size: ', train_size)
    for name, value in zip(history.model.metrics_names, evaluate):
        print(name, ': ', value)

    print('-'*80)
