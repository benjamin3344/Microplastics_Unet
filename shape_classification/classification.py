import os
from keras.models import Model
from keras.optimizers import Adam
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense, Dropout, Flatten
from pathlib import Path
import numpy as np
from livelossplot.inputs.keras import PlotLossesCallback
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import splitfolders
import shutil
import pickle
import sys

from datasets import copyTree

import keras.backend as K

log = open("training.log", "w")
sys.stdout = log

def create_model(input_shape, n_classes, optimizer='rmsprop', fine_tune=0):
    """
    Compiles a model integrated with VGG16 pretrained layers

    input_shape: tuple - the shape of input images (width, height, channels)
    n_classes: int - number of classes for the output layer
    optimizer: string - instantiated optimizer to use for training. Defaults to 'RMSProp'
    fine_tune: int - The number of pre-trained layers to unfreeze.
                If set to 0, all pretrained layers will freeze during training
    """

    # Pretrained convolutional layers are loaded using the Imagenet weights.
    # Include_top is set to False, in order to exclude the model's fully-connected layers.
    conv_base = VGG16(include_top=False,
                      weights='imagenet',
                      input_shape=input_shape)

    # Defines how many layers to freeze during training.
    # Layers in the convolutional base are switched from trainable to non-trainable
    # depending on the size of the fine-tuning parameter.
    if fine_tune > 0:
        for layer in conv_base.layers[:-fine_tune]:
            layer.trainable = False
    else:
        for layer in conv_base.layers:
            layer.trainable = False

    # Create a new 'top' of the model (i.e. fully-connected layers).
    # This is 'bootstrapping' a new top_model onto the pretrained layers.
    top_model = conv_base.output
    top_model = Flatten(name="flatten")(top_model)
    # top_model = Dense(4096, activation='relu')(top_model)
    # top_model = Dense(1072, activation='relu')(top_model)
    top_model = Dense(64, activation='relu')(top_model)
    top_model = Dense(32, activation='relu')(top_model)
    top_model = Dropout(0.2)(top_model)
    output_layer = Dense(n_classes, activation='softmax')(top_model)

    # Group the convolutional base and new fully-connected layers into a Model object.
    model = Model(inputs=conv_base.input, outputs=output_layer)

    # Compiles the model for training.
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

def plot_heatmap(y_true, y_pred, class_names, ax, title):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(
        cm,
        annot=True,
        square=True,
        xticklabels=class_names,
        yticklabels=class_names,
        fmt='d',
        cmap=plt.cm.Blues,
        cbar=False,
        ax=ax
    )
    ax.set_title(title, fontsize=16)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)

#
# def f1_score(y_true, y_pred):
#     y_true = K.cast(y_true, dtype = 'float32')
#     y_pred = K.cast(y_pred, dtype = 'float32')
#     # Count positive samples.
#     c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
#     c3 = K.sum(K.round(K.clip(y_true, 0, 1)))
#     # If there are no true samples, fix the F1 score at 0.
#     if c3 == 0:
#         return 0.
#     # How many selected items are relevant?
#     precision = 1. * c1 / c2
#     # How many relevant items are selected?
#     recall = 1. * c1 / c3
#     # Calculate f1_score
#     f1_score = 2. * (precision * recall) / (precision + recall)
#     return f1_score
#
# def precision(y_true, y_pred):
#     y_true = K.cast(y_true, dtype = 'float32')
#     y_pred = K.cast(y_pred, dtype = 'float32')
#     # Count positive samples.
#     c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
#     c3 = K.sum(K.round(K.clip(y_true, 0, 1)))
#     # If there are no true samples, fix the F1 score at 0.
#     if c3 == 0:
#         return 0.
#     # How many selected items are relevant?
#     precision = 1. * c1 / c2
#     return precision
#
# def recall(y_true, y_pred):
#     y_true = K.cast(y_true, dtype = 'float32')
#     y_pred = K.cast(y_pred, dtype = 'float32')
#     # Count positive samples.
#     c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
#     c3 = K.sum(K.round(K.clip(y_true, 0, 1)))
#     # If there are no true samples, fix the F1 score at 0.
#     if c3 == 0:
#         return 0.
#     # How many relevant items are selected?
#     recall = 1. * c1 / c3
#     return recall


# scratch_model = load_model('s')
BATCH_SIZE = 8

mode = '5-cross'
if mode == '5-cross':
    print('good')
    # shutil.rmtree('data/microplastics/train/')
    # shutil.rmtree('data/microplastics/val/')
    # shutil.rmtree('data/microplastics/test/')
    splitfolders.ratio('data/microplastics/images', output="data/microplastics/", seed=2, ratio=(0.2, 0.2, 0.6))
    shutil.move('data/microplastics/train/', 'data/microplastics/1/')
    shutil.move('data/microplastics/val/', 'data/microplastics/2/')
    shutil.move('data/microplastics/test/', 'data/microplastics/6/')
    splitfolders.ratio('data/microplastics/6/', output="data/microplastics/", seed=2, ratio=(0.33, 0.33, 0.34))
    shutil.move('data/microplastics/train/', 'data/microplastics/3/')
    shutil.move('data/microplastics/val/', 'data/microplastics/4/')
    shutil.move('data/microplastics/test/', 'data/microplastics/5/')

for crossval_i in range(5):  ## KEYPOINT3
    if os.path.isdir('data/microplastics/train/') == True:
        shutil.rmtree('data/microplastics/train/')
        shutil.rmtree('data/microplastics/test/')
    os.mkdir('data/microplastics/train/')

    list_dir = ['data/microplastics/1/', 'data/microplastics/2/', 'data/microplastics/3/', 'data/microplastics/4/',
                'data/microplastics/5/']
    copyTree(list_dir[crossval_i], 'data/microplastics/test/')
    del list_dir[crossval_i]
    for path in list_dir:
        copyTree(path, 'data/microplastics/train/')
    print('training-testing spliting finished for {}/5 cross validation'.format(crossval_i + 1))
    print(mode)

    train_generator = ImageDataGenerator(rotation_range=90,
                                         width_shift_range=0.2,
                                         height_shift_range=0.2,
                                         shear_range=0.2,
                                         zoom_range=0.2,
                                         horizontal_flip=True,
                                         vertical_flip=True,
                                         validation_split=0.2,
                                         fill_mode='reflect',
                                         preprocessing_function=preprocess_input)

    test_generator = ImageDataGenerator(preprocessing_function=preprocess_input)  # VGG16 preprocessing

    train_data_dir = 'data/microplastics/train'
    test_data_dir = 'data/microplastics/test'

    class_subset = sorted(os.listdir('data/microplastics/images'))[:3]  # Using only the first 3 classes

    traingen = train_generator.flow_from_directory(train_data_dir,
                                                   target_size=(224, 224),
                                                   class_mode='categorical',
                                                   classes=class_subset,
                                                   subset='training',
                                                   batch_size=BATCH_SIZE,
                                                   shuffle=True,
                                                   seed=5)

    validgen = train_generator.flow_from_directory(train_data_dir,
                                                   target_size=(224, 224),
                                                   class_mode='categorical',
                                                   classes=class_subset,
                                                   subset='validation',
                                                   batch_size=BATCH_SIZE,
                                                   shuffle=True,
                                                   seed=5)

    testgen = test_generator.flow_from_directory(test_data_dir,
                                                 target_size=(224, 224),
                                                 class_mode=None,
                                                 classes=class_subset,
                                                 batch_size=1,
                                                 shuffle=False,
                                                 seed=5)





    input_shape = (224, 224, 3)
    optim_1 = Adam(learning_rate=0.001)
    n_classes = 3

    # n_steps = 5 * traingen.samples // BATCH_SIZE
    # n_val_steps = 5 * validgen.samples // BATCH_SIZE
    n_steps = 1 * traingen.n // BATCH_SIZE
    n_val_steps = 1 * validgen.n// BATCH_SIZE
    n_epochs = 50

    # First we'll train the model without Fine-tuning
    # vgg_model = create_model(input_shape, n_classes, optim_1, fine_tune=0)
    # plot_loss_1 = PlotLossesCallback()

    # Use a smaller learning rate
    optim_2 = Adam(lr=0.0001)
    # Re-compile the model, this time leaving the last 2 layers unfrozen for Fine-Tuning
    vgg_model_ft = create_model(input_shape, n_classes, optim_2, fine_tune=2)

    plot_loss_2 = PlotLossesCallback()
    # ModelCheckpoint callback - save best weights
    tl_checkpoint_1 = ModelCheckpoint(filepath='tl_model_v1.weights.{}.best.hdf5'.format(crossval_i),
                                      save_best_only=True,
                                      verbose=1)

    early_stop = EarlyStopping(monitor='loss',
                               patience=5,
                               restore_best_weights=True,
                               mode='min')

    # Retrain model with fine-tuning
    vgg_ft_history = vgg_model_ft.fit(traingen,
                                      batch_size=BATCH_SIZE,
                                      epochs=n_epochs,
                                      validation_data=validgen,
                                      steps_per_epoch=n_steps,
                                      validation_steps=n_val_steps,
                                      callbacks=[tl_checkpoint_1, early_stop, plot_loss_2],
                                      # callbacks=[tl_checkpoint_1, plot_loss_2],
                                      verbose=1)

    # Generate predictions
    vgg_model_ft.load_weights('tl_model_v1.weights.{}.best.hdf5'.format(crossval_i)) # initialize the best trained weights
    true_classes = testgen.classes
    class_indices = traingen.class_indices
    class_indices = dict((v,k) for k,v in class_indices.items())

    vgg_preds_ft = vgg_model_ft.predict(testgen)
    vgg_pred_classes_ft = np.argmax(vgg_preds_ft, axis=1)

    print('**1:', vgg_pred_classes_ft, type(vgg_pred_classes_ft))
    print('**2:', true_classes, type(true_classes))
    vgg_acc_ft = accuracy_score(true_classes, vgg_pred_classes_ft)
    print("VGG16 Model Accuracy with Fine-Tuning: {:.2f}%".format(vgg_acc_ft * 100))
    print("precision", precision_score(true_classes, vgg_pred_classes_ft, average='weighted')*100,
          "recall", recall_score(true_classes, vgg_pred_classes_ft, average='weighted')*100,
          "f1_score", f1_score(true_classes, vgg_pred_classes_ft, average='weighted')*100)


    class_names = testgen.class_indices.keys()



    fig = plt.figure()
    ax = fig.add_subplot(111)
    plot_heatmap(true_classes, vgg_pred_classes_ft, class_names, ax, title="Transfer Learning (VGG16) with Fine-Tuning")

    # fig.suptitle("Confusion Matrix Model Comparison", fontsize=24)
    # fig.tight_layout()
    # fig.subplots_adjust(top=1.25)
    # plt.show()
    plt.savefig('confusion_matrix_{}.png'.format(crossval_i))