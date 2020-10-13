# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import pprint

def plot_history(histories, key='loss', filename):
    plt.figure(figsize=(16,10))
    for name, history in histories:
        print(history.history)
        val = plt.plot(history.epoch, history.history['val_'+key],
                        '--', label=name.title()+' Val')
        plt.plot(history.epoch, history.history[key], color=val[0].get_color(),
                    label=name.title()+' Train')

    plt.xlabel('Epochs')
    plt.ylabel(key.replace('_',' ').title())
    plt.legend()

    plt.xlim([0,max(history.epoch)])
    plt.savefig(filename)


classes = ['unripe', 'ripe', 'overripe'] #分類するクラス
nb_classes = len(classes)

train_data_dir = './dataset/train'
validation_data_dir = './dataset/val'

nb_train_samples = 150
nb_validation_samples = 20

img_width, img_height = 224, 224


from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(
    rescale=1.0 / 255
)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    color_mode='rgb',
    classes=classes,
    class_mode='categorical',
    batch_size=32
)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    color_mode='rgb',
    classes=classes,
    class_mode='categorical',
    batch_size=32
)

from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense

input_tensor = Input(shape=(img_width, img_height, 3))
base_model = ResNet50(include_top=False, weights='imagenet',input_tensor=input_tensor)


x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(256, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(3, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False


for layer in model.layers[:100]:
    layer.trainable = False

from tensorflow.keras.optimizers import SGD
model.compile(
    loss='categorical_crossentropy',
    optimizer=SGD(lr=0.0001, momentum=0.9),
    metrics=['accuracy']
)


from tensorflow.keras.callbacks import ModelCheckpoint
# チェックポイント
checkpoint = ModelCheckpoint(
                    filepath="./weights/model-{epoch:02d}-{val_loss:.2f}.h5",
                    monitor='val_loss',
                    save_best_only=True,
                    period=10
)

'''
from plot_history import PlotHistory
import matplotlib.pyplot as plt
dir_name = './dst'
title = f'{model_name}_{optimizer_name}'
ph = PlotHistory(
    save_interval=1, dir_name=dir_name, csv_output=True, title=title
)
'''

history = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples,
    epochs=100,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples,
    callbacks=[checkpoint]
)

plot_history([('resnet-50', history)], key='loss', 'figure_loss.png')
plot_history([('resnet-50', history)], key='accuracy', 'figure_accuracy.png')