# -*- coding: utf-8 -*-
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense


def get_model(img_width, img_height, num_classes, transfer=True):
    # ネットワーク作成
    input_tensor = Input(shape=(img_width, img_height, 3))
    base_model = ResNet50(include_top=False, weights='imagenet',input_tensor=input_tensor)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    # モデルの定義
    model = Model(inputs=base_model.input, outputs=predictions)

    if transfer is True:
        # ベースモデルの重みを固定化
        for layer in base_model.layers:
            layer.trainable = False
    else:
        pass

    return model