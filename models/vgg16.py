# -*- coding: utf-8 -*-
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense
from tensorflow.keras.layers import BatchNormalization, Flatten


def get_model(img_width, img_height, num_classes, transfer=False, weights='imagenet', fine_tuning=None):
    # ネットワーク作成
    input_tensor = Input(shape=(img_width, img_height, 3))
    base_model = VGG16(include_top=False, weights=weights ,input_tensor=input_tensor)
    x = base_model.output
    x = Flatten()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    # モデルの定義
    model = Model(inputs=base_model.input, outputs=predictions)

    # 転移学習の処理
    if transfer is True:
        # ベースモデルの重みを凍結
        for layer in base_model.layers:
            layer.trainable = False
    else:
        pass

    # Fin-tuning用の処理 (intのみ受付)
    if isinstance(fine_tuning, int):
        for i, layer in enumerate(base_model.layers):
            if i < fine_tuning:
                layer.trainable = False
    else:
        pass



    return model