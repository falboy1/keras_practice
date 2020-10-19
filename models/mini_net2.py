# -*- coding: utf-8 -*-
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Flatten, Dense, Dropout


# ネットワークを構築し定義したモデルを取得
def get_model(img_width, img_height, num_classes):
    input_tensor = Input(shape=(img_width, img_height, 3))
    x = Conv2D(112, kernel_size=(3, 3), activation='relu')(input_tensor)
    x = Conv2D(56, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Conv2D(28, kernel_size=(3, 3), activation='relu')(x)
    x = Conv2D(14, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Dropout(0.25)(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.25)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    # モデルの定義
    model = Model(inputs=input_tensor, outputs=predictions)
    return model