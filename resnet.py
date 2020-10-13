# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import pprint

# tensorflow関連
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint


# 学習曲線を画像保存する関数
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



def main():
    # 出力クラス
    classes = ['unripe', 'ripe', 'overripe'] #分類するクラス
    nb_classes = len(classes)

    # データセットディレクトリ
    train_data_dir = './dataset/train'
    validation_data_dir = './dataset/val'

    # データセット画像数
    nb_train_samples = 150
    nb_validation_samples = 20

    # 入力画像サイズ
    img_width, img_height = 224, 224

    # データ拡張パラメータ (訓練画像)
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

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        color_mode='rgb',
        classes=classes,
        class_mode='categorical',
        batch_size=32
    )

    # データ拡張パラメータ (検証用)
    validation_datagen = ImageDataGenerator(
        rescale=1.0 / 255
    )

    validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        color_mode='rgb',
        classes=classes,
        class_mode='categorical',
        batch_size=32
    )

    # ネットワーク作成
    input_tensor = Input(shape=(img_width, img_height, 3))
    base_model = ResNet50(include_top=False, weights='imagenet',input_tensor=input_tensor)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    predictions = Dense(3, activation='softmax')(x)

    # モデルの定義
    model = Model(inputs=base_model.input, outputs=predictions)

    # ベースモデルの重みを固定化
    for layer in base_model.layers:
        layer.trainable = False

    # 学習方法の設定
    model.compile(
        loss='categorical_crossentropy',
        optimizer=SGD(lr=0.0001, momentum=0.9),
        metrics=['accuracy']
    )

    # チェックポイント用のコールバック
    checkpoint = ModelCheckpoint(
                        filepath="./weights/model-{epoch:02d}-{val_loss:.2f}.h5",
                        monitor='val_loss',
                        save_best_only=True,
                        period=10
    )

    # 訓練の実行
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples,
        epochs=100,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples,
        callbacks=[checkpoint]
    )

    # 学習曲線を保存
    plot_history([('resnet-50', history)], key='loss', 'figure_loss.png')
    plot_history([('resnet-50', history)], key='accuracy', 'figure_accuracy.png')


if __name__ == '__main__':
    main()