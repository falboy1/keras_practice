# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import  confusion_matrix
import numpy as np


# historyオブジェクトから学習曲線を画像保存する関数
def plot_history(filename, histories ,key='loss'):
    plt.figure(figsize=(16,10))
    for name, history in histories:
        val = plt.plot(history.epoch, history.history['val_'+key],
                        '--', label=name.title()+' Val')
        plt.plot(history.epoch, history.history[key], color=val[0].get_color(),
                    label=name.title()+' Train')

    plt.xlabel('Epochs', fontsize='large')
    plt.ylabel(key.replace('_',' ').title(), fontsize='large')
    plt.legend()

    plt.xlim([0,max(history.epoch)])
    plt.savefig(filename)
    plt.show()


# 混合行列を画像保存する関数
def plot_confusion_matrix(cm, classes, filename, normalize=False, title='confusion_matrix', cmap=plt.cm.Greens):
    plt.figure(figsize=(14,10))
    plt.imshow(cm, cmap=cmap)
    plt.title(title, fontsize=25)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=15)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=15)
    plt.yticks(tick_marks, classes, fontsize=15)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() /2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                horizontalalignment="center", fontsize=15,
                color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.xlabel("ground_truth", fontsize=25)
    plt.ylabel("predict", fontsize=25)
    plt.savefig(filename)
    
