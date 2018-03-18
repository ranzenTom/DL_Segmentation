import keras.backend as K
import tensorflow as tf

"""Metrics defined in paper https://arxiv.org/pdf/1411.4038.pdf"""

def mean_IU(y_true, y_pred):
    result = 0
    nc = tf.cast(tf.shape(y_true)[-1], tf.float32)
    for i in range(2):
        nii = tf.reduce_sum(tf.round(tf.multiply(y_true[:,:,:,i],y_pred[:,:,:,i]))) #nii = number of pixels of classe i predicted to belong to class i
        ti = tf.reduce_sum(y_true[:,:,:,i]) #number of pixels of class i
        loc_sum = 0
        for j in range(2):
            nji = tf.reduce_sum(tf.round(tf.multiply(y_true[:,:,:,j],y_pred[:,:,:,i]))) #number of pixels of classe j predicted to belong to class i
            loc_sum += nji
        result += nii/(ti - nii + loc_sum)
    return (1/nc)*result


def pixel_accuracy(y_true, y_pred):
    sum_nii = tf.reduce_sum(tf.round(tf.multiply(y_true[:, :, :, :], y_pred[:, :, :,:])))  # nii = number of pixels of classe i predicted to belong to class i
    sum_ti = tf.reduce_sum(y_true[:, :, :, :])  # ti = number of pixels of class i
    return sum_nii/sum_ti


def mean_accuracy(y_true, y_pred):
    result = 0
    nc = tf.cast(tf.shape(y_true)[-1], tf.float32)  # number of classes
    for i in range(2):
        nii = tf.reduce_sum(tf.round(tf.multiply(y_true[:, :, :, i], y_pred[:, :, :,i])))  # number of pixels of classe i predicted to belong to class i
        ti = tf.reduce_sum(y_true[:, :, :, i])  # number of pixels of class i
        if ti != 0: #sometimes one class is not represented in the picture
            result += (nii/ti)
        else:
            nc-=1
    return (1/nc)*result

def mean_accuracy_new(y_true, y_pred):
    nii = tf.reduce_sum(tf.round(tf.multiply(y_true, y_pred)), axis=[0,1,2])
    ti = tf.reduce_sum(y_true, axis=[0,1,2])
    return tf.reduce_mean(nii/ti)

def mean_IU_new(y_true, y_pred):
    nii = tf.reduce_sum(tf.round(tf.multiply(y_true, y_pred)), axis=[0,1,2])
    ti = tf.reduce_sum(y_true, axis=[0,1,2])
