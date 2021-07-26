# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 18:33:56 2021

@author: zjl-seu
"""
import tensorflow as tf
from tensorflow.keras import Model 
from tensorflow.keras.layers import Conv2D, Dropout, Dense, MaxPool2D, Flatten, BatchNormalization, Activation 

class model_vgg16(Model):
    def __init__(self):
        super(model_vgg16, self).__init__()
        self.c1_1 = Conv2D(filters=64, kernel_size=(3,3), padding="same", name="c1_1")
        self.b1_1 = BatchNormalization()
        self.a1_1 = Activation("relu")
        self.c1_2 = Conv2D(filters=64, kernel_size=(3,3), padding="same", name="c1_2")
        self.b1_2 = BatchNormalization()
        self.a1_2 = Activation("relu")
        self.p1 = MaxPool2D(pool_size=(2,2), strides=2, padding="same")
        self.d1 = Dropout(0.2)
        
        self.c2_1 = Conv2D(filters=128, kernel_size=(3,3), padding="same", name="c2_1")
        self.b2_1 = BatchNormalization()
        self.a2_1 = Activation("relu")
        self.c2_2 = Conv2D(filters=128, kernel_size=(3,3), padding="same", name="c2_2")
        self.b2_2 = BatchNormalization()
        self.a2_2 = Activation("relu")
        self.p2 = MaxPool2D(pool_size=(2,2), strides=2, padding="same")
        self.d2 = Dropout(0.2)
        
        self.c3_1 = Conv2D(filters=256, kernel_size=(3,3), padding="same", name="c3_1")
        self.b3_1 = BatchNormalization()
        self.a3_1 = Activation("relu")
        self.c3_2 = Conv2D(filters=256, kernel_size=(3,3), padding="same", name="c3_2")
        self.b3_2 = BatchNormalization()
        self.a3_2 = Activation("relu")
        self.c3_3 = Conv2D(filters=256, kernel_size=(3,3), padding="same", name="c3_3")
        self.b3_3 = BatchNormalization()
        self.a3_3 = Activation("relu")
        self.p3 = MaxPool2D(pool_size=(2,2), strides=2, padding="same")
        self.d3 = Dropout(0.2)
        
        self.c4_1 = Conv2D(filters=512, kernel_size=(3,3), padding="same", name="c4_1")
        self.b4_1 = BatchNormalization()
        self.a4_1 = Activation("relu")
        self.c4_2 = Conv2D(filters=512, kernel_size=(3,3), padding="same", name="c4_2")
        self.b4_2 = BatchNormalization()
        self.a4_2 = Activation("relu")
        self.c4_3 = Conv2D(filters=512, kernel_size=(3,3), padding="same", name="c4_3")
        self.b4_3 = BatchNormalization()
        self.a4_3 = Activation("relu")
        self.p4 = MaxPool2D(pool_size=(2,2), strides=2, padding="same")
        self.d4 = Dropout(0.2)
        
        self.c5_1 = Conv2D(filters=512, kernel_size=(3,3), padding="same", name="c5_1")
        self.b5_1 = BatchNormalization()
        self.a5_1 = Activation("relu")
        self.c5_2 = Conv2D(filters=512, kernel_size=(3,3), padding="same", name="c5_2")
        self.b5_2 = BatchNormalization()
        self.a5_2 = Activation("relu")
        self.c5_3 = Conv2D(filters=512, kernel_size=(3,3), padding="same", name="c5_3")
        self.b5_3 = BatchNormalization()
        self.a5_3 = Activation("relu")
        self.p5 = MaxPool2D(pool_size=(2,2), strides=2, padding="same")
        self.d5 = Dropout(0.2)
          
        self.flatten = Flatten()
        self.f6 = Dense(4096, activation="relu", name="f6")
        self.d6 = Dropout(0.2)
        self.f7 = Dense(512, activation="relu", name="f7")
        self.d7 = Dropout(0.2)
        self.f8 = Dense(1, activation="sigmoid", name="f8")
    def call(self, x):
        x = self.c1_1(x)
        x = self.b1_1(x)
        x = self.a1_1(x)
        x = self.c1_2(x)
        x = self.b1_2(x)
        x = self.a1_2(x)
        x = self.p1(x)
        x = self.d1(x)
        
        x = self.c2_1(x)
        x = self.b2_1(x)
        x = self.a2_1(x)
        x = self.c2_2(x)
        x = self.b2_2(x)
        x = self.a2_2(x)
        x = self.p2(x)
        x = self.d2(x)
        
        x = self.c3_1(x)
        x = self.b3_1(x)
        x = self.a3_1(x)
        x = self.c3_2(x)
        x = self.b3_2(x)
        x = self.a3_2(x)
        x = self.c3_3(x)
        x = self.b3_3(x)
        x = self.a3_3(x)
        x = self.p3(x)
        x = self.d3(x)
        
        x = self.c4_1(x)
        x = self.b4_1(x)
        x = self.a4_1(x)
        x = self.c4_2(x)
        x = self.b4_2(x)
        x = self.a4_2(x)
        x = self.c4_3(x)
        x = self.b4_3(x)
        x = self.a4_3(x)
        x = self.p4(x)
        x = self.d4(x)
        
        x = self.c5_1(x)
        x = self.b5_1(x)
        x = self.a5_1(x)
        x = self.c5_2(x)
        x = self.b5_2(x)
        x = self.a5_2(x)
        x = self.c5_3(x)
        x = self.b5_3(x)
        x = self.a5_3(x)
        x = self.p5(x)
        x = self.d5(x)
          
        x = self.flatten(x)
        x = self.f6(x)
        x = self.d6(x)
        x = self.f7(x)
        x = self.d7(x)
        y = self.f8(x)
        return y
    
class model_rpn(Model):
    def __init__(self):
        super(model_rpn, self).__init__()
        self.c1_1 = Conv2D(filters=64, kernel_size=(3,3), padding="same", name="c1_1")
        self.b1_1 = BatchNormalization()
        self.a1_1 = Activation("relu")
        self.c1_2 = Conv2D(filters=64, kernel_size=(3,3), padding="same", name="c1_2")
        self.b1_2 = BatchNormalization()
        self.a1_2 = Activation("relu")
        self.p1 = MaxPool2D(pool_size=(2,2), strides=2, padding="same")
        self.d1 = Dropout(0.2)
        
        self.c2_1 = Conv2D(filters=128, kernel_size=(3,3), padding="same", name="c2_1")
        self.b2_1 = BatchNormalization()
        self.a2_1 = Activation("relu")
        self.c2_2 = Conv2D(filters=128, kernel_size=(3,3), padding="same", name="c2_2")
        self.b2_2 = BatchNormalization()
        self.a2_2 = Activation("relu")
        self.p2 = MaxPool2D(pool_size=(2,2), strides=2, padding="same")
        self.d2 = Dropout(0.2)
        
        self.c3_1 = Conv2D(filters=256, kernel_size=(3,3), padding="same", name="c3_1")
        self.b3_1 = BatchNormalization()
        self.a3_1 = Activation("relu")
        self.c3_2 = Conv2D(filters=256, kernel_size=(3,3), padding="same", name="c3_2")
        self.b3_2 = BatchNormalization()
        self.a3_2 = Activation("relu")
        self.c3_3 = Conv2D(filters=256, kernel_size=(3,3), padding="same", name="c3_3")
        self.b3_3 = BatchNormalization()
        self.a3_3 = Activation("relu")
        self.p3_1 = MaxPool2D(pool_size=(2,2), strides=2, padding="same")
        self.p3_2 = MaxPool2D(pool_size=(2,2), strides=2, padding="same")
        self.d3 = Dropout(0.2)
        
        self.c4_1 = Conv2D(filters=512, kernel_size=(3,3), padding="same", name="c4_1")
        self.b4_1 = BatchNormalization()
        self.a4_1 = Activation("relu")
        self.c4_2 = Conv2D(filters=512, kernel_size=(3,3), padding="same", name="c4_2")
        self.b4_2 = BatchNormalization()
        self.a4_2 = Activation("relu")
        self.c4_3 = Conv2D(filters=512, kernel_size=(3,3), padding="same", name="c4_3")
        self.b4_3 = BatchNormalization()
        self.a4_3 = Activation("relu")
        self.p4 = MaxPool2D(pool_size=(2,2), strides=2, padding="same")
        self.d4 = Dropout(0.2)
        
        self.c5_1 = Conv2D(filters=512, kernel_size=(3,3), padding="same", name="c5_1")
        self.b5_1 = BatchNormalization()
        self.a5_1 = Activation("relu")
        self.c5_2 = Conv2D(filters=512, kernel_size=(3,3), padding="same", name="c5_2")
        self.b5_2 = BatchNormalization()
        self.a5_2 = Activation("relu")
        self.c5_3 = Conv2D(filters=512, kernel_size=(3,3), padding="same", name="c5_3")
        self.b5_3 = BatchNormalization()
        self.a5_3 = Activation("relu")
        
        self.rpn_c1 = Conv2D(filters=256, kernel_size=(5,2), activation='relu', padding="same", use_bias=False, name="rpn_c1")
        self.rpn_c2 = Conv2D(filters=512, kernel_size=(5,2), activation='relu', padding="same", use_bias=False, name="rpn_c2")
        self.rpn_c3 = Conv2D(filters=512, kernel_size=(5,2), activation='relu', padding="same", use_bias=False, name="rpn_c3")
        
        self.bboxes_c1 = Conv2D(filters=36, kernel_size=[1,1], padding='same', use_bias=False, name="boxes_c1")
        self.scores_c1 = Conv2D(filters=18, kernel_size=[1,1], padding='same', use_bias=False, name="scores_c1")
        
    def call(self, x):
        x = self.c1_1(x)
        x = self.b1_1(x)
        x = self.a1_1(x)
        x = self.c1_2(x)
        x = self.b1_2(x)
        x = self.a1_2(x)
        x = self.p1(x)
        x = self.d1(x)
        
        x = self.c2_1(x)
        x = self.b2_1(x)
        x = self.a2_1(x)
        x = self.c2_2(x)
        x = self.b2_2(x)
        x = self.a2_2(x)
        x = self.p2(x)
        x = self.d2(x)
        
        x = self.c3_1(x)
        x = self.b3_1(x)
        x = self.a3_1(x)
        x = self.c3_2(x)
        x = self.b3_2(x)
        x = self.a3_2(x)
        x = self.c3_3(x)
        x = self.b3_3(x)
        x = self.a3_3(x)
        x = self.p3_1(x)
        x1 = self.rpn_c1(self.p3_2(x))
        x = self.d3(x)
         
        x = self.c4_1(x)
        x = self.b4_1(x)
        x = self.a4_1(x)
        x = self.c4_2(x)
        x = self.b4_2(x)
        x = self.a4_2(x)
        x = self.c4_3(x)
        x = self.b4_3(x)
        x = self.a4_3(x)
        x = self.p4(x)
        x2= self.rpn_c2(x)
        x = self.d4(x)
        
        x = self.c5_1(x)
        x = self.b5_1(x)
        x = self.a5_1(x)
        x = self.c5_2(x)
        x = self.b5_2(x)
        x = self.a5_2(x)
        x = self.c5_3(x)
        x = self.b5_3(x)
        x = self.a5_3(x)
        x3 = self.rpn_c3(x)
        
        x = tf.concat([x1, x2, x3], axis=-1)
        conv_cls_scores = self.scores_c1(x) 
        conv_cls_bboxes = self.bboxes_c1(x) 
        cls_scores = tf.reshape(conv_cls_scores, [1, conv_cls_scores.shape[1], conv_cls_scores.shape[2], 9, 2])
        cls_bboxes = tf.reshape(conv_cls_bboxes, [1, conv_cls_bboxes.shape[1], conv_cls_bboxes.shape[2], 9, 4])
        return cls_scores, cls_bboxes
    
class model_TYPE(Model):
    def __init__(self):
        super(model_TYPE, self).__init__()
        self.c1 = Conv2D(filters=16, kernel_size=(3,3), padding='same', name="c1")
        self.b1 = BatchNormalization()
        self.a1 = Activation("relu")
        self.p1 = MaxPool2D(pool_size=(2,2), strides=2, padding="same")
        self.d1 = Dropout(0.2)
        self.c2 = Conv2D(filters=16, kernel_size=(3,3), padding='same', name="c2")
        self.b2 = BatchNormalization()
        self.a2 = Activation("relu")
        self.p2 = MaxPool2D(pool_size=(2,2), strides=2, padding="same")
        self.d2 = Dropout(0.2)
        self.flatten = Flatten()
        self.f3 = Dense(64, activation="relu",  name="f1")
        self.d3 = Dropout(0.2)
        self.f4 = Dense(1, activation="sigmoid", name="f2")    
    def call(self, x):
        x = self.c1(x)
        x = self.b1(x)
        x = self.a1(x)
        x = self.p1(x)
        x = self.d1(x)
        x = self.c2(x)
        x = self.b2(x)
        x = self.a2(x)
        x = self.p2(x)
        x = self.d2(x)
        x = self.flatten(x)
        x = self.f3(x)
        x = self.d3(x)
        y = self.f4(x)
        return y
    
class model_LPR(tf.keras.Model):
    def __init__(self):
        super(model_LPR, self).__init__()
        self.c1 = Conv2D(32, 3, padding='same')
        self.b1 = BatchNormalization()
        self.a1 = Activation('relu')
        self.p1 = MaxPool2D(2)
        self.c2 = Conv2D(64, 3, padding='same')
        self.b2 = BatchNormalization()
        self.a2 = Activation('relu')
        self.p2 = MaxPool2D(2)
        self.c3 = Conv2D(128, 3,padding='same')
        self.b3 = BatchNormalization()
        self.a3 = Activation('relu')
        self.p3 = MaxPool2D(2)
        self.c4 = Conv2D(256, 5)
        self.b4 = BatchNormalization()
        self.a4 = Activation('relu')
        self.c5 = Conv2D(1024, 1)
        self.b5 = BatchNormalization()
        self.a5 = Activation('relu')
        self.c6 = Conv2D(84, 1)
        self.a6 = Activation('softmax')
    def call(self, x):
        x = self.c1(x)
        x = self.b1(x)
        x = self.a1(x)
        x = self.p1(x)
        x = self.c2(x)
        x = self.b2(x)
        x = self.a2(x)
        x = self.p2(x)
        x = self.c3(x)
        x = self.b3(x)
        x = self.a3(x)
        x = self.p3(x)
        x = self.c4(x)
        x = self.b4(x)
        x = self.a4(x)
        x = self.c5(x)
        x = self.b5(x)
        x = self.a5(x)
        x = self.c6(x)
        y = self.a6(x)
        return y