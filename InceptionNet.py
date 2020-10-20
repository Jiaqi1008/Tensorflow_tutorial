import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense,GlobalAveragePooling2D,Conv2D,BatchNormalization,Activation,MaxPool2D,Dropout
from tensorflow.keras import datasets
import numpy as np
from matplotlib import pyplot as plt
np.set_printoptions(threshold=np.inf)


(x_train,y_train),(x_test,y_test)=datasets.cifar10.load_data()
x_train,x_test=x_train/255,x_test/255

class ConBNRelu(Model): #the smallest part of the block
    def __init__(self,ch,kernelsize=3,strides=1,padding='same'):
        super(ConBNRelu,self).__init__()
        self.model=tf.keras.models.Sequential([
            Conv2D(ch,kernelsize,strides=strides,padding=padding),
            BatchNormalization(),
            Activation('relu')
        ])

    def call(self, inputs, training=None, mask=None):
        y=self.model(inputs)
        return y

class InceptionBlk(Model):  #use ConBNRelu to build four-way Inception block
    def __init__(self,ch,strides=1):
        super(InceptionBlk,self).__init__()
        self.ch=ch
        self.strides=strides
        self.c1=ConBNRelu(ch,kernelsize=1,strides=strides)
        self.c2_1=ConBNRelu(ch,kernelsize=1,strides=strides)
        self.c2_2=ConBNRelu(ch,kernelsize=3,strides=1)
        self.c3_1=ConBNRelu(ch,kernelsize=1,strides=strides)
        self.c3_2=ConBNRelu(ch,kernelsize=5,strides=1)
        self.p4_1=MaxPool2D(pool_size=3,strides=1,padding='same')
        self.c4_2=ConBNRelu(ch,kernelsize=1,strides=strides)

    def call(self, inputs, training=None, mask=None):
        x1=self.c1(inputs)
        x2_1=self.c2_1(inputs)
        x2_2 = self.c2_2(x2_1)
        x3_1=self.c3_1(inputs)
        x3_2 = self.c3_2(x3_1)
        x4_1=self.p4_1(inputs)
        x4_2 = self.c4_2(x4_1)
        y=tf.concat([x1,x2_2,x3_2,x4_2],axis=3)
        return y

class InceptionNet(Model):  #use the block to build the model
    def __init__(self,num_blocks,num_classes,init_ch=16,**kwargs):
        super(InceptionNet,self).__init__(**kwargs)
        self.init_ch=init_ch
        self.out_channels=init_ch
        self.c1=ConBNRelu(init_ch)
        self.blocks=tf.keras.Sequential()
        for block_id in range(num_blocks):
            for layer_id in range(2):
                if layer_id==0:
                    block=InceptionBlk(self.out_channels,strides=2)
                else:
                    block=InceptionBlk(self.out_channels,strides=1)
                self.blocks.add(block)
            self.out_channels*=2    #the first layer's strides is 2, to make the information carried unchange, double the output channels
        self.p1=GlobalAveragePooling2D()
        self.f1=Dense(num_classes,activation='softmax')

    def call(self, inputs, training=None, mask=None):
        x=self.c1(inputs)
        x=self.blocks(x)
        x=self.p1(x)
        y=self.f1(x)
        return y

model=InceptionNet(num_blocks=2,num_classes=10)

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

checkpoint_path='./CNN/InceptionNet_ckpt/ckpt'

if os.path.exists(checkpoint_path+'.index'):
    print("-"*12+" loading model saved"+"-"*12)
    model.load_weights(checkpoint_path)
checkpoint=tf.keras.callbacks.ModelCheckpoint(checkpoint_path,save_best_only=True,save_weights_only=True)
callback_list=[checkpoint]

history=model.fit(x_train,y_train,batch_size=1024,epochs=5,validation_data=(x_test,y_test),
                  validation_freq=1,callbacks=callback_list)

model.summary()

weight_path='./CNN/InceptionNet_weight.txt'
with open(weight_path,'w') as file:
    for v in model.trainable_variables:
        file.write(str(v.name)+'\n')
        file.write(str(v.shape)+'\n')
        file.write(str(v.numpy())+'\n')

acc=history.history['sparse_categorical_accuracy']
val_acc=history.history['val_sparse_categorical_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']

plt.subplot(121)
plt.plot(acc,label='Accuracy')
plt.plot(val_acc,label='Validation Accuracy')
plt.title("Train and Validation Accuracy")
plt.legend()

plt.subplot(122)
plt.plot(loss,label='Loss')
plt.plot(val_loss,label='Validation Loss')
plt.title("Train and Validation Loss")
plt.legend()
plt.show()