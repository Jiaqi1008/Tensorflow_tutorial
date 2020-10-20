import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense,GlobalAveragePooling2D,Conv2D,BatchNormalization,Activation
from tensorflow.keras import datasets
import numpy as np
from matplotlib import pyplot as plt
np.set_printoptions(threshold=np.inf)


(x_train,y_train),(x_test,y_test)=datasets.cifar10.load_data()
x_train,x_test=x_train/255,x_test/255

class ResBlk(Model):  #build ResNet block,adjust dimension by residual
    def __init__(self,filters,strides=1,residual_path=False):
        super(ResBlk,self).__init__()
        self.filters=filters
        self.strides=strides
        self.residual=residual_path

        self.c1=Conv2D(filters,kernel_size=(3,3),strides=strides,padding='same',use_bias=False)
        self.b1=BatchNormalization()
        self.a1=Activation('relu')
        self.c2=Conv2D(filters,kernel_size=(3,3),strides=1,padding='same',use_bias=False)
        self.b2=BatchNormalization()
        if residual_path:
            self.down_c1=Conv2D(filters,(1,1),strides=strides,padding='same',use_bias=False)
            self.down_b1=BatchNormalization()
        self.a2=Activation('relu')

    def call(self, inputs, training=None, mask=None):
        residual=inputs
        x=self.c1(inputs)
        x=self.b1(x)
        x=self.a1(x)
        x=self.c2(x)
        x=self.b2(x)
        if self.residual:
            residual=self.down_c1(inputs)
            residual=self.down_b1(residual)
        y=self.a2(x+residual)
        return y

class ResNet(Model):  #use the block to build the model
    def __init__(self,block_list,initial_filter=64):
        super(ResNet,self).__init__()
        self.num_block=len(block_list)
        self.block_list=block_list
        self.out_filter=initial_filter
        self.c1=Conv2D(self.out_filter,kernel_size=(3,3),strides=1,padding='same',use_bias=False)
        self.b1=BatchNormalization()
        self.a1=Activation('relu')
        self.blocks=tf.keras.Sequential()
        for block_id in range(len(block_list)):
            for layer_id in range(block_list[block_id]):
                if block_id!=0 and layer_id==0:
                    block=ResBlk(self.out_filter,strides=2,residual_path=True)
                else:
                    block=ResBlk(self.out_filter,residual_path=False)
                self.blocks.add(block)
            self.out_filter*=2    #the first layer's strides is 2, to make the information carried unchange, double the output channels
        self.p1=GlobalAveragePooling2D()
        self.f1=Dense(10)

    def call(self, inputs, training=None, mask=None):
        x=self.c1(inputs)
        x=self.b1(x)
        x=self.a1(x)
        x=self.blocks(x)
        x=self.p1(x)
        y=self.f1(x)
        return y

model=ResNet([2,2,2,2])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

checkpoint_path='./CNN/ResNet_ckpt/ckpt'

if os.path.exists(checkpoint_path+'.index'):
    print("-"*12+" loading model saved"+"-"*12)
    model.load_weights(checkpoint_path)
checkpoint=tf.keras.callbacks.ModelCheckpoint(checkpoint_path,save_best_only=True,save_weights_only=True)
callback_list=[checkpoint]

history=model.fit(x_train,y_train,batch_size=128,epochs=5,validation_data=(x_test,y_test),
                  validation_freq=1,callbacks=callback_list)

model.summary()

weight_path='./CNN/ResNet_weight.txt'
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