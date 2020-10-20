import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense,Flatten,Conv2D,BatchNormalization,Activation,MaxPool2D,Dropout
from tensorflow.keras import datasets
import numpy as np
from matplotlib import pyplot as plt
np.set_printoptions(threshold=np.inf)


(x_train,y_train),(x_test,y_test)=datasets.cifar10.load_data()
x_train,x_test=x_train/255,x_test/255

class CifarModel(Model):
    def __init__(self):
        super(CifarModel,self).__init__()
        self.c=Conv2D(filters=6,kernel_size=(5,5),padding='same')
        self.b=BatchNormalization()
        self.a=Activation('relu')
        self.p=MaxPool2D(pool_size=(2,2),strides=2,padding='same')
        self.d1=Dropout(0.2)

        self.f=Flatten()
        self.D1=Dense(128,activation='relu',)
        self.d2=Dropout(0.2)
        self.D2=Dense(10,activation='softmax')

    def call(self, inputs, training=None, mask=None):
        x=self.c(inputs)
        x=self.b(x)
        x=self.a(x)
        x=self.p(x)
        x=self.d1(x)

        x=self.f(x)
        x=self.D1(x)
        x=self.d2(x)
        y=self.D2(x)
        return y

model=CifarModel()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

checkpoint_path='./CNN/cifar_ckpt/ckpt'

if os.path.exists(checkpoint_path+'.index'):
    print("-"*12+" loading model saved"+"-"*12)
    model.load_weights(checkpoint_path)
checkpoint=tf.keras.callbacks.ModelCheckpoint(checkpoint_path,save_best_only=True,save_weights_only=True)
callback_list=[checkpoint]

history=model.fit(x_train,y_train,batch_size=32,epochs=5,validation_data=(x_test,y_test),
                  validation_freq=1,callbacks=callback_list)

model.summary()

weight_path='./CNN/cifar_weight.txt'
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