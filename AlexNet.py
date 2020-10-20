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

class AlexNet(Model):
    def __init__(self):
        super(AlexNet,self).__init__()
        self.c1=Conv2D(filters=96,kernel_size=(3,3),padding='valid')
        self.b1=BatchNormalization()
        self.a1=Activation('relu')
        self.p1=MaxPool2D(pool_size=(3,3),strides=2)

        self.c2=Conv2D(filters=256,kernel_size=(3,3),padding='valid')
        self.b2=BatchNormalization()
        self.a2=Activation('relu')
        self.p2=MaxPool2D(pool_size=(3,3),strides=2)

        self.c3=Conv2D(filters=384,kernel_size=(3,3),padding='same')
        self.a3=Activation('relu')

        self.c4=Conv2D(filters=384,kernel_size=(3,3),padding='same')
        self.a4=Activation('relu')

        self.c5=Conv2D(filters=256,kernel_size=(3,3),padding='valid')
        self.a5=Activation('relu')
        self.p5=MaxPool2D(pool_size=(3,3),strides=2)

        self.flatten=Flatten()
        self.f1=Dense(2048,activation='relu')
        self.d1=Dropout(0.5)
        self.f2=Dense(2048,activation='relu')
        self.d2=Dropout(0.5)
        self.f3=Dense(10,activation='softmax')

    def call(self, inputs, training=None, mask=None):
        x=self.c1(inputs)
        x=self.b1(x)
        x=self.a1(x)
        x=self.p1(x)

        x=self.c2(x)
        x=self.b2(x)
        x=self.a2(x)
        x=self.p2(x)

        x=self.c3(x)
        x=self.a3(x)

        x=self.c4(x)
        x=self.a4(x)

        x=self.c5(x)
        x=self.a5(x)
        x=self.p5(x)

        x=self.flatten(x)
        x=self.f1(x)
        x=self.d1(x)
        x=self.f2(x)
        x=self.d2(x)
        y=self.f3(x)
        return y

model=AlexNet()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

checkpoint_path='./CNN/AlexNet_ckpt/ckpt'

if os.path.exists(checkpoint_path+'.index'):
    print("-"*12+" loading model saved"+"-"*12)
    model.load_weights(checkpoint_path)
checkpoint=tf.keras.callbacks.ModelCheckpoint(checkpoint_path,save_best_only=True,save_weights_only=True)
callback_list=[checkpoint]

history=model.fit(x_train,y_train,batch_size=32,epochs=5,validation_data=(x_test,y_test),
                  validation_freq=1,callbacks=callback_list)

model.summary()

weight_path='./CNN/AlexNet_weight.txt'
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