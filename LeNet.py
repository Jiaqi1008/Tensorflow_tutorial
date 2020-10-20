import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense,Flatten,Conv2D,Activation,MaxPool2D
from tensorflow.keras import datasets
import numpy as np
from matplotlib import pyplot as plt
np.set_printoptions(threshold=np.inf)


(x_train,y_train),(x_test,y_test)=datasets.cifar10.load_data()
x_train,x_test=x_train/255,x_test/255

class LeNet(Model):
    def __init__(self):
        super(LeNet,self).__init__()
        self.c1=Conv2D(filters=6,kernel_size=(5,5),padding='valid')
        self.a1=Activation('sigmoid')
        self.p1=MaxPool2D(pool_size=(2,2),strides=2,padding='valid')

        self.c2=Conv2D(filters=16,kernel_size=(5,5),padding='valid')
        self.a2=Activation('sigmoid')
        self.p2=MaxPool2D(pool_size=(2,2),strides=2,padding='valid')

        self.flatten=Flatten()
        self.f1=Dense(120,activation='sigmoid')
        self.f2=Dense(84,activation='sigmoid')
        self.f3=Dense(10,activation='softmax')

    def call(self, inputs, training=None, mask=None):
        x=self.c1(inputs)
        x=self.a1(x)
        x=self.p1(x)

        x=self.c2(x)
        x=self.a2(x)
        x=self.p2(x)

        x=self.flatten(x)
        x=self.f1(x)
        x=self.f2(x)
        y=self.f3(x)
        return y

model=LeNet()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

checkpoint_path='./CNN/LeNet_ckpt/ckpt'

if os.path.exists(checkpoint_path+'.index'):
    print("-"*12+" loading model saved"+"-"*12)
    model.load_weights(checkpoint_path)
checkpoint=tf.keras.callbacks.ModelCheckpoint(checkpoint_path,save_best_only=True,save_weights_only=True)
callback_list=[checkpoint]

history=model.fit(x_train,y_train,batch_size=32,epochs=5,validation_data=(x_test,y_test),
                  validation_freq=1,callbacks=callback_list)

model.summary()

weight_path='./CNN/LeNet_weight.txt'
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