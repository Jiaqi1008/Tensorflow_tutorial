import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from tensorflow.keras.layers import Dense,Flatten
from tensorflow.keras import Model


minst=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=minst.load_data()
x_train,x_test=x_train/255,x_test/255

class MnistModel(Model):
    def __init__(self):
        super(MnistModel,self).__init__()
        self.f=Flatten()
        self.d1=Dense(128,activation='relu')
        self.d2=Dense(10,activation='softmax')

    def call(self, x, training=None, mask=None):
        y_f=self.f(x)
        y_d1=self.d1(y_f)
        y=self.d2(y_d1)
        return y

model=MnistModel()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

model.fit(x_train,y_train,batch_size=32,epochs=5,validation_data=(x_test,y_test),validation_freq=1)

model.summary()