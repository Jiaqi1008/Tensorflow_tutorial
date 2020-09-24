import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from tensorflow.keras.layers import Dense,Flatten
from tensorflow.keras import Model


fashion=tf.keras.datasets.fashion_mnist
(x_train,y_train),(x_test,y_test)=fashion.load_data()
x_train,x_test=x_train/255,x_test/255

class FashionModel(Model):
    def __init__(self):
        super(FashionModel,self).__init__()
        self.f=Flatten()
        self.d1=Dense(128,activation='relu')
        self.d2=Dense(10,activation='softmax')

    def call(self, inputs, training=None, mask=None):
        y_f=self.f(inputs)
        y_d1=self.d1(y_f)
        y=self.d2(y_d1)
        return y

model=FashionModel()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

model.fit(x_train,y_train,batch_size=32,epochs=5,validation_data=(x_test,y_test),validation_freq=1)

model.summary()