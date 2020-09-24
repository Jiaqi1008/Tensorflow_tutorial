import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
# step 1 import library
import tensorflow as tf
from sklearn import datasets
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
import numpy as np

# step 2 prepare train and test
# load data
x_data=datasets.load_iris().data
y_data=datasets.load_iris().target

# shuffle data
np.random.seed(627)
np.random.shuffle(x_data)
np.random.seed(627)
np.random.shuffle(y_data)
tf.random.set_seed(627)

# step 3 set your own model
class IrisModel(Model):
    def __init__(self):
        super(IrisModel,self).__init__()
        self.d1=Dense(3,activation='softmax',kernel_regularizer=tf.keras.regularizers.l2())

    def call(self, x, training=None, mask=None):
        y=self.d1(x)
        return y

model=IrisModel()

# step 4 compile model
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

# step 5 fit model
model.fit(x_data,y_data,batch_size=32,epochs=500,validation_split=0.2,validation_freq=20)

# step 6 show summary of model
model.summary()