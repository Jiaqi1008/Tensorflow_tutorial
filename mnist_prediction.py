import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.layers import Dense,Flatten
from tensorflow.keras import Model

checkpoint_save_path="./MNIST_data/checkpoint/ckpt"

class MnistModel(Model):
    def __init__(self):
        super(MnistModel,self).__init__()
        self.f=Flatten()
        self.d1=Dense(128,activation='relu')
        self.d2=Dense(10,activation='softmax')

    def call(self, inputs, training=None, mask=None):
        y_f=self.f(inputs)
        y_d1=self.d1(y_f)
        y=self.d2(y_d1)
        return y
model=MnistModel()

model.load_weights(checkpoint_save_path)

preNum=int(input("Input the number of test pictures:"))

for i in range(preNum):
    image_path=input("Input the path of test picture:")
    image=Image.open('./MNIST_data/'+image_path)
    image=image.resize((28,28),Image.ANTIALIAS)
    image_arr=np.array(image.convert("L"))

    for i in range(28):
        for j in range(28):
            if image_arr[i][j]<200:
                image_arr[i][j]=255
            else:
                image_arr[i][j]=0

    image_arr=image_arr/255
    x_predict=image_arr[np.newaxis,...]
    result=model.predict(x_predict)
    pred=tf.argmax(result,axis=1)
    print("\n")
    tf.print(pred)
    print("\n")