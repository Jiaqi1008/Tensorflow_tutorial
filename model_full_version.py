import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from tensorflow.keras.layers import Dense,Flatten
from tensorflow.keras import Model
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
np.set_printoptions(threshold=np.inf)


# path list
train_path="./MNIST_data/MNIST_data_train_jpg_60000/"
train_txt="./MNIST_data/MNIST_data_train_jpg_60000.txt"
x_train_savepath="./MNIST_data/MNIST_data_x_train.npy"
y_train_savepath="./MNIST_data/MNIST_data_y_train.npy"
test_path="./MNIST_data/MNIST_data_test_jpg_10000/"
test_txt="./MNIST_data/MNIST_data_test_jpg_10000.txt"
x_test_savepath="./MNIST_data/MNIST_data_x_test.npy"
y_test_savepath="./MNIST_data/MNIST_data_y_test.npy"

# data preprocessing
def generateds(path,txt):
    x, y = [], []
    counter,length=0,0
    with open(txt,'r') as file:
        for _ in file:
            length+=1
    with open(txt,'r') as file:
        for line in file:
            counter+=1
            value=line.split()
            image=Image.open(path+value[0])
            image=np.array(image.convert('L'))
            image=image/255
            x.append(image)
            y.append(value[1])
            print("\rLoading:%.2f%%"%(counter/length*100),end='',flush=True)
    print("\n")
    x=np.array(x)
    y=np.array(y)
    y=y.astype(np.int64)
    return x,y

if os.path.exists(x_train_savepath) and os.path.exists(y_train_savepath) and os.path.exists(x_test_savepath) and os.path.exists(y_test_savepath):
    print("-"*12+" load data"+"-"*12)
    x_train_save=np.load(x_train_savepath)
    y_train=np.load(y_train_savepath)
    x_test_save=np.load(x_test_savepath)
    y_test=np.load(y_test_savepath)
    x_train=np.reshape(x_train_save,(len(x_train_save),28,28))
    x_test=np.reshape(x_test_save,(len(x_test_save),28,28))
else:
    print("-"*12+" generate data"+"-"*12)
    x_train,y_train=generateds(train_path,train_txt)
    x_test,y_test=generateds(test_path,test_txt)
    print("-"*12+" save data"+"-"*12)
    x_train_save=np.reshape(x_train,(len(x_train),-1))
    x_test_save=np.reshape(x_test,(len(x_test),-1))
    np.save(x_train_savepath,x_train_save)
    np.save(y_train_savepath, y_train)
    np.save(x_test_savepath, x_test_save)
    np.save(y_test_savepath, y_test)
x_train=x_train.reshape((x_train.shape[0],28,28,1)) #reshape for the data augmentation

# set for data augmentation
image_gen_train=ImageDataGenerator(
    rescale=1./1.,
    rotation_range=45,
    width_shift_range=0.15,
    height_shift_range=0.15,
    horizontal_flip=True,
    zoom_range=0.5
)
image_gen_train.fit(x_train)

# set model
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

# training and testing
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

# save checkpoint during training
checkpoint_save_path="./MNIST_data/checkpoint/ckpt"
if os.path.exists(checkpoint_save_path+'.index'):
    print("-"*12+" loading model saved"+"-"*12)
    model.load_weights(checkpoint_save_path)
checkpoint=tf.keras.callbacks.ModelCheckpoint(checkpoint_save_path,save_best_only=True,save_weights_only=True)
callback_list=[checkpoint]

# adding data augmentation here
history=model.fit(image_gen_train.flow(x_train,y_train,batch_size=32),epochs=5,
          validation_data=(x_test,y_test),validation_freq=1,callbacks=callback_list)

model.summary()

# save the weighs of model, if you want to print them out,just uncomment the lines below
with open("./MNIST_data/weight.txt",'w') as file:
    for v in model.trainable_variables:
        file.write(str(v.name)+'\n')
        file.write(str(v.shape)+'\n')
        file.write(str(v.numpy())+'\n')
# print(model.trainable_variables)

# plot acc and loss curve
acc=history.history['sparse_categorical_accuracy']
val_acc=history.history['val_sparse_categorical_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']

plt.subplot(121)
plt.plot(acc,label='Train Accuracy')
plt.plot(val_acc,label='Validation Accuracy')
plt.title("Train and Validation Accuracy")
plt.legend()

plt.subplot(122)
plt.plot(loss,label='Train Loss')
plt.plot(val_loss,label='Validation Loss')
plt.title("Train and Validation Loss")
plt.legend()

plt.show()