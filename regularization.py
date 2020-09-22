import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


# read data
df=pd.read_csv("dot.csv")
x_data=np.array(df[["x1","x2"]])
y_data=np.array(df["y_c"])
x_train=np.vstack(x_data).reshape(-1,2)
y_train=np.vstack(y_data).reshape(-1,1)
y_c=[["red" if y else "blue"] for y in y_train]

# preprocessing data
x_train=tf.cast(x_train,dtype=tf.float32)
y_train=tf.cast(y_train,dtype=tf.float32)
train_db=tf.data.Dataset.from_tensor_slices((x_train,y_train)).batch(32)

# set parameters
lr=0.005
epoch=801

# set model
w1=tf.Variable(tf.random.normal([2,11],dtype=tf.float32,seed=627))
b1=tf.Variable(tf.constant(0.01,shape=[11]))
w2=tf.Variable(tf.random.normal([11,1],dtype=tf.float32,seed=627))
b2=tf.Variable(tf.constant(0.01,shape=[1]))

# train
for epoch in range(epoch):
    for x_train,y_train in train_db:
        with tf.GradientTape() as tape:
            y1=tf.matmul(x_train,w1)+b1
            y1=tf.nn.relu(y1)
            y=tf.matmul(y1,w2)+b2

            # calculate loss without regularization
            # MSE=tf.reduce_mean(tf.square(y_train-y))
            # loss=MSE

            # calculate loss with regularization
            MSE=tf.reduce_mean(tf.square(y_train-y))
            reg=[]
            # L2 regulatization
            reg.append(tf.nn.l2_loss(w1))
            reg.append(tf.nn.l2_loss(w2))
            loss_reg=tf.reduce_sum(reg)
            loss=MSE+0.03*loss_reg
        grad=tape.gradient(loss,[w1,b1,w2,b2])
        # refresh parameters
        w1.assign_sub(lr*grad[0])
        b1.assign_sub(lr*grad[1])
        w2.assign_sub(lr*grad[2])
        b2.assign_sub(lr*grad[3])
    if epoch%50==0:
        print("Epoch:{},Loss:{}".format(epoch,loss))

# test
print("-"*24)
print("Predicting...")
xx,yy=np.mgrid[-3:3:0.1,-3:3:0.1]
grid=np.c_[xx.ravel(),yy.ravel()]
grid=tf.cast(grid,dtype=tf.float32)
pro=[]
for x_test in grid:
    y1_test=tf.matmul([x_test],w1)+b1
    y1_test=tf.nn.relu(y1_test)
    y_test=tf.matmul(y1_test,w2)+b2
    pro.append(y_test)

# plot
x1=x_data[:,0]
x2=x_data[:,1]
plt.scatter(x1,x2,color=np.squeeze(y_c))
pro=np.array(pro).reshape(xx.shape)
plt.contour(xx,yy,pro,levels=[0.5])
plt.show()