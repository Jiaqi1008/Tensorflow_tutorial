import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from sklearn import datasets
import numpy as np
from matplotlib import pyplot as plt


# data loading
X=datasets.load_iris().data
Y=datasets.load_iris().target

# data shuffling
np.random.seed(627)
np.random.shuffle(X)
np.random.seed(627)
np.random.shuffle(Y)
tf.random.set_seed(627)

# data split
X_train=X[:-30]
Y_train=Y[:-30]
X_test=X[-30:]
Y_test=Y[-30:]

# data preprocessing
X_train=tf.cast(X_train,dtype=tf.float32)
X_test=tf.cast(X_test,dtype=tf.float32)
train_db=tf.data.Dataset.from_tensor_slices((X_train,Y_train)).batch(32)
test_db=tf.data.Dataset.from_tensor_slices((X_test,Y_test)).batch(32)

# set model
w=tf.Variable(tf.random.truncated_normal([4,3],stddev=0.1,seed=627))
b=tf.Variable(tf.random.truncated_normal([3],stddev=0.1,seed=627))

# set parameters
lr=0.03
train_loss=[]   #for loss graph
test_acc=[]     #for acc graph
epoch=500
loss_all=0

# train
for epoch in range(epoch):
    for train_x,train_y in train_db:
        with tf.GradientTape() as tape:
            y=tf.matmul(train_x,w)+b
            y=tf.nn.softmax(y)
            y_=tf.one_hot(train_y,depth=3)
            loss=tf.reduce_mean(tf.square(y_-y))
            loss_all+=loss.numpy()
        grad=tape.gradient(loss,[w,b])
    #refresh the parameter
        w.assign_sub(lr*grad[0])
        b.assign_sub(lr*grad[1])
    print("Epoch {}, loss:{}".format(epoch,loss_all/4))
    train_loss.append(loss_all)
    loss_all=0

# test
    total_correct,total_num=0,0
    for test_x,test_y in test_db:
        y=tf.matmul(test_x,w)+b
        y=tf.nn.softmax(y)
        pred=tf.argmax(y,axis=1)
        pred=tf.cast(pred,dtype=test_y.dtype)
        correct=tf.cast(tf.equal(test_y,pred),dtype=tf.int32)
        correct=tf.reduce_sum(correct)
        total_correct+=int(correct)
        total_num+=test_x.shape[0]
    acc=total_correct/total_num
    test_acc.append(acc)
    print("Test accuracy:%.2f%%" %(acc*100))
    print("-"*24)

# plot loss graph
plt.subplot(211)
plt.title('Loss function Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(train_loss,label="$Loss$")
plt.legend()

# plot acc graph
plt.subplot(212)
plt.title('Acc function Curve')
plt.xlabel('Epoch')
plt.ylabel('Acc')
plt.plot(test_acc,label="$Acc$")
plt.legend()
plt.show()