import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from tensorflow.keras.layers import Dense,SimpleRNN,Dropout
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error
import math
from matplotlib import pyplot as plt
np.set_printoptions(threshold=np.inf)


df=pd.read_csv('./SH600519.csv')
train_set=df.iloc[0:2426-300,2:3].values
test_set=df.iloc[2426-300:,2:3].values

# normalize the data
sc=MinMaxScaler(feature_range=(0,1))
train_set_scaled=sc.fit_transform(train_set)
test_set_scaled=sc.transform(test_set)

x_train=[]
y_train=[]
x_test=[]
y_test=[]

for i in range(60,len(train_set_scaled)):
    x_train.append(train_set_scaled[i-60:i,0])
    y_train.append(train_set_scaled[i,0])
x_train,y_train=np.array(x_train),np.array(y_train)
# input shape to the RNN should be [sample_num,RNN_cell_num,input_feature_num]
x_train=np.reshape(x_train,(x_train.shape[0],60,1))

np.random.seed(627)
np.random.shuffle(x_train)
np.random.seed(627)
np.random.shuffle(y_train)
tf.random.set_seed(627)

for i in range(60,len(test_set_scaled)):
    x_test.append(test_set_scaled[i-60:i,0])
    y_test.append(test_set_scaled[i,0])
x_test,y_test=np.array(x_test),np.array(y_test)
# input shape to the RNN should be [sample_num,RNN_cell_num,input_feature_num]
x_test=np.reshape(x_test,(x_test.shape[0],60,1))

model=tf.keras.Sequential([
    SimpleRNN(80,return_sequences=True),
    Dropout(0.2),
    SimpleRNN(100),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer=tf.keras.optimizers.Adam(0.001),loss='mean_squared_error')

checkpoint_path='./RNN/rnn_stock_ckpt/ckpt'

if os.path.exists(checkpoint_path+'.index'):
    print("-"*12+" loading model saved"+"-"*12)
    model.load_weights(checkpoint_path)
checkpoint=tf.keras.callbacks.ModelCheckpoint(checkpoint_path,save_best_only=True,
                                              save_weights_only=True,monitor='val_loss')#use loss to evaluate model as there is no test data
callback_list=[checkpoint]

history=model.fit(x_train,y_train,batch_size=64,epochs=50,validation_data=(x_test,y_test),
                  validation_freq=1,callbacks=callback_list)

model.summary()

weight_path='./RNN/rnn_stock.txt'
with open(weight_path,'w') as file:
    for v in model.trainable_variables:
        file.write(str(v.name)+'\n')
        file.write(str(v.shape)+'\n')
        file.write(str(v.numpy())+'\n')

loss=history.history['loss']
val_loss=history.history['val_loss']

plt.plot(loss,label='Loss')
plt.plot(val_loss,label='Validation Loss')
plt.title("Train and Validation Loss")
plt.legend()
plt.show()

##################predict##############################
predict_stock_price=model.predict(x_test)
predict_stock_price=sc.inverse_transform(predict_stock_price)
real_stock_price=sc.inverse_transform(test_set_scaled[60:])


plt.plot(real_stock_price,color='red',label='real stock price')
plt.plot(predict_stock_price,color='blue',label='predict stock price')
plt.title('stock price prediction')
plt.xlabel('time')
plt.ylabel('stock price')
plt.legend()
plt.show()

##################evaluate##############################
mse=mean_squared_error(predict_stock_price,real_stock_price)
rmse=math.sqrt(mse)
mae=mean_absolute_error(predict_stock_price,real_stock_price)
print('Mean square error:%.6f'%mse)
print('Root mean square error:%.6f'%rmse)
print('Mean absolute error:%.6f'%mae)


