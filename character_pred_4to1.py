import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from tensorflow.keras.layers import Dense,SimpleRNN
import numpy as np
from matplotlib import pyplot as plt
np.set_printoptions(threshold=np.inf)

input_data='abcde'
w_id={'a':0,'b':1,'c':2,'d':3,'e':4}
id_onehot={0:[1.,0.,0.,0.,0.],1:[0.,1.,0.,0.,0.],2:[0.,0.,1.,0.,0.],3:[0.,0.,0.,1.,0.],4:[0.,0.,0.,0.,1.]}

x_train=[[id_onehot[w_id['a']],id_onehot[w_id['b']],id_onehot[w_id['c']],id_onehot[w_id['d']]],
         [id_onehot[w_id['b']],id_onehot[w_id['c']],id_onehot[w_id['d']],id_onehot[w_id['e']]],
         [id_onehot[w_id['c']],id_onehot[w_id['d']],id_onehot[w_id['e']],id_onehot[w_id['a']]],
         [id_onehot[w_id['d']],id_onehot[w_id['e']],id_onehot[w_id['a']],id_onehot[w_id['b']]],
         [id_onehot[w_id['e']],id_onehot[w_id['a']],id_onehot[w_id['b']],id_onehot[w_id['c']]]]
y_train=[w_id['e'],w_id['a'],w_id['b'],w_id['c'],w_id['d']]

# input shape to the RNN should be [sample_num,RNN_cell_num,input_feature_num]
x_train=np.reshape(x_train,(len(x_train),4,5))
y_train=np.array(y_train)

np.random.seed(627)
np.random.shuffle(x_train)
np.random.seed(627)
np.random.shuffle(y_train)
tf.random.set_seed(627)

model=tf.keras.Sequential([
    SimpleRNN(3),
    Dense(5,activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

checkpoint_path='./RNN/character_pred_4to1_ckpt/ckpt'

if os.path.exists(checkpoint_path+'.index'):
    print("-"*12+" loading model saved"+"-"*12)
    model.load_weights(checkpoint_path)
checkpoint=tf.keras.callbacks.ModelCheckpoint(checkpoint_path,save_best_only=True,
                                              save_weights_only=True,monitor='loss')#use loss to evaluate model as there is no test data
callback_list=[checkpoint]

history=model.fit(x_train,y_train,batch_size=32,epochs=100,validation_freq=1,callbacks=callback_list)

model.summary()

weight_path='./RNN/character_pred_4to1.txt'
with open(weight_path,'w') as file:
    for v in model.trainable_variables:
        file.write(str(v.name)+'\n')
        file.write(str(v.shape)+'\n')
        file.write(str(v.numpy())+'\n')

acc=history.history['sparse_categorical_accuracy']
loss=history.history['loss']

plt.subplot(121)
plt.plot(acc,label='Accuracy')
plt.title("Train Accuracy")
plt.legend()

plt.subplot(122)
plt.plot(loss,label='Loss')
plt.title("Train Loss")
plt.legend()
plt.show()

#####################################################
preNum=input('Input the number of prediction:')
for i in range(int(preNum)):
    alphabet=input('Input the character:')
    alphabet_onehot=[id_onehot[w_id[a]] for a in alphabet]
    alphabet_onehot=np.reshape(alphabet_onehot,(1,4,5))
    pred=tf.argmax(model.predict(alphabet_onehot),axis=1)
    pred=int(pred)
    tf.print(alphabet+'->'+input_data[pred])


