import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from tensorflow.keras.layers import Dense,SimpleRNN,Embedding
import numpy as np
from matplotlib import pyplot as plt
np.set_printoptions(threshold=np.inf)

input_data='abcdefghijklmnopqrstuvwxyz'
w_id={'a':0,'b':1,'c':2,'d':3,'e':4,'f':5,'g':6,'h':7,'i':8,'j':9,'k':10,'l':11,'m':12,'n':13,'o':14,'p':15,
      'q':16,'r':17,'s':18,'t':19,'u':20,'v':21,'w':22,'x':23,'y':24,'z':25}

training_set_scaled=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]

x_train=[]
y_train=[]

for i in range(4,26):
    x_train.append(training_set_scaled[i-4:i])
    y_train.append(training_set_scaled[i])
# input shape to the RNN should be [sample_num,RNN_cell_num]
x_train=np.reshape(x_train,(len(x_train),4))
y_train=np.array(y_train)

np.random.seed(627)
np.random.shuffle(x_train)
np.random.seed(627)
np.random.shuffle(y_train)
tf.random.set_seed(627)

model=tf.keras.Sequential([
    Embedding(26,2),
    SimpleRNN(10),
    Dense(26,activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

checkpoint_path='./RNN/Embedding_pred_4to1_ckpt/ckpt'

if os.path.exists(checkpoint_path+'.index'):
    print("-"*12+" loading model saved"+"-"*12)
    model.load_weights(checkpoint_path)
checkpoint=tf.keras.callbacks.ModelCheckpoint(checkpoint_path,save_best_only=True,
                                              save_weights_only=True,monitor='loss')#use loss to evaluate model as there is no test data
callback_list=[checkpoint]

history=model.fit(x_train,y_train,batch_size=32,epochs=100,validation_freq=1,callbacks=callback_list)

model.summary()

weight_path='./RNN/Embedding_pred_4to1.txt'
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
    alphabet_Embedding=[w_id[a] for a in alphabet]
    alphabet_Embedding=np.reshape(alphabet_Embedding,(1,4))
    pred=tf.argmax(model.predict([alphabet_Embedding]),axis=1)
    pred=int(pred)
    tf.print(alphabet+'->'+input_data[pred])


