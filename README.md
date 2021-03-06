# Tensorflow_tutorial
This is a code repository for the tensorflow tutorial. The tutorial was published by Peking University, presented by Assistant Professor Jian Cao. Anyone who needs the tutorial videos can find them online in Chinese University Mooc.
## 1.Basic knowledge of Tensorflow ##
There is a simple example of iris classificaiton using the dataset loaded from sklearn library. In this example, `iris_classification.py`, you can find a very strightforward way to realize iris classification by neural network with tensorflow.
## 2.Basic knowledge of Neural networks ##
Learning rate, activation function, loss function, regularizer and optimizer are introduced in this section. Learning rate is a parameter that controls the model's learning speed, or a step model learns. It should be properly selected as if it is too large, the model won't converge, while too small will make the model converge too slow. Activation function is something that makes your output fit some kind of distribution. Loss function helps to evaluate the model and it is also what you rely on to refresh model parameters and optimizer is the way you use to refresh your model. Regularization is a method to help reduce overfitting. 

There is an example, `regularization.py`, that shows the differences when a model has regularization or not. The data is `dot.csv`
## 3.Baseline of setting up a model (six-step method) ##
Basically, all the deep learning model can be set up by six-step method.

**1.Import**

Import the libraries you need in this model.

**2.Training and test data**

In this part, data is read into the system and is divided into training part and test part. Note that data preprocessing like normalization and reshape is also included in this part.

**3.Build model:**

Build the model you want in this part. There are two ways to realize, one being sequential while another class. Sequential model is better for simple structures as you can add the layers you need into the model. Class model is more flexible so it is good for complex structures.

**4.Complie the model**

In thid part you should determine what methods you use to complie model. Optimizer, loss function and metric should be noted in this part.

**5.Fit the model**

Fit model with your training and test data. You need to set up epoch, batch size here.

**6.See the details**

See the detailed information of your model in this step.

`iris_classification_sequential.py` and `iris_classification_class.py` are based on the previous iris classification model while `mnist_class.py` and `fashion_class.py` are class model that use the MNIST dataset and Fashion dataset.

## 4.Advanced model setting up ##

In previous chapter, we use six-step method to build a deep learning model. It can be understood that six-step method is a trunk and we can make the tree better by adding some branched and leaves on. Thus, in this charpter, we add different kinds of components into the model built by six-step method and let it be a classic model for all the deep learning task. You can find use `get_mnist.py` to save MNIST dataset on your device and `model_full_version.py` is the model with all the features. Lastly, you can use `mnist_prediction.py` to realize hand-write digits detection. We add following parts into our model. 

* import self made dataset
* data augment
* checkpoint saving
* parameters saving
* visualization of accurate and loss function
* test data from outside source

## 5. CNN ##

In this section, we use CNN to process pictures. First of all, we can give out the overall definition of the CNN. CNN is just used as feacture extraction tool and the basic steps to realize a CNN process can be divided into five parts called "CBAPD".

* C:Convolution layer
* B:Batch Nomalization layer
* A:Activation layer
* P:Pool layer
* D:Dropout layer

Above five steps can be the overall CNN structure and in practical usage, some of the parts can be omitted. In `cifar.py`, a simple demo has been employed on the data set of cifar 10. Also, you can find the realization of some famous net for computer vision like LeNet, VGGNet, AlexNet, InceptionNet and ResNet in `LeNet.py`, `VGGNet.py`, `AlexNet.py`, `InceptionNet.py` and `ResNet.py` respectly.

## 6.RNN ##

In this section, we use RNN to process pictures. First of all, we can give out the overall definition of the RNN, LSTM and GRU. RNN family structure is also used as feacture extraction tool but it can store the previous knowledge of the training data, which means it is very fit for time-series data as well as the document data. The next problem is how to change non-numerical data into numerical data. One way is called one-hot coding. `character_pred_1to1.py` and `character_pred_4to1.py` use one-hot coding and RNN to realize character prediction. Also, word embedding can be introduced. With the help of the tensorflow embedding layer and RNN, `Embedding_pred_1to1.py` and `Embedding_pred_4to1.py` predict character.

Then, we try to use RNN structure to predict time-series data, like stock. First, we should use API to aquire the stock data in `get_stock.py`. Then, with the RNN, LSTM, GRU layers, stock price can be predicted repectively in `rnn_stock.py`, `lstm_stock.py` and `gru_stock.py`.
