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
coming soon
