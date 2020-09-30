import os
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from matplotlib.image import imsave
import itertools


# data loading
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# plot test
# for i in range(9):
#     plt.subplot(3, 3, i + 1)
#     plt.imshow(X_train[i], cmap='gray', interpolation='none')
#     plt.title("Class {}".format(y_train[i]))
# plt.show()

# set paths and create paths if not exist
train_data_path = './MNIST_data/MNIST_data_train_jpg_60000'
train_label_path = './MNIST_data/MNIST_data_train_jpg_60000.txt'
test_data_path = './MNIST_data/MNIST_data_test_jpg_10000'
test_label_path = './MNIST_data/MNIST_data_test_jpg_10000.txt'
if not os.path.exists(train_data_path):
    os.makedirs(train_data_path)
if not os.path.exists(test_data_path):
    os.makedirs(test_data_path)

# build dataset, format of image: "index_label.jpg", format of label file: "name label"
image_counter = itertools.count(0)
print("Training data is building")
with open(train_label_path,"w",encoding="utf-8") as file:
    for image, label in zip(X_train, y_train):
        image_name = next(image_counter)
        image_path = os.path.join(train_data_path, str(image_name)+'_'+str(label)+'.jpg')
        imsave(image_path, image, cmap='gray')
        file.write(str(image_name)+'_' +str(label)+ '.jpg'+' '+str(label)+"\n")
        print('\rProgress: %.2f%%'%((image_name+1)/len(X_train)*100), end='')
print("\nFinished")

print("-"*24)
print("Test data is building")
with open(test_label_path,"w",encoding="utf-8") as file:
    for image, label in zip(X_test, y_test):
        image_name = next(image_counter)
        image_path = os.path.join(test_data_path, str(image_name)+'_'+str(label)+'.jpg')
        imsave(image_path, image, cmap='gray')
        file.write(str(image_name) + '_' + str(label) + '.jpg' + ' ' + str(label) + "\n")
        print('\rProgress: %.2f%%'%((image_name+1-len(X_train))/len(X_test)*100), end='')
print("\nFinished")