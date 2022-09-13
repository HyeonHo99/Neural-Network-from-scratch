import time
import numpy as np
from utils import load_mnist
from activations import *
from layers import *
from models import ClassifierModel
import matplotlib.pyplot as plt
import copy

np.random.seed(123)
np.tanh = lambda x: x

model = ClassifierModel()

# =============================== EDIT HERE ===============================

"""
    Build model Architecture and do experiment.
    You can add layer as below examples.
    NOTE THAT, layers are executed in the order that you add.
    Enjoy :)

    < Add Layers examples >
    - FC Layer
        model.add_layer('FC Example Layer', FCLayer(input_dim=1234, output_dim=123))
    - Softmax Layer
        model.add_layer('Softmax Layer', SoftmaxLayer())
"""

# mnist
dataset = 'mnist'

# Hyper-parameters
num_epochs = 20
learning_rate = 0.0002
reg_lambda = 0.00001
print_every = 1

batch_size = 128

# ##MLP MODEL
# model.add_layer('FC_1',FCLayer(input_dim=28*28,output_dim=256))
# model.add_layer('FC_1_RELU',ReLU())
# model.add_layer('FC_2',FCLayer(input_dim=256,output_dim=128))
# model.add_layer('FC_2_RELU',ReLU())
# model.add_layer('FC_3',FCLayer(input_dim=128,output_dim=10))
# model.add_layer('Softmax',SoftmaxLayer())

##CNN MODEL
model.add_layer('CONV_1-1',ConvolutionLayer(in_channels=1,out_channels=64,kernel_size=3,stride=1,pad=1))
model.add_layer('CONV_1-1_RELU',ReLU())
model.add_layer('MAX_POOL_1',MaxPoolingLayer(kernel_size=2,stride=2))
# model.add_layer('CONV_1-2',ConvolutionLayer(in_channels=64,out_channels=64,kernel_size=3,stride=1,pad=1))
# model.add_layer('CONV_1-2_RELU',ReLU())
# model.add_layer('MAX_POOL_1',MaxPoolingLayer(kernel_size=2,stride=2))

model.add_layer('CONV_2-1',ConvolutionLayer(in_channels=64,out_channels=128,kernel_size=3,stride=1,pad=0))
model.add_layer('CONV_2-1_RELU',ReLU())
model.add_layer('MAX_POOL_2',MaxPoolingLayer(kernel_size=2,stride=2))

model.add_layer('FC_1',FCLayer(input_dim=6*6*128,output_dim=256))
model.add_layer('FC_3',FCLayer(input_dim=256,output_dim=10))

model.add_layer('Softmax',SoftmaxLayer())
# =========================================================================
assert dataset in ['mnist']

# Dataset
x_train, y_train, x_test, y_test = load_mnist('./data')

# Random 10% of train data as valid data
num_train = len(x_train)
perm = np.random.permutation(num_train)
num_valid = int(len(x_train) * 0.1)

valid_idx = perm[:num_valid]
train_idx = perm[num_valid:]

x_valid, y_valid = x_train[valid_idx], y_train[valid_idx]
x_train, y_train = x_train[train_idx], y_train[train_idx]

num_train, channel, height, width = x_train.shape
num_class = y_train.shape[1]

model.summary()
time.sleep(1)

train_accuracy = []
valid_accuracy = []

best_epoch = -1
best_acc = -1
best_model = None

print('Training Starts...')
num_batch = int(np.ceil(num_train / batch_size))

for epoch in range(1, num_epochs + 1):
    # model Train
    start = time.time()
    epoch_loss = 0.0
    for b in range(num_batch):
        s = b * batch_size
        e = (b+1) * batch_size if (b+1) * batch_size < len(x_train) else len(x_train)

        x_batch = x_train[s:e]
        y_batch = y_train[s:e]

        loss = model.forward(x_batch, y_batch, reg_lambda)
        epoch_loss += loss

        model.backward(reg_lambda)
        model.update(learning_rate)

        print('[%4d / %4d]\t batch loss : %.4f' % (s, num_train, loss))

    end = time.time()
    lapse_time = end - start
    print('Epoch %d took %.2f seconds\n' % (epoch, lapse_time))

    if epoch % print_every == 0:
        # TRAIN ACCURACY
        prob = model.predict(x_train)
        pred = np.argmax(prob, -1).astype(int)
        true = np.argmax(y_train, -1).astype(int)

        correct = len(np.where(pred == true)[0])
        total = len(true)
        train_acc = correct / total
        train_accuracy.append(train_acc)

        # VAL ACCURACY
        prob = model.predict(x_valid)
        pred = np.argmax(prob, -1).astype(int)
        true = np.argmax(y_valid, -1).astype(int)

        correct = len(np.where(pred == true)[0])
        total = len(true)
        valid_acc = correct / total
        valid_accuracy.append(valid_acc)

        print('[EPOCH %d] Loss = %.5f' % (epoch, epoch_loss))
        print('Train Accuracy = %.3f' % train_acc + ' // ' + 'Valid Accuracy = %.3f' % valid_acc)

        if best_acc < valid_acc:
            print('Best Accuracy updated (%.4f => %.4f)' % (best_acc, valid_acc))
            best_acc = valid_acc
            best_epoch = epoch
            best_model = copy.deepcopy(model)

print('Training Finished...!!')
print('Best Valid acc : %.2f at epoch %d' % (best_acc, best_epoch))

# TEST ACCURACY
prob = best_model.predict(x_test)
pred = np.argmax(prob, -1).astype(int)
true = np.argmax(y_test, -1).astype(int)

correct = len(np.where(pred == true)[0])
total = len(true)
test_acc = correct / total

print('Test Accuracy at Best Epoch : %.2f' % (test_acc))