import os
import numpy as np

def softmax(z):
    # We provide numerically stable softmax.
    z = z - np.max(z, axis=1, keepdims=True)
    _exp = np.exp(z)

    _sum = np.sum(_exp, axis=1, keepdims=True)
    sm = _exp / _sum

    return sm

def zero_pad(x, pad):
    ########################################################################
    # Zero padding
    # Given x and pad value, pad input 'x' around height & width.
    #
    # [Input]
    # x: 4-D input batch data
    # - Shape : (# data, In Channel, Height, Width)
    #
    # pad: pad value. how much to pad on one side.
    # e.g. pad=2 => pad 2 zeros on left, right, up & down.
    #
    # [Output]
    # padded_x : padded x
    # - Shape : (Batch size, In Channel, Padded_Height, Padded_Width)
    ########################################################################

    padded_x = None
    N, C, H, W = x.shape

    H += 2 * pad
    W += 2 * pad
    padded_x = np.zeros((N, C, H, W))
    padded_x[:, :, pad: -pad, pad: -pad] += x

    return padded_x

def load_mnist(data_path):
    x_train = np.load(os.path.join(data_path, 'mnist_train_x.npy'))
    y_train = np.load(os.path.join(data_path, 'mnist_train_y.npy'))
    x_test = np.load(os.path.join(data_path, 'mnist_test_x.npy'))
    y_test = np.load(os.path.join(data_path, 'mnist_test_y.npy'))

    x_train = x_train.reshape(len(x_train), 1, 28, 28)
    x_test = x_test.reshape(len(x_test), 1, 28, 28)

    # Y as one-hot
    y_train = np.eye(10)[y_train]
    y_test = np.eye(10)[y_test]
    print(x_train.shape, x_test.shape)

    return x_train, y_train, x_test, y_test


def rel_error(x, y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


def check_conv_validity(x, w, stride, pad):
    N, C, W, H = x.shape
    F, _, WW, HH = w.shape

    # ensure square kernel
    assert HH == WW

    # # in_channel of input and kernel should match.
    if x.shape[1] != w.shape[1]:
        raise ValueError('Input and kernel # channel mismatch.')

    # input width, height >= kernel
    if (W + 2 * pad) < WW or (H + 2 * pad) < WW:
        raise ValueError('Kernel size is larger than input size')
    # kernel size & stride mismatch with input
    if (W + 2 * pad - WW) % stride != 0 or (H + 2 * pad - WW) % stride != 0:
        remain = (W + 2 * pad - WW) % stride if (W + 2 * pad - WW) % stride != 0 else (H + 2 * pad - WW) % stride
        raise ValueError(
            f'Input size {W}, Kernel size {WW}, stride {stride}, pad {pad} mismatch: {remain} row(s) or column(s) remain.')


def check_pool_validity(x, HH, stride):
    N, C, W, H = x.shape

    # input width, height >= kernel
    if W < HH or H < HH:
        raise ValueError('Pooling size is larger than input size')
    # kernel size & stride mismatch with input
    if (W - HH) % stride != 0 or (H - HH) % stride != 0:
        remain = (W - HH) % stride if (W - HH) % stride != 0 else (H - HH) % stride
        raise ValueError(
            f'Input size {W}, Pooling size {HH}, stride {stride} mismatch: {remain} row(s) or column(s) remain.')