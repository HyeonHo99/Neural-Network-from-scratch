import numpy as np
from utils import check_conv_validity, check_pool_validity, softmax, zero_pad


class FCLayer:
    def __init__(self, input_dim, output_dim):
        # Weight Initialization
        self.W = np.random.randn(input_dim, output_dim) / np.sqrt(input_dim / 2)
        self.b = np.zeros(output_dim)

    def forward(self, x):
        """
        FC Layer Forward.
        Use variables : self.x, self.W, self.b

        [Input]
        x: Input features.
        - Shape : (batch size, In Channel, Height, Width)
        or
        - Shape : (batch size, input_dim)

        [Output]
        self.out : fc result
        - Shape : (batch size, output_dim)
        """
        # Flatten input if needed.
        self.orig_shape = x.shape
        if len(x.shape) > 2:
            batch_size = x.shape[0]
            x = x.reshape(batch_size, -1)

        self.x = x
        self.out = np.dot(self.x, self.W) + self.b

        return self.out

    def backward(self, d_prev, reg_lambda):
        """
        FC Layer Backward.
        Use variables : self.x, self.W

        [Input]
        d_prev: Gradients value so far in back-propagation process.
        reg_lambda: L2 regularization weight. (Not used in activation function)

        [Output]
        dx : Gradients w.r.t input x
        - Shape : (batch_size, input_dim) - same shape as input x
        """
        dx = None  # Gradient w.r.t. input x
        self.dW = None  # Gradient w.r.t. weight (self.W)
        self.db = None  # Gradient w.r.t. bias (self.b)
        dx = np.dot(d_prev, self.W.T)
        self.dW = np.dot(self.x.T, d_prev)
        self.dW += reg_lambda * self.W
        self.db = np.sum(d_prev, axis=0)
        dx = dx.reshape(self.orig_shape)
        return dx

    def update(self, learning_rate):
        self.W -= self.dW * learning_rate
        self.b -= self.db * learning_rate

    def summary(self):
        return 'Input -> Hidden : %d -> %d ' % (self.W.shape[0], self.W.shape[1])



class ConvolutionLayer:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, pad=0):
        # if isinstance(kernel_size, int):
        #     kernel_size = (kernel_size, kernel_size)

        self.w = np.random.randn(out_channels, in_channels, kernel_size, kernel_size)
        self.b = np.zeros(out_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        ##################################################################################################
        # Convolution Layer Forward.
        #
        # [Input]
        # x: 4-D input batch data
        # - Shape : (Batch size, In Channel, Height, Width)
        #
        # [Output]
        # conv : convolution result
        # - Shape : (Conv_Height, Conv_Width)
        # - Conv_Height & Conv_Width can be calculated using 'Height', 'Width', 'W size', 'Stride'
        #
        ##################################################################################################
        batch_size, in_channel, _, _ = x.shape
        conv = self.convolution(x, self.w, self.b, self.stride, self.pad)
        self.output_shape = conv.shape
        return conv

    def convolution(self, x, w, b, stride=1, pad=0):
        #########################################################################################################
        # Convolution Operation.
        #
        # [Input]
        # x: 4-D input batch data
        # - Shape : (Batch size, In Channel, Height, Width)
        # w: 4-D convolution filter
        # - Shape : (Out Channel, In Channel, Kernel Height, Kernel Width)
        # b: 1-D bias
        # - Shape : (Out Channel)
        # - default : None
        # stride : Stride size
        # - dtype : int
        # - default : 1
        # pad: pad value, how much to pad around
        # - dtype : int
        # - default : 0
        #
        # [Output]
        # conv : convolution result
        # - Shape : (Batch size, Out Channel, Conv_Height, Conv_Width)
        # - Conv_Height & Conv_Width can be calculated using 'Height', 'Width', 'Kernel Height', 'Kernel Width'
        #########################################################################################################

        # Check validity
        check_conv_validity(x, w, stride, pad)

        if pad > 0:
            x = zero_pad(x, pad)

        self.x = x

        N, C, H, W = x.shape
        F, _, HH, WW = w.shape
        out_H = ((H - HH) // stride) + 1
        out_W = ((W - WW) // stride) + 1
        conv = np.zeros((N, F, out_H, out_W))

        ##im2col
        self.col = np.zeros((N, C, HH, WW, out_H, out_W))
        for h in range(HH):
            h_end = h + out_H * stride
            for w in range(WW):
                w_end = w + out_W * stride
                self.col[:, :, h, w, :, :] = self.x[:, :, h:h_end:stride, w:w_end:stride]
        self.col = self.col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_H * out_W, -1)
        self.col_w = self.w.reshape((F, -1)).T

        conv = np.dot(self.col, self.col_w)
        conv = conv.reshape(N, out_H, out_W, F)
        conv = conv.transpose(0, 3, 1, 2)
        if self.pad > 0:
            self.x = x[:, :, self.pad:-self.pad, self.pad:-self.pad]
        return conv

    def backward(self, d_prev, reg_lambda):
        ####################################################################
        # Convolution Layer Backward.
        # Compute derivatives w.r.t x, W, b (self.x, self.W, self.b)
        # Apply L2 regularization
        #
        # [Input]
        #   d_prev: Gradients value so far in back-propagation process.
        #   reg_lambda: L2 regularization weight. (Not used in activation function)
        #
        # [Output]
        #   self.dx : Gradient values of input x (self.x)
        #   - Shape : (Batch size, channel, Height, Width)
        ####################################################################
        N, C, H, W = self.x.shape

        F, _, HH, WW = self.w.shape
        _, _, H_filter, W_filter = self.output_shape

        if len(d_prev.shape) < 3:
            d_prev = d_prev.reshape(*self.output_shape)

        self.dw = np.zeros_like(self.w)
        self.db = np.zeros_like(self.b)
        self.dx = np.zeros_like(self.x)
        d_prev = d_prev.transpose(0, 2, 3, 1).reshape(-1, F)

        self.dw = np.dot(self.col.T, d_prev)
        self.dw = self.dw.transpose(1, 0).reshape(F, C, HH, WW)
        self.dw = self.dw + reg_lambda * self.w

        dcol = np.dot(d_prev, self.col_w.T)

        ##col2im
        o_h = (H + 2 * self.pad - HH) // self.stride + 1
        o_w = (W + 2 * self.pad - WW) // self.stride + 1

        col = dcol.reshape(N, o_h, o_w, C, HH, WW).transpose(0, 3, 4, 5, 1, 2)
        img = np.zeros((N, C, H + 2 * self.pad + self.stride - 1, W + 2 * self.pad + self.stride - 1))
        for h in range(HH):
            h_end = h + self.stride * o_h
            for w in range(WW):
                w_end = w + self.stride * o_w
                img[:, :, h:h_end:self.stride, w:w_end:self.stride] += col[:, :, h, w, :, :]
        self.dx = img[:, :, self.pad:self.pad + H, self.pad:self.pad + W]

        self.db = np.sum(d_prev, axis=0)
        return self.dx

    def update(self, learning_rate):
        # Update weights
        self.w -= self.dw * learning_rate
        self.b -= self.db * learning_rate

    def summary(self):
        return 'Filter Size : ' + str(self.w.shape) + ' Stride : %d, Zero padding: %d' % (self.stride, self.pad)



class MaxPoolingLayer:
    def __init__(self, kernel_size, stride=2):
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        ##############################################################################################
        # Max-Pooling Layer Forward. Pool maximum value by striding W.
        #
        # [Input]
        # x: 4-D input batch data
        # - Shape : (Batch size, In Channel, Height, Width)
        #
        # [Output]
        # max_pool : max_pool result
        # - Shape : (Batch size, Out Channel, Pool_Height, Pool_Width)
        # - Pool_Height & Pool_Width can be calculated using 'Height', 'Width', 'Kernel_size', 'Stride'
        ###############################################################################################
        max_pool = None
        N, C, H, W = x.shape
        check_pool_validity(x, self.kernel_size, self.stride)

        self.x = x
        out_h = (H - self.kernel_size) // self.stride + 1
        out_w = (W - self.kernel_size) // self.stride + 1

        col = np.zeros((N, C, self.kernel_size, self.kernel_size, out_h, out_w))

        for h in range(self.kernel_size):
            h_end = h + self.stride * out_h
            for w in range(self.kernel_size):
                w_end = w + self.stride * out_w
                col[:, :, h, w, :, :] = self.x[:, :, h:h_end:self.stride, w:w_end:self.stride]
        col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
        col = col.reshape(-1, self.kernel_size * self.kernel_size)
        self.arg_max = np.argmax(col, axis=1)
        max_pool = np.max(col, axis=1).reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)
        self.output_shape = max_pool.shape
        return max_pool

    def backward(self, d_prev, reg_lambda):
        ##############################################################################################
        # Max-Pooling Layer Backward.
        # In backward pass, max-pool distributes gradients to where it came from in forward pass.
        #
        # [Input]
        #   d_prev: Gradients value so far in back-propagation process.
        #       - Shape can be varies since either Conv. layer or FC-layer can follow.
        #           (Batch_size, Channel, Height, Width) or (Batch_size, FC Dimension)
        #   reg_lambda: L2 regularization weight. (Not used in pooling layer)
        #
        # [Output]
        #   dx : max_pool gradients
        #   - Shape : (batch_size, channel, height, width) - same shape as input x
        ##############################################################################################
        if len(d_prev.shape) < 3:
            d_prev = d_prev.reshape(*self.output_shape)
        N, C, H, W = d_prev.shape
        dx = np.zeros_like(self.x)

        N, C, H, W = self.x.shape
        out_h = (H - self.kernel_size) // self.stride + 1
        out_w = (W - self.kernel_size) // self.stride + 1

        d_prev = d_prev.transpose(0, 2, 3, 1)
        ps = self.kernel_size * self.kernel_size
        dmax = np.zeros((d_prev.size, ps))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = d_prev.flatten()
        dmax = dmax.reshape(d_prev.shape + (ps,))
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        col = dcol.reshape(N, out_h, out_w, C, self.kernel_size, self.kernel_size).transpose(0, 3, 4, 5, 1, 2)
        img = np.zeros((N, C, H + self.stride - 1, W + self.stride - 1))
        for h in range(self.kernel_size):
            h_end = h + self.stride * out_h
            for w in range(self.kernel_size):
                w_end = w + self.stride * out_w
                img[:, :, h:h_end:self.stride, w:w_end:self.stride] += col[:, :, h, w, :, :]
        dx = img[:, :, :H, :W]
        return dx

    def update(self, learning_rate):
        # NOT USED IN MAX-POOL
        pass

    def summary(self):
        return 'Pooling Size : ' + str((self.kernel_size, self.kernel_size)) + ' Stride : %d' % (self.stride)


class SoftmaxLayer:
    def __init__(self):
        # No parameters
        pass

    def forward(self, x):
        """
        Softmax Layer Forward.
        Apply softmax (not log softmax or others...) on axis-1

        Use 'softmax' function above in this file.
        We recommend you see the function.

        [Input]
        x: Score to apply softmax
        - Shape: (batch_size, # of class)

        [Output]
        y_hat: Softmax probability distribution.
        - Shape: (batch_size, # of class)
        """
        self.y_hat = softmax(x)

        return self.y_hat

    def backward(self, d_prev=1, reg_lambda=0):
        """
        Softmax Layer Backward.
        Gradients w.r.t input score.

        That is,
        Forward  : softmax prob = softmax(score)
        Backward : dL / dscore => 'dx'

        Compute dx (dL / dscore).
        Check loss function in HW3 word file.

        [Input]
        d_prev : Gradients flow from upper layer.

        [Output]
        dx: Gradients of softmax layer input 'x'
        """
        batch_size = self.y.shape[0]

        d_s = self.y_hat - self.y
        dx = (d_prev * d_s) / batch_size

        return dx


    def ce_loss(self, y_hat, y):
        """
        Compute Cross-entropy Loss.
        Use epsilon (eps) for numerical stability in log.

        Check loss function in HW3 word file.

        [Input]
        y_hat: Probability after softmax.
        - Shape : (batch_size, # of class)

        y: One-hot true label
        - Shape : (batch_size, # of class)

        [Output]
        self.loss : cross-entropy loss
        - float
        """
        self.loss = None
        eps = 1e-10
        self.y_hat = y_hat
        self.y = y

        label = np.argmax(y, -1)
        log_y_hat = np.log(y_hat + eps)
        self.loss = -log_y_hat[range(y.shape[0]), label].mean()

        return self.loss

    def update(self, learning_rate):
        # Not used in softmax layer.
        pass

    def summary(self):
        return 'Softmax layer'