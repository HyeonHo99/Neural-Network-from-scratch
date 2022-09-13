import numpy as np


class ReLU:
    """
    ReLU Function. ReLU(x) = max(0, x)
    Implement forward & backward path of ReLU.

    ReLU(x) = x if x > 0.
              0 otherwise.
    Be careful. It's '>', not '>='.
    """

    def __init__(self):
        # 1 (True) if ReLU input <= 0
        self.zero_mask = None

    def forward(self, z):
        """
        ReLU Forward.
        ReLU(x) = max(0, x)

        z --> (ReLU) --> out

        [Inputs]
            z : ReLU input in any shape.

        [Outputs]
            self.out : Values applied elementwise ReLU function on input 'z'.
        """
        out = None
        self.zero_mask = (z <= 0)
        out = z.copy()
        out[self.zero_mask] = 0

        return out

    def backward(self, d_prev, reg_lambda):
        """
        ReLU Backward.

        z --> (ReLU) --> out
        dz <-- (dReLU) <-- d_prev(dL/dout)

        [Inputs]
            d_prev : Gradients flow from upper layer.
                - d_prev = dL/dk, where k = ReLU(z).
            reg_lambda: L2 regularization weight. (Not used in activation function)
        [Outputs]
            dz : Gradients w.r.t. ReLU input z.
        """
        dz = None
        dz = d_prev.copy()
        dz[self.zero_mask] = 0

        return dz

    def update(self, learning_rate):
        # NOT USED IN ReLU
        pass

    def summary(self):
        return 'ReLU Activation'


class Sigmoid:
    """
    Sigmoid Function.
    Implement forward & backward path of Sigmoid.
    """
    def __init__(self):
        self.out = None

    def forward(self, z):
        """
        Sigmoid Forward.

        z --> (Sigmoid) --> self.out

        [Inputs]
            z : Sigmoid input in any shape.

        [Outputs]
            self.out : Values applied elementwise sigmoid function on input 'z'.
        """
        self.out = None
        self.out = 1/(1+np.exp(-z))

        return self.out

    def backward(self, d_prev, reg_lambda):
        """
        Sigmoid Backward.

        z --> (Sigmoid) --> self.out
        dz <-- (dSigmoid) <-- d_prev(dL/d self.out)

        [Inputs]
            d_prev : Gradients flow from upper layer.
            reg_lambda: L2 regularization weight. (Not used in activation function)

        [Outputs]
            dz : Gradients w.r.t. Sigmoid input z .
        """
        dz = None
        # =============== EDIT HERE ===============
        dz = d_prev * self.out * (1-self.out)

        # =========================================
        return dz

    def update(self, learning_rate):
        # NOT USED IN Sigmoid
        pass

    def summary(self):
        return 'Sigmoid Activation'

class Tanh:
    """
    Hyperbolic Tangent Function(Tanh).
    Implement forward & backward path of Tanh.
    """
    def __init__(self):
        self.out = None

    def forward(self, z):
        """
        Hyperbolic Tangent Forward.

        z --> (Tanh) --> self.out

        [Inputs]
            z : Tanh input in any shape.

        [Outputs]
            self.out : Values applied elementwise tanh function on input 'z'.

        =====CAUTION!=====
        You are not allowed to use np.tanh function!
        """
        self.out = None
        self.out = (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

        return self.out

    def backward(self, d_prev, reg_lambda):
        """
        Hyperbolic Tangent Backward.

        z --> (Tanh) --> self.out
        dz <-- (dTanh) <-- d_prev(dL/d self.out)

        [Inputs]
            d_prev : Gradients flow from upper layer.
            reg_lambda: L2 regularization weight. (Not used in activation function)

        [Outputs]
            dz : Gradients w.r.t. Tanh input z .
            In other words, the derivative of tanh should be reflected on d_prev.
        """
        dz = None
        dz = d_prev * (1-self.out) * (1+self.out)

        return dz

    def update(self, learning_rate):
        pass

    def summary(self):
        return 'Tanh Activation'
