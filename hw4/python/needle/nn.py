"""The module.
"""
from typing import List
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np

class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(in_features, out_features, requires_grad=True, device=device))
        self.bias = Parameter(init.kaiming_uniform(out_features, 1, requires_grad=True, device=device).transpose()) if bias else None
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        out = ops.matmul(X, self.weight)
        if self.bias:
            out += ops.broadcast_to(self.bias, out.shape)
        return out  
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        len = 1
        for shape in X.shape[1:]:
            len *= shape
        return X.reshape((X.shape[0], len))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.tanh(x)
        ### END YOUR SOLUTION


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return (1 + ops.exp(-x)) ** (-1)
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        res = x
        for module in self.modules:
            res = module.forward(res)
        return res
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        batch_size, num_classes = logits.shape
        onehot_y = init.one_hot(num_classes,y,device=logits.device)
        res = (ops.logsumexp(logits, axes=(1,)) - (logits * onehot_y).sum(axes=(1,))).sum() / batch_size
        return res
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, requires_grad=True, device=device))
        self.bias = Parameter(init.zeros(dim, requires_grad=True, device=device))
        self.running_mean = init.zeros(dim, device=device) 
        self.running_var = init.ones(dim, device=device)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        batch_size, feature_dim = x.shape
        if self.training:   
            mean = x.sum(axes=(0,)) / batch_size
            self.running_mean = (1-self.momentum)*self.running_mean + self.momentum * mean.data
            mean = mean.reshape((1,feature_dim)).broadcast_to(x.shape)
            var = ((x - mean)**2).sum(axes=(0,)) / batch_size
            self.running_var = (1-self.momentum)*self.running_var + self.momentum * var.data
            var = var.reshape((1,feature_dim)).broadcast_to(x.shape)
            norm = (x - mean) / ((var + self.eps) ** 0.5)
            res = self.weight.reshape((1,feature_dim)).broadcast_to(x.shape) * norm + self.bias.reshape((1,feature_dim)).broadcast_to(x.shape)
            return res
        else:
            norm = (x - self.running_mean.broadcast_to(x.shape)) / ((self.running_var.broadcast_to(x.shape) + self.eps) ** 0.5)
            return self.weight.broadcast_to(x.shape) * norm + self.bias.broadcast_to(x.shape)
        ### END YOUR SOLUTION


class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2,3)).transpose((1,2))


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim,device=device,dtype=dtype,requires_grad=True))
        self.bias = Parameter(init.zeros(dim,device=device,dtype=dtype,requires_grad=True))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        batch_size, feature_dim = x.shape
        mean = x.sum(axes=(1,)).reshape((batch_size,1)).broadcast_to(x.shape) / feature_dim
        var = ((x - mean)**2).sum(axes=(1,)).reshape((batch_size,1)).broadcast_to(x.shape) / feature_dim
        res = self.weight.reshape((1,feature_dim)).broadcast_to(x.shape) * (x - mean) / ((var + self.eps)**(1/2)) + self.bias.reshape((1,feature_dim)).broadcast_to(x.shape)
        return res
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            res = x * init.randb(*x.shape,p=self.p) / (1-self.p)
        else:
            res = x 
        return res
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x) + x
        ### END YOUR SOLUTION

class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(
            fan_in = self.in_channels*self.kernel_size*self.kernel_size,
            fan_out = self.out_channels*self.kernel_size*self.kernel_size,
            shape = (self.kernel_size,self.kernel_size,self.in_channels,self.out_channels), 
            device = device, dtype = dtype, requires_grad=True)) 
        self.padding = (self.kernel_size-1)//2
        if bias:
            interval = 1.0/(self.in_channels * self.kernel_size**2)**0.5
            self.bias = Parameter(init.rand(self.out_channels,low = -interval, high=interval, device = device, dtype = dtype, requires_grad=True))
        else:
            self.bias = None
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        #NCHW->NHWC
        x = x.transpose((1,2)).transpose((2,3))
        out = ops.conv(x,self.weight,padding=self.padding,stride=self.stride)
        if self.bias:
            out = out + self.bias.reshape((1,1,1,self.out_channels)).broadcast_to(out.shape)
        return out.transpose((2,3)).transpose((1,2))
        ### END YOUR SOLUTION


class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        assert nonlinearity == 'tanh' or nonlinearity == 'relu'
        if nonlinearity == 'tanh':
            self.activate = Tanh()
        else:
            self.activate = ReLU()
        k = np.sqrt(1.0/hidden_size)
        self.device = device
        self.dtype = dtype
        self.hidden_size = hidden_size
        self.W_ih = Parameter(init.rand(input_size,hidden_size,low = -k, high=k,device=device,dtype=dtype,requires_grad=True))
        self.W_hh = Parameter(init.rand(hidden_size,hidden_size,low = -k, high=k,device=device,dtype=dtype,requires_grad=True))
        self.bias_ih = Parameter(init.rand(hidden_size, low=-k, high=k, device=device, dtype=dtype, requires_grad=True)) if bias else None
        self.bias_hh = Parameter(init.rand(hidden_size, low=-k, high=k, device=device, dtype=dtype, requires_grad=True)) if bias else None
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
        for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        bs, _ = X.shape
        if h is None:
            h = init.zeros(bs, self.hidden_size, device=self.device, dtype=self.dtype)
        if self.bias_hh:   
            h_prime = self.activate(X@self.W_ih + self.bias_ih.reshape((1,self.hidden_size)).broadcast_to(h.shape) + h@self.W_hh + self.bias_hh.reshape((1,self.hidden_size)).broadcast_to(h.shape))
        else:
            h_prime = self.activate(X@self.W_ih  + h@self.W_hh)
        return h_prime
        ### END YOUR SOLUTION


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.num_layers = num_layers
        self.rnn_cells = [RNNCell(input_size,hidden_size,bias,nonlinearity,device,dtype)]
        for i in range(num_layers-1):
            self.rnn_cells.append(RNNCell(hidden_size,hidden_size,bias,nonlinearity,device,dtype))
        ### END YOUR SOLUTION

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs:
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION

        if h0 is None:
            h0 = [None]*self.num_layers
        else:
            h0 = ops.split(h0,0)
        final_hiddens = []
        X_spilits = ops.split(X,0)
        for layer_num, rnn_cell in enumerate(self.rnn_cells):
            h_t = h0[layer_num]
            h_nexts=[]
            for x_t in X_spilits:
                h_next = rnn_cell(x_t,h_t)
                h_nexts.append(h_next)
                h_t = h_next
            final_hiddens.append(h_next)
            X_spilits = h_nexts
        return ops.stack(X_spilits,0),ops.stack(final_hiddens,0)   
        ### END YOUR SOLUTION


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.tanh = Tanh()
        self.sigmoid = Sigmoid()
        self.device = device
        self.dtype = dtype
        self.hidden_size = hidden_size
        k = np.sqrt(1.0/hidden_size)

        self.W_ih = Parameter(init.rand(input_size,4*hidden_size,low = -k, high=k,device=device,dtype=dtype,requires_grad=True))
        self.W_hh = Parameter(init.rand(hidden_size,4*hidden_size,low = -k, high=k,device=device,dtype=dtype,requires_grad=True))
        self.bias_ih = Parameter(init.rand(4*hidden_size, low=-k, high=k, device=device, dtype=dtype, requires_grad=True)) if bias else None
        self.bias_hh = Parameter(init.rand(4*hidden_size, low=-k, high=k, device=device, dtype=dtype, requires_grad=True)) if bias else None
        ### END YOUR SOLUTION


    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        bs, _ = X.shape
        if h:
            h0,c0 = h
        else:
            h0,c0 = init.zeros(bs, self.hidden_size, device=self.device, dtype=self.dtype),init.zeros(bs, self.hidden_size, device=self.device, dtype=self.dtype)
        #(batch_size,4*hidden_size)
        if self.bias_hh:
            ifgo = X @ self.W_ih + self.bias_ih.reshape((1,4*self.hidden_size)).broadcast_to((bs,4*self.hidden_size)) + h0@self.W_hh + self.bias_hh.reshape((1,4*self.hidden_size)).broadcast_to((bs,4*self.hidden_size))
        else:
            ifgo = X @ self.W_ih + h0 @ self.W_hh
        ifgo_l = list(ops.split(ifgo,1))
        i_l,f_l,g_l,o_l = ifgo_l[:self.hidden_size],ifgo_l[self.hidden_size:2*self.hidden_size],ifgo_l[2*self.hidden_size:3*self.hidden_size],ifgo_l[3*self.hidden_size:]
        i,f,g,o = ops.stack(i_l,1),ops.stack(f_l,1),ops.stack(g_l,1),ops.stack(o_l,1)
        i,f,g,o = self.sigmoid(i),self.sigmoid(f),self.tanh(g),self.sigmoid(o)
        c_prime = f*c0+i*g
        h_prime = o*self.tanh(c_prime)
        return h_prime,c_prime
        ### END YOUR SOLUTION

class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        ### BEGIN YOUR SOLUTION
        self.num_layers = num_layers
        self.lstm_cells=[LSTMCell(input_size,hidden_size,bias,device,dtype)]
        for i in range(self.num_layers-1):
            self.lstm_cells.append(LSTMCell(hidden_size,hidden_size,bias,device,dtype))
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            c_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        if h:
            h0,c0 = ops.split(h[0],0), ops.split(h[1],0)
        
        final_hiddens = []
        final_cells = []
        X_spilits = ops.split(X,0)
        for layer_num, lstm_cell in enumerate(self.lstm_cells):
            h_t = h0[layer_num] if h else None
            c_t = c0[layer_num] if h else None
            h_nexts=[]
            for x_t in X_spilits:
                if h_t:
                    h_next,c_next = lstm_cell(x_t,(h_t,c_t))
                else:
                    h_next,c_next = lstm_cell(x_t)
                h_nexts.append(h_next)
                h_t = h_next
                c_t = c_next
            final_hiddens.append(h_next)
            final_cells.append(c_next)
            X_spilits = h_nexts

        return ops.stack(X_spilits,0),(ops.stack(final_hiddens,0),ops.stack(final_cells,0))
        ### END YOUR SOLUTION


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        ### BEGIN YOUR SOLUTION
        self.device = device
        self.dtype = dtype
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = Parameter(init.randn(num_embeddings, embedding_dim, mean=0.0, std=1.0, device=device, dtype=dtype, requires_grad=True))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        ### BEGIN YOUR SOLUTION
        seq_len, bs = x.shape
        x_onehot = init.one_hot(self.num_embeddings,x,self.device,self.dtype)
        embed = x_onehot.reshape((seq_len*bs,self.num_embeddings))@self.weight
        return embed.reshape((seq_len,bs,self.embedding_dim))
        ### END YOUR SOLUTION