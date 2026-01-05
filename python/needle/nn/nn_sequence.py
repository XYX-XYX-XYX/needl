"""The module.
"""
from typing import List
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module
import math

class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.power_scalar(1 + ops.exp(-x), -1)
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
        
        self.device = device
        self.dtype = dtype
        bound = 1.0 / math.sqrt(hidden_size)
        self.W_ih = Parameter(init.rand(input_size, hidden_size, low=-bound, high=bound, device=device, dtype=dtype))
        self.W_hh = Parameter(init.rand(hidden_size, hidden_size, low=-bound, high=bound, device=device, dtype=dtype))
        if bias is True:
            self.bias_ih = Parameter(init.rand(hidden_size, low=-bound, high = bound, device=device, dtype=dtype))
            self.bias_hh = Parameter(init.rand(hidden_size, low=-bound, high = bound, device=device, dtype=dtype))
        self.nonlinearity = nonlinearity
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
        if h is None:
            h = init.zeros(X.shape[0], self.W_hh.shape[0], device=self.device, dtype=self.dtype)
        z = X @ self.W_ih
        if hasattr(self, 'bias_ih'):
            z = z + self.bias_ih.reshape(tuple([1, self.bias_ih.shape[0]])).broadcast_to(z.shape)
        z = z + h @ self.W_hh
        if hasattr(self, 'bias_hh'):
            z = z + self.bias_hh.reshape(tuple([1, self.bias_hh.shape[0]])).broadcast_to(z.shape)
        if self.nonlinearity == 'tanh':
            output = ops.tanh(z)
        elif self.nonlinearity == 'relu':
            output = ops.relu(z)
        else:
            raise ValueError("RNN Nonlinearty only supports relu and tanh")
        return output

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
        rnn_cells[k].bias_ih: The learnable input-hidde n bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.rnn_cells = [
            RNNCell(input_size if i == 0 else hidden_size, 
                    hidden_size,
                    bias,
                    nonlinearity,
                    device,
                    dtype)
            for i in range(num_layers)
        ]
        ### END YOUR SOLUTION

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTIO
        inputs = ops.split(X, 0)  
        h_n_lists = []
        if h0 is not None:
            h0_tuple = ops.split(h0, 0)
        else:
            h0_tuple = [None] * len(self.rnn_cells)
        for i in range(len(self.rnn_cells)):
            h_t_lists = []
            h = h0_tuple[i] 
            for j in range(X.shape[0]):
                h = self.rnn_cells[i](inputs[j], h)
                h_t_lists.append(h)
            h_n_lists.append(h)
            inputs = h_t_lists
        h_n = ops.stack(h_n_lists, 0)
        h_t = ops.stack(inputs, 0)
        return h_t, h_n
            
                
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
        self.device = device
        self.dtype = dtype
        bound = 1.0 / math.sqrt(hidden_size)
        self.W_ih = Parameter(init.rand(input_size, 4 * hidden_size, low=-bound, high=bound, device=device, dtype=dtype))
        self.W_hh = Parameter(init.rand(hidden_size, 4 * hidden_size, low=-bound, high=bound, device=device, dtype=dtype))
        if bias is True:
            self.bias_ih = Parameter(init.rand(4 * hidden_size, low=-bound, high = bound, device=device, dtype=dtype))
            self.bias_hh = Parameter(init.rand(4 * hidden_size, low=-bound, high = bound, device=device, dtype=dtype))
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
        if h is None:
            h0 = init.zeros(X.shape[0], self.W_hh.shape[0], device=self.device, dtype=self.dtype)
            c0 = init.zeros(X.shape[0], self.W_hh.shape[0], device=self.device, dtype=self.dtype)
            h = (h0, c0)
        
        z = X @ self.W_ih
        if hasattr(self, 'bias_hh'):
            z = z + self.bias_hh.reshape(tuple([1, self.bias_hh.shape[0]])).broadcast_to(z.shape)
        if hasattr(self, 'bias_ih'):
            z = z + self.bias_ih.reshape(tuple([1, self.bias_ih.shape[0]])).broadcast_to(z.shape) 
        
        z = z + h[0] @ self.W_hh
        batch_size = X.shape[0]
        hidden_size = self.W_hh.shape[0]

        z = z.reshape((batch_size, 4, hidden_size))
        i, f, g, o = ops.split(z, 1)
        sig = Sigmoid()
        i = sig(i)
        f = sig(f)
        g = ops.tanh(g)
        o = sig(o)

        c_next = f * h[1] + i * g

        h_next = o * ops.tanh(c_next)
        return h_next, c_next
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
        self.lstm_cells = [
            LSTMCell(
                input_size if i == 0 else hidden_size,
                hidden_size, 
                bias,
                device, 
                dtype
                ) 
            for i in range(num_layers)
        ]
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
        prev_layer_outputs = ops.split(X, 0)
        h_n_list = []
        c_n_list = []

        if h is not None:
            h0 = ops.split(h[0], 0)
            c0 = ops.split(h[1], 0)
        else:
            h0 = [None] * len(self.lstm_cells)
            c0 = [None] * len(self.lstm_cells)
        
        for i, cell in enumerate(self.lstm_cells):
            prev_h = h0[i]
            prev_c = c0[i]
            output_list = []
            for j in range(X.shape[0]):
                output_h, output_c = cell(prev_layer_outputs[j], tuple([prev_h, prev_c]) if prev_h is not None else None)
                prev_h = output_h
                prev_c = output_c
                output_list.append(prev_h)
            h_n_list.append(output_h)
            c_n_list.append(output_c)
            prev_layer_outputs = output_list
        h_n = ops.stack(h_n_list, 0)
        c_n = ops.stack(c_n_list, 0)
        output = ops.stack(output_list, 0)
        return output, (h_n, c_n)
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
        self.num_embeddings = num_embeddings
        self.embeddings = embedding_dim
        self.device = device
        self.dtype = dtype
        self.weight = Parameter(init.randn(num_embeddings, embedding_dim, device=device, dtype=dtype))

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
        x_one_hot = init.one_hot(self.num_embeddings, x, self.device, self.dtype)
        seq_len, bs = x.shape
        output = x_one_hot.reshape([seq_len * bs, self.num_embeddings]) @ self.weight
        return output.reshape([seq_len, bs, self.embeddings])
        ### END YOUR SOLUTION