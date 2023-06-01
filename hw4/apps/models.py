import sys
sys.path.append('./python')
import needle as ndl
import needle.nn as nn
import math
import numpy as np
np.random.seed(0)

class ConvBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, device=None, dtype="float32"):
        super().__init__()
        self.Conv2D = nn.Conv(in_channels,out_channels,kernel_size,stride,device=device,dtype=dtype)
        self.batch_norm = nn.BatchNorm2d(dim=out_channels,device=device,dtype=dtype)
        self.activate = nn.ReLU()
    def forward(self, x):
        x = self.Conv2D(x)
        x = self.batch_norm(x)
        out = self.activate(x)
        return out

class ResNet9(nn.Module):
    def __init__(self, device=None, dtype="float32"):
        super().__init__()
        ### BEGIN YOUR SOLUTION ###
        self.conv1 = ConvBN(3,16,7,4,device,dtype)
        self.conv2 = ConvBN(16,32,3,2,device,dtype)
        self.res1 = nn.Residual(nn.Sequential(
            ConvBN(32,32,3,1,device,dtype),
            ConvBN(32,32,3,1,device,dtype)
        ))
        self.conv3 = ConvBN(32,64,3,2,device,dtype)
        self.conv4 = ConvBN(64,128,3,2,device,dtype)
        self.res2 = nn.Residual(nn.Sequential(
            ConvBN(128,128,3,1,device,dtype),
            ConvBN(128,128,3,1,device,dtype)
        ))
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(128,128,device=device,dtype=dtype)
        self.activate = nn.ReLU()
        self.linear2 = nn.Linear(128,10,device=device,dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x):
        ### BEGIN YOUR SOLUTION
        resnet9 = nn.Sequential(
            self.conv1,
            self.conv2,
            self.res1,
            self.conv3,
            self.conv4,
            self.res2,
            self.flatten,
            self.linear1,
            self.activate,
            self.linear2
        )
        out = resnet9(x)
        return out
        ### END YOUR SOLUTION

class LanguageModel(nn.Module):
    def __init__(self, embedding_size, output_size, hidden_size, num_layers=1,
                 seq_model='rnn', device=None, dtype="float32"):
        """
        Consists of an embedding layer, a sequence model (either RNN or LSTM), and a
        linear layer.
        Parameters:
        output_size: Size of dictionary
        embedding_size: Size of embeddings
        hidden_size: The number of features in the hidden state of LSTM or RNN
        seq_model: 'rnn' or 'lstm', whether to use RNN or LSTM
        num_layers: Number of layers in RNN or LSTM
        """
        super(LanguageModel, self).__init__()
        ### BEGIN YOUR SOLUTION
        self.hidden_size = hidden_size
        self.embedding_layer = nn.Embedding(output_size,embedding_size,device,dtype)
        if seq_model == "rnn":
            self.seq_model = nn.RNN(embedding_size,hidden_size,num_layers,device=device,dtype=dtype)
        else:
            self.seq_model = nn.LSTM(embedding_size,hidden_size,num_layers,device=device,dtype=dtype)
        self.linear_layer = nn.Linear(hidden_size,output_size,device=device,dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x, h=None):
        """
        Given sequence (and the previous hidden state if given), returns probabilities of next word
        (along with the last hidden state from the sequence model).
        Inputs:
        x of shape (seq_len, bs)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        Returns (out, h)
        out of shape (seq_len*bs, output_size)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        """
        ### BEGIN YOUR SOLUTION
        seq_len, bs = x.shape
        x_embed = self.embedding_layer(x)
        last_layer_hidden,last_time_hidden = self.seq_model(x_embed,h)
        logits = self.linear_layer(last_layer_hidden.reshape((seq_len*bs,self.hidden_size)))
        return logits, last_time_hidden
        ### END YOUR SOLUTION

if __name__ == "__main__":
    model = ResNet9()
    x = ndl.ops.randu((1, 32, 32, 3), requires_grad=True)
    model(x)
    cifar10_train_dataset = ndl.data.CIFAR10Dataset("data/cifar-10-batches-py", train=True)
    train_loader = ndl.data.DataLoader(cifar10_train_dataset, 128, ndl.cpu(), dtype="float32")
    print(dataset[1][0].shape)