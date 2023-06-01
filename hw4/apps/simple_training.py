import sys
sys.path.append('../python')
import needle as ndl
import needle.nn as nn
from needle import backend_ndarray as nd
from models import *
import time

### CIFAR-10 training ###

def epoch_general_cifar10(dataloader, model, loss_fn=nn.SoftmaxLoss(), opt=None):
    """
    Iterates over the dataloader. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    average_loss = 0.0
    average_accuracy = 0.0
    total = len(dataloader.dataset)
    batch_size = dataloader.batch_size
    if opt:
        model.train()
        for i, batch in enumerate(dataloader):
            #training process
            data, labels = batch
            logits = model(data)
            loss = loss_fn()(logits,labels)
            opt.reset_grad()
            loss.backward()
            opt.step()
            #statistical 
            average_loss += loss.numpy()
            logits = logits.numpy()
            labels = labels.numpy()
            prediction = np.argmax(logits,axis=1).reshape(labels.shape)
            average_accuracy += np.sum(labels==prediction)
    else:
        model.eval()
        for i, batch in enumerate(dataloader):
            #training process
            data, labels = batch
            logits = model(data)
            loss = loss_fn()(logits,labels)
            #statistical 
            average_loss += loss.numpy()
            logits = logits.numpy()
            labels = labels.numpy()
            prediction = np.argmax(logits,axis=1).reshape(labels.shape)
            average_accuracy += np.sum(labels==prediction)

    return average_accuracy/total, average_loss/total * batch_size
    ### END YOUR SOLUTION

def train_cifar10(model, dataloader, n_epochs=1, optimizer=ndl.optim.Adam,
          lr=0.001, weight_decay=0.001, loss_fn=nn.SoftmaxLoss):
    """
    Performs {n_epochs} epochs of training.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    opt = optimizer(params=model.parameters(),lr=lr,weight_decay=weight_decay)
    for i in range(n_epochs):
        print(f"{i+1}/{n_epochs}............")
        avg_acc, avg_loss = epoch_general_cifar10(dataloader,model,loss_fn,opt)
        print(f"avg_acc is {avg_acc}, avg_loss is {avg_loss}")
    return avg_acc, avg_loss
    ### END YOUR SOLUTION


def evaluate_cifar10(model, dataloader, loss_fn=nn.SoftmaxLoss):
    """
    Computes the test accuracy and loss of the model.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    avg_acc, avg_loss = epoch_general_cifar10(dataloader,model,loss_fn)
    return avg_acc, avg_loss
    ### END YOUR SOLUTION



### PTB training ###
def epoch_general_ptb(data, model, seq_len=40, loss_fn=nn.SoftmaxLoss(), opt=None,
        clip=None, device=None, dtype="float32"):
    """
    Iterates over the data. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        data: data of shape (nbatch, batch_size) given from batchify function
        model: LanguageModel instance
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    nbatch, batch_size = data.shape
    loss_l = []
    acc_l = []
    sample_num = 0
    if opt:
        model.train()
    else:
        model.eval()
    for i in range(0,nbatch,seq_len):
        #train
        x, target = ndl.data.get_batch(data,i,seq_len,device,dtype)
        logits,_ = model(x)
        loss = loss_fn(logits,target)
        if opt:
            opt.reset_grad()
            loss.backward()
            if clip:
                opt.clip_grad_norm(clip)
            opt.step()
        #statistical
        loss_l.append(loss.numpy())
        prediction = np.argmax(logits.numpy(),axis=1)
        sample_num += target.shape[0]
        acc_l.append(np.sum(target.numpy()==prediction))
    return np.sum(acc_l)/sample_num, np.average(loss_l)


def train_ptb(model, data, seq_len=40, n_epochs=1, optimizer=ndl.optim.SGD,
          lr=4.0, weight_decay=0.0, loss_fn=nn.SoftmaxLoss, clip=None,
          device=None, dtype="float32"):
    """
    Performs {n_epochs} epochs of training.

    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    opt = optimizer(model.parameters(),lr=lr,weight_decay=weight_decay)
    for i in range(n_epochs):
        avg_acc,avg_loss = epoch_general_ptb(data,model,seq_len,loss_fn(),opt,clip,device,dtype)
        print(f"{i+1}/{n_epochs}...........")
        print(f"avg_acc is {avg_acc}")
        print(f"avg_loss is {avg_loss}")
    return avg_acc, avg_loss
    ### END YOUR SOLUTION


def evaluate_ptb(model, data, seq_len=40, loss_fn=nn.SoftmaxLoss,
        device=None, dtype="float32"):
    """
    Computes the test accuracy and loss of the model.

    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    avg_acc, avg_loss = epoch_general_ptb(data,model,seq_len,loss_fn(),device=device,dtype=dtype)
    return avg_acc, avg_loss
    ### END YOUR SOLUTION


if __name__ == "__main__":
    ### For testing purposes
    device = ndl.cuda()
    # dataset = ndl.data.CIFAR10Dataset("/home/fjy/needle/hw4/data/cifar-10-batches-py", train=True)
    # dataloader = ndl.data.DataLoader(\
    #         dataset=dataset,
    #         batch_size=128,
    #         shuffle=True,
    #         device = device,
    #         )
    
    # model = ResNet9(device=device, dtype="float32")
    # print("training...................")
    # train_cifar10(model, dataloader, n_epochs=50, optimizer=ndl.optim.Adam,
    #      lr=0.001, weight_decay=0.001)
    print("training...................")
    corpus = ndl.data.Corpus("/home/fjy/needle/hw4/data/ptb",max_lines=5000)
    seq_len = 40
    batch_size = 16
    hidden_size = 100
    train_data = ndl.data.batchify(corpus.train, batch_size, device=device, dtype="float32")
    model = LanguageModel(1, len(corpus.dictionary), hidden_size, num_layers=2, device=device)
    train_ptb(model, train_data, seq_len, n_epochs=10, device=device)
