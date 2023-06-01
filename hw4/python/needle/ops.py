"""Operatpr table."""
# Global operator table.
from numbers import Number
from typing import Optional, List
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
from . import init
import numpy

from .backend_selection import array_api, NDArray


class MakeTensorTuple(TensorTupleOp):
    def compute(self, *args) -> tuple:
        return tuple(args)

    def gradient(self, out_grad, node):
        assert isinstance(out_grad, TensorTuple)
        return tuple([out_grad[i] for i in range(len(out_grad))])


def make_tuple(*args):
    return MakeTensorTuple()(*args)


class TupleGetItem(TensorOp):
    def __init__(self, index):
        self.index = index

    def __call__(self, a: TensorTuple, fold_const=True) -> Value:
        assert isinstance(a, TensorTuple)
        # constant folding
        if fold_const and isinstance(a.op, MakeTensorTuple):
            return a.inputs[self.index]
        return Tensor.make_from_op(self, [a])

    def compute(self, a):
        return a[self.index]

    def gradient(self, out_grad, node):
        index = self.index
        in_grad = []
        for i, value in enumerate(node.inputs[0]):
            if i != index:
                in_grad.append(init.zeros_like(value))
            else:
                in_grad.append(out_grad)
        return MakeTensorTuple()(*in_grad)


def tuple_get_item(value, index):
    return TupleGetItem(index)(value)


class FusedAddScalars(TensorTupleOp):
    def __init__(self, c0: float, c1: float):
        self.c0 = c0
        self.c1 = c1

    def compute(self, a):
        return a + self.c0, a + self.c1

    def gradient(self, out_grad, node):
        return out_grad[0] + out_grad[1]


def fused_add_scalars(x, c0, c1):
    return FusedAddScalars(c0, c1)(x)


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + numpy.float32(self.scalar)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * numpy.float32(self.scalar)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a ** self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        s = self.scalar
        return (s * a**(s-1)) * out_grad
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return  a / b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs
        return  out_grad/b, (-a/(b**2)) * out_grad 
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a / self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        s = self.scalar
        return out_grad/s
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        permuted = [i for i in range(a.ndim)]
        if self.axes:
            permuted[self.axes[0]]=self.axes[1]
            permuted[self.axes[1]]=self.axes[0]
            return a.permute(permuted)
        else:
            permuted[a.ndim-1]=a.ndim-2
            permuted[a.ndim-2]=a.ndim-1
            return a.permute(permuted)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad.transpose(self.axes)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.reshape(a.compact(),self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        return out_grad.reshape(a.shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape).compact()

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        axes = []
        #broadcast can append new dimension in the front
        shift = len(out_grad.shape) - len(a.shape)
        axes.extend(range(shift))
        #if broadcast at the dimension, we add the dimention into axes
        for i in range(len(a.shape)):
            if a.shape[i]==1 and self.shape[i+shift]>1:
                axes.append(i+shift)
        #sum over all broadcast dimensions and reshape
        return out_grad.sum(tuple(axes)).reshape(a.shape)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if self.axes is None:
            return array_api.summation(a, axis=None)
        else:
            for axis in sorted(self.axes, reverse=True):
                a = array_api.summation(a, axis=axis)
            return a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        origin_shape = node.inputs[0].shape
        sum_axis = range(len(origin_shape)) if self.axes is None else self.axes
        fix_shape = list(origin_shape)
        for i in sum_axis:
            fix_shape[i] = 1
        return broadcast_to(out_grad.reshape(fix_shape), origin_shape)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a @ b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs
        agrad, bgrad = out_grad @ transpose(b), transpose(a) @ out_grad 
        # deal with batch matrix multiplication
        if len(agrad.shape) > len(a.shape):
            agrad = agrad.sum(tuple([i for i in range(len(agrad.shape)-len(a.shape))]))
        if len(bgrad.shape) > len(b.shape):
            bgrad = bgrad.sum(tuple([i for i in range(len(bgrad.shape)-len(b.shape))]))  

        return agrad, bgrad 
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return -out_grad
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        return out_grad/a
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        return out_grad * exp(a)
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(a,0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * Tensor(node.realize_cached_data() > 0, dtype="float32", device=node.device)
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        Z_max_keepdim = Z.max(axis=self.axes, keepdims=True)
        Z_max = Z.max(axis=self.axes)
        res = array_api.log(array_api.summation(array_api.exp(Z-Z_max_keepdim.broadcast_to(Z.shape)), axis=self.axes)) + Z_max
        return res
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        z = node.inputs[0].realize_cached_data()
        diff = z - z.max(axis=self.axes, keepdims=True).broadcast_to(z.shape)
        diff_exp = array_api.exp(diff)
        data = diff_exp / array_api.summation(diff_exp, axis=self.axes, keepdims=True).broadcast_to(diff_exp.shape) 
        shape = diff_exp.shape
        fix_shape = list(shape)
        for i in range(len(shape)):
            if self.axes is None or i in self.axes:
                fix_shape[i]=1
            elif i == len(shape)-1 and -1 in self.axes:
                fix_shape[i]=1  

        return (Tensor(data, device=data.device) * out_grad.reshape(fix_shape).broadcast_to(shape))
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)


class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.tanh(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * (1 - tanh(node.inputs[0]) ** 2)
        ### END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args):
        ### BEGIN YOUR SOLUTION
        stack_shape = list(args[0].shape)
        arg_len = len(args)
        stack_shape.insert(self.axis, arg_len)
        stack_arr = array_api.empty(shape=stack_shape,device=args[0].device)
        slice_idx = [slice(0,l) for l in stack_shape]
        for i in range(arg_len):
            slice_idx[self.axis]=i
            stack_arr[tuple(slice_idx)] = args[i]
        return stack_arr
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return split(out_grad, self.axis)
        ### END YOUR SOLUTION
        

def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        ### BEGIN YOUR SOLUTION
        stack_shape = list(A.shape)
        arg_len = stack_shape[self.axis]
        slice_idx = [slice(0,l) for l in stack_shape]
        tensors = []
        stack_shape.pop(self.axis)
        for i in range(arg_len):
            slice_idx[self.axis]=i
            tensors.append(A[tuple(slice_idx)].compact().reshape(stack_shape))
        return tuple(tensors)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return stack(out_grad,self.axis)
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.flip(a,self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return flip(out_grad,self.axes)
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)



class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        shape = list(a.shape)
        for axis in self.axes:
            shape[axis] = shape[axis] * (self.dilation+1)
        dilate_arr = array_api.full(shape=shape,fill_value=0,device=a.device)
        slice_idx = [slice(0,len) for len in shape]
        for axis in self.axes:
            slice_idx[axis] = slice(0,slice_idx[axis].stop, self.dilation+1)
        dilate_arr[tuple(slice_idx)]=a
        return dilate_arr
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return undilate(out_grad,self.axes,self.dilation)
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)

class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        shape = list(a.shape)
        for axis in self.axes:
            shape[axis] = shape[axis] / (self.dilation+1)
        slice_idx = [slice(0,len) for len in a.shape]
        for axis in self.axes:
            slice_idx[axis] = slice(0,slice_idx[axis].stop, self.dilation+1)
        undilate_arr=a[tuple(slice_idx)]
        return undilate_arr
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return dilate(out_grad,self.axes,self.dilation)
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        ### BEGIN YOUR SOLUTION
        N,H,W,C_in = A.shape
        A = A.pad(((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)))
        K,_,_,C_out = B.shape
        Ns, Hs, Ws, Cs = A.strides
        inner_dim = K * K * C_in
        strided_A = A.as_strided(shape=(N, (H + 2*self.padding - K + 1) // self.stride, (W +2*self.padding - K + 1) // self.stride, K, K, C_in),
                                 strides=(Ns, Hs * self.stride, Ws * self.stride, Hs, Ws, Cs)).compact(). \
                                reshape((N * (H +2*self.padding- K + 1) // self.stride * (W +2*self.padding- K + 1) // self.stride, inner_dim))
        
        out = strided_A @ (B.compact().reshape((K * K * C_in, C_out)))
        return out.compact().reshape((N, (H +2*self.padding- K + 1) // self.stride, (W +2*self.padding- K + 1) // self.stride, C_out))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        #out_grad.shape = (N, (H + 2P - K + 1) // self.stride, (W + 2P - K + 1) // self.stride, C_out)
        A, B = node.inputs
        N,H,W,C_in = A.shape
        K,_,_,C_out = B.shape
        if self.stride > 1:
            out_grad = dilate(out_grad,(1,2),self.stride-1)
        A_grad = conv(out_grad,flip(B,(0,1)).transpose(),padding=K-1-self.padding)
        B_grad = conv(A.transpose((0,3)),out_grad.transpose((0,1)).transpose((1,2)),padding=self.padding).transpose((0,1)).transpose((1,2))
        return A_grad, B_grad
        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)



