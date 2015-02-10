""" Simple benchmark for theano's various GPU convolution implementations.
"""
import theano
import theano.tensor as T
from pylearn2.sandbox.cuda_convnet.filter_acts import FilterActs
from theano.sandbox.cuda.basic_ops import gpu_contiguous
import numpy as np
import time

def benchmark_convolution(conv_func, input_shape, kernel_shape):
    """ Benchmarks a convolution with randomly generated data.
    """
    avg_time = 0
    
    for i in range(10):
        inputs = np.random.uniform(-1, 1, input_shape).astype(theano.config.floatX)
        kernels = np.random.uniform(-1, 1, input_shape).astype(theano.config.floatX)
        start = time.clock()
        conv_func(inputs, kernels)
        end = time.clock()
        avg_time += end - start
    return float(avg_time) / 10

if __name__ == "__main__":
    fmaps = T.tensor4('fmaps')
    kernels = T.tensor4('kernels')
    # FFT convolution
    # mode = theano.compile.get_default_mode()
    # mode = mode.including('conv_fft_valid', 'conv_fft_full')
    # fft_conv = theano.function(
    #     [fmaps, kernels],
    #     T.nnet.conv2d(fmaps, kernels),
    #     mode=mode
    # )
    # GEMM convolution
    mode = theano.compile.get_default_mode()
    mode = mode.including('conv_gemm')
    gemm_conv = theano.function(
        [fmaps, kernels],
        T.nnet.conv2d(fmaps, kernels),
        mode=mode
    )
    # cuda-convnet convolution
    conv_op = FilterActs()
    inputs_shuffled = fmaps.dimshuffle(1, 2, 3, 0)
    filters_shuffled = kernels.dimshuffle(1, 2, 3, 0)
    out_shuffled = conv_op(
        gpu_contiguous(inputs_shuffled),
        gpu_contiguous(filters_shuffled[:, ::-1, ::-1, :])
    )
    out_fmaps = out_shuffled.dimshuffle(3, 0, 1, 2)
    cuda_convnet_conv = theano.function(
        [fmaps, kernels],
        out_fmaps
    )

    # cudnn
    mode = theano.compile.get_default_mode()
    mode = mode.including('cudnn')
    cudnn_conv = theano.function(
        [fmaps, kernels],
        T.nnet.conv2d(fmaps, kernels),
        mode=mode
    )
    
    test_shapes = [
        ([128, 3, 128, 128], [96, 3, 11, 11]),
        ([128, 64, 64, 64], [128, 64, 9, 9]),
        ([128, 128, 32, 32], [128, 128, 9, 9]),
        ([128, 128, 16, 16], [128, 128, 7, 7]),
        ([128, 384, 14, 14], [384, 384, 3, 3])
    ]
    
    for shapes in test_shapes:
        inp_shape, ker_shape = shapes
        print shapes
        print "GEMM"
        print benchmark_convolution(gemm_conv, inp_shape, ker_shape)
        print "cuda-convnet"
        print benchmark_convolution(cuda_convnet_conv, inp_shape, ker_shape)
        print "cudnn"
        print benchmark_convolution(cudnn_conv, inp_shape, ker_shape)
