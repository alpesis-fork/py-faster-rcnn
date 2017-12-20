
import sys
import ctypes
import struct
import argparse


from tensor_and_model import TensorFloat



def extract_weights(net, filepath):
    """
    """

    print "weights"

    # weights
    layer_size = 0
    for name, blobs in net.params.iteritems():
        for index in range(len(blobs)):
            print "- ", name, blobs[index].data.shape
            this_name = str(name.replace("/", "_"))
            if index == 0:
                filename = filepath + "weights_" + this_name + "_w.bin" 
            elif index == 1:
                filename = filepath + "weights_" + this_name + "_b.bin"
            else:
                filename = filepath

            this_tensor = TensorFloat()
            dim = len(blobs[index].data.shape)
            this_tensor.capacity = 1
            for d in range(dim):
                this_tensor.capacity *= blobs[index].data.shape[d]
            this_tensor.data = (this_tensor.capacity * ctypes.c_float)()

            data = []
            if dim == 4:
                this_tensor.shape.n = blobs[index].data.shape[0]
                this_tensor.shape.channels = blobs[index].data.shape[1]
                this_tensor.shape.height = blobs[index].data.shape[2]
                this_tensor.shape.width = blobs[index].data.shape[3]
            elif dim == 2:
                this_tensor.shape.n = blobs[index].data.shape[0]
                this_tensor.shape.channels = blobs[index].data.shape[1]
                this_tensor.shape.height = 1
                this_tensor.shape.width = 1
            else:
                pass


            with open(filename, mode="wb") as f:
                f.write(this_tensor)
                f.write(blobs[index].data[...])

        layer_size += 1



def extract_outputs(net, outpath):
    """
    """

    net.forward()

    print "out"


    for name, blobs in net.blobs.iteritems():
        print "- ", name, net.blobs[name].data.shape
        this_name = str(name.replace("/", "_"))
        filename = outpath + "outs_" + this_name + ".bin"

        this_tensor = TensorFloat() 
        dim = len(net.blobs[name].data.shape)
        this_tensor.capacity = 1
        for d in range(dim):
            this_tensor.capacity *= net.blobs[name].data.shape[d]

        data = []
        if dim == 4:
            this_tensor.shape.n = net.blobs[name].data.shape[0]
            this_tensor.shape.channels = net.blobs[name].data.shape[1]
            this_tensor.shape.height = net.blobs[name].data.shape[2]
            this_tensor.shape.width = net.blobs[name].data.shape[3]
        if dim == 2:
            this_tensor.shape.n = net.blobs[name].data.shape[0]
            this_tensor.shape.channels = net.blobs[name].data.shape[1]
            this_tensor.shape.height = 1
            this_tensor.shape.width = 1
        
        with open(filename, mode="wb") as f:
            f.write(this_tensor)
            f.write(net.blobs[name].data[...])

