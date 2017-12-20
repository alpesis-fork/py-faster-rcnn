"""
Caffe2Wellframe

  Usage:   

    $ workon caffe
    $ python caffe2wellframe.py -c <caffe_path> \ 
                                -cn <net_proto> \
                                -cm <model_path> \
                                -wm <wellframe_model_path> \
                                -wo <wellframe_out_path>
"""

import sys
import ctypes
import struct
import argparse


from tensor_and_model import TensorFloat


def get_args():
    """
    """

    parser = argparse.ArgumentParser("Caffe Extractor")

    parser.add_argument('-c', dest='caffepath',
                              type=str,
                              required=True,
                              help='Caffe path')

    parser.add_argument('-cn', dest='net',
                               type=str,
                               required=True,
                               help='Network prototxt')

    parser.add_argument('-cm', dest='model', 
                               type=str,
                               required=True,
                               help='Caffemodel path')

    parser.add_argument('-wm', dest='modelpath',
                               type=str,
                               required=True,
                               help='Wellframe model path')

    parser.add_argument('-wo', dest='outpath',
                               type=str,
                               required=True,
                               help='Wellframe out path')

    return parser.parse_args()


def extract_weights(net, filepath):
    """
    """

    print "weights"

    # weights
    layer_size = 0
    for name, blobs in net.params.iteritems():
        for index in range(len(blobs)):
            print "- ", name, blobs[index].data.shape
            if index == 0:
                filename = filepath + "weights_" + name + "_w.bin" 
            elif index == 1:
                filename = filepath + "weights_" + name + "_b.bin"
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
        filename = outpath + "outs_" + name + ".bin"

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


if __name__ == "__main__":

    args = get_args()
    sys.path.append(args.caffepath)
    import caffe

    net = caffe.Net(args.net, args.model, caffe.TEST)
    extract_weights(net, args.modelpath)
    extract_outputs(net, args.outpath)
