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


from extractor_single import extract_weights
from extractor_single import extract_outputs
from tensor_and_model import TensorFloat


def get_args():
    """
    """


    parser = argparse.ArgumentParser("Caffe Extractor")

    parser.add_argument('-c', dest='caffepath',
                              type=str,
                              required=True,
                              help='Caffe path')

    parser.add_argument('-cf', dest='frcnnpath',
                               type=str,
                               required=True,
                               help='Faster R-CNN lib path')

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



if __name__ == "__main__":

    args = get_args()
    sys.path.append(args.caffepath)
    sys.path.append(args.frcnnpath)
    sys.path.append(args.frcnnpath + "lib/")
    sys.path.append(args.frcnnpath + "tools/")
    import caffe
    import demo

    caffe.set_mode_cpu()
    net = caffe.Net(args.net, args.model, caffe.TEST)
    extract_weights(net, args.modelpath)

    net.forward()
    extract_outputs(net, args.outpath)
