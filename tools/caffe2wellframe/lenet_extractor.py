"""
LeNet Extractor

  Usage:   

    $ workon caffe
    $ python lenet_extractor.py -c <caffe_path> \ 
                                -cn <net_proto> \
                                -cm <model_path> \
                                -wm <wellframe_model_path> \
                                -wo <wellframe_out_path>
"""

import sys
import argparse


import weights_extractor
import outs_extractor


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



if __name__ == "__main__":

    args = get_args()
    sys.path.append(args.caffepath)
    import caffe

    net = caffe.Net(args.net, args.model, caffe.TEST)
    weights_extractor.extract_weights(net, args.modelpath)

    net.forward()   
    outs_extractor.extract_outputs(net, args.outpath)

