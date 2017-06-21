import util


def extract_weights(net, filepath):
    """
    """

    print "weights"


    # weights
    layer_size = 0
    for name, blobs in net.params.iteritems():

        n_blobs = len(blobs)

        this_name = str(name.replace("/", "_"))
        outpath = filepath + "weights_" + this_name

        # weights
        weight_capacity = util.compute_capacity(blobs[0].data.shape)
        print "- ", name, blobs[0].data.shape, weight_capacity

        weightpath = outpath + "_w.txt"
        util.save_data (blobs[0], weightpath)

        if n_blobs == 1:
            pass
        # bias
        elif n_blobs == 2:
            bias_capacity = util.compute_capacity(blobs[1].data.shape)
            print "- ", name, blobs[1].data.shape, bias_capacity

            biaspath = outpath + "_b.txt"
            util.save_data (blobs[1], biaspath)

        else:
            raise "n_blobs: %d" % (n_blobs)

        layer_size += 1


