import util

def extract_outputs(net, filepath):
    """
    """

    print "outs"

    for name, blobs in net.blobs.iteritems():

        capacity = util.compute_capacity(net.blobs[name].data.shape)
        print "- ", name, net.blobs[name].data.shape, capacity

        this_name = str(name.replace("/", "_"))
        outpath = filepath + "outs_" + this_name + ".txt" 
        util.save_data(net.blobs[name], outpath) 

