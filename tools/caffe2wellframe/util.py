def compute_capacity(shape):
    capacity = 1
    for d in range(len(shape)):
        capacity *= shape[d]

    return capacity


def save_data(blob, filepath):

    f = open(filepath, "w")

    dim = len(blob.data.shape)

    if dim == 0:
        pass

    elif dim == 1:
        for n in range(blob.data.shape[0]):
            f.write("{0},".format(blob.data[...][n]))

    elif dim == 2:
        for n in range(blob.data.shape[0]):
            for c in range(blob.data.shape[1]):
                f.write("{0},".format(blob.data[...][n,c]))

    elif dim == 3:
        for n in range(blob.data.shape[0]):
            for c in range(blob.data.shape[1]):
                for h in range(blob.data.shape[2]):
                    f.write("{0},".format(blob.data[...][n,c].item(h))) 

    elif dim == 4:
        for n in range(blob.data.shape[0]):
            for c in range(blob.data.shape[1]):
                for h in range(blob.data.shape[2]):
                    for w in range(blob.data.shape[3]):
                        f.write("{0},".format(blob.data[...][n,c].item(h, w)))

    else:
        raise "Weight_dim: %d" % (dim)

    f.close()

