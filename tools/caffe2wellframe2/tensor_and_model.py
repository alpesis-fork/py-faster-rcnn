import ctypes


class TensorShape(ctypes.Structure):

    _fields_ = [("n", ctypes.c_uint),
                ("channels", ctypes.c_uint),
                ("height", ctypes.c_uint),
                ("width", ctypes.c_uint)]


class TensorFloat(ctypes.Structure):

    _fields_ = [("shape", TensorShape),
                ("capacity", ctypes.c_ulong),
                ("data", ctypes.POINTER(ctypes.c_float)),
                ("data_gpu", ctypes.POINTER(ctypes.c_float))]


def test_tensor_shape():

    tensor_shape = TensorShape()
    tensor_shape.n = ctypes.c_uint(4);
    tensor_shape.channels = ctypes.c_uint(100);
    tensor_shape.height = ctypes.c_uint(200);
    tensor_shape.width = ctypes.c_uint(150);

    print(tensor_shape.n)
    print(tensor_shape.channels)
    print(tensor_shape.height)
    print(tensor_shape.width)


def test_tensor_float():

    tensor_shape = TensorShape()
    tensor_shape.n = ctypes.c_uint(1);
    tensor_shape.channels = ctypes.c_uint(2);
    tensor_shape.height = ctypes.c_uint(3);
    tensor_shape.width = ctypes.c_uint(4);
   
 
    tensor = TensorFloat()
    tensor.shape = tensor_shape
    capacity = tensor.shape.n * tensor.shape.channels * tensor.shape.height * tensor.shape.width
    tensor.capacity = ctypes.c_ulong(capacity)
    tensor.data = (ctypes.c_float * tensor.capacity)()
    for i in xrange(tensor.capacity): tensor.data[i] = i

    print(tensor.shape.n)
    print(tensor.shape.channels)
    print(tensor.shape.height)
    print(tensor.shape.width)
    print(tensor.capacity)
    print(tensor.data[tensor.capacity/2])


if __name__ == '__main__':

    test_tensor_shape()
    test_tensor_float()
