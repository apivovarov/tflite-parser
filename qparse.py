#!/usr/bin/env python3

import tflite.Model
import tflite.BuiltinOperator
import numpy as np
import os
tflite_model_file = os.path.join("../models/", "iv2_quant.tflite")
tflite_model_buf = open(tflite_model_file, "rb").read()
model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)
np.set_printoptions(threshold=100000, linewidth=100)

class TensorWrapper(object):
    """Tensor wrapper for TFLite Tensor"""
    def __init__(self, tensor_idx, tensor, buffer, qnn_params=None):
        self.tensor_idx = tensor_idx
        self.tensor = tensor
        self.buffer = buffer
        self.qnn_params = qnn_params

def get_tensor_value(tensor_wrapper):
    """Get tensor buffer value from given tensor wrapper"""
    assert isinstance(tensor_wrapper, TensorWrapper)

    try:
        from tflite.TensorType import TensorType
    except ImportError:
        raise ImportError("The tflite package must be installed")

    if tensor_wrapper.tensor.Type() == TensorType.UINT8:
        return np.frombuffer(tensor_wrapper.buffer.DataAsNumpy(), dtype=np.uint8).reshape(
            tensor_wrapper.tensor.ShapeAsNumpy())
    if tensor_wrapper.tensor.Type() == TensorType.FLOAT32:
        return np.frombuffer(tensor_wrapper.buffer.DataAsNumpy(), dtype=np.float32).reshape(
            tensor_wrapper.tensor.ShapeAsNumpy())
    if tensor_wrapper.tensor.Type() == TensorType.INT32:
        return np.frombuffer(tensor_wrapper.buffer.DataAsNumpy(), dtype=np.int32).reshape(
            tensor_wrapper.tensor.ShapeAsNumpy())
    if tensor_wrapper.tensor.Type() == TensorType.INT64:
        return np.frombuffer(tensor_wrapper.buffer.DataAsNumpy(), dtype=np.int64).reshape(
            tensor_wrapper.tensor.ShapeAsNumpy())
    if tensor_wrapper.tensor.Type() == TensorType.BOOL:
        return np.frombuffer(tensor_wrapper.buffer.DataAsNumpy(), dtype=np.bool_).reshape(
            tensor_wrapper.tensor.ShapeAsNumpy())
    raise NotImplementedError("Tensor type {} is currently not supported"
                              .format(str(tensor_wrapper.tensor.Type())))


def get_tensors(model, subgraph, tensors_idx_list):
    """Get tensor wrapper list from given TFLite tensor index list"""
    return_list = list()
    for tensor_idx in tensors_idx_list:
        if tensor_idx < 0:
            return_list.append(TensorWrapper(tensor_idx, 0, 0))
            continue

        tensor = subgraph.Tensors(tensor_idx)
        buffer_idx = tensor.Buffer()
        buffer = model.Buffers(buffer_idx)

        # Check if the tensors are quantized. Parse if yes.
        qnn_params = None
        tflite_qnn_params = tensor.Quantization()
        if tflite_qnn_params is not None:
            scale = float(tflite_qnn_params.ScaleAsNumpy())
            zero_point = int(tflite_qnn_params.ZeroPointAsNumpy())
            pmin = float(tflite_qnn_params.MinAsNumpy())
            pmax = float(tflite_qnn_params.MaxAsNumpy())
            # Check that the scale and zero points are valid.
            if scale != 0 or zero_point != 0:
                qnn_params = dict()
                qnn_params['scale'] = scale
                qnn_params['zero_point'] = zero_point
                qnn_params['min'] = pmin
                qnn_params['max'] = pmax
        return_list.append(TensorWrapper(tensor_idx, tensor, buffer, qnn_params))
    return return_list

def print_data(id):
    print("--DATA------------------")
    t = get_tensors(model, subgraph, [id])[0]
    print(t.qnn_params)
    print("--------------------")

def print_bias(id):
    print("--BIAS------------------")
    t = get_tensors(model, subgraph, [id])[0]
    vv = get_tensor_value(t)
    maxv = t.qnn_params['scale'] * 2147483647
    print(t.qnn_params)
    print("bias min, max: ", -maxv, maxv)
    print("int32_t[",vv.size,"] v =", np.array2string(vv, separator=","))
    print("--------------------")

def print_weight(id):
    print("--WEIGHTS------------------")
    t = get_tensors(model, subgraph, [id])[0]
    vv = get_tensor_value(t)
    print(t.qnn_params)
    # OHWI to HWIO
    print("weight shape: ", vv.shape)
    vv = vv.transpose(1,2,3,0).flatten()
    ### Use OHWI format
    #vv = vv.flatten()
    print("uint8_t[",vv.size,"] v =", np.array2string(vv, separator=","))
    print("--------------------")

subgraph = model.Subgraphs(0)

#print_data(id=6)
#print_weight(id=2)
#print_bias(id=0)

#print_data(id=1)
#print_weight(id=5)
#print_bias(id=4)

#print_data(id=3)

import tensorflow.lite as lite

def read_txt_img(fn):
    vv = list()
    with open(fn) as fp:
        for cnt, line in enumerate(fp):
            vv.append(int(line))
    return vv

img = np.array(read_txt_img("cat224-3.txt"), dtype="uint8").reshape((1,224,224,3))
print("img shape", img.shape)
ip = lite.Interpreter(model_path=tflite_model_file)
ip.allocate_tensors()
inp_id = ip.get_input_details()[0]["index"]
out_id = ip.get_output_details()[0]["index"]
print(inp_id, out_id)
ip.set_tensor(inp_id, img)
ip.invoke()
res = ip.get_tensor(out_id)
print(res.shape)
res = res.flatten()
print("OUTPUT");
#for h in range(10,14):
#    print("h:", h)
#    for w in range(20,25):
#        pos = 112 * 24 * h + 24 * w
#        print(res[pos:pos + 10])

# row = list()
# rid = 0
# for i in range(0, 112*112*24):
#     if i%24 == 0:
#         row.append(res[i]*0.023528477177023888)
#         if len(row) == 112:
#             print(rid, ":", min(row), max(row))
#             row = list()
#             rid +=1


print("--------------------")


#res_f = 0.098893*(res-50)
from scipy.special import softmax
#res_f = softmax(res_f)

print("Got input:", img.flatten()[0:4])
print("Got output:", res[0:10])
print("Got output:", res[-10:])
#print("Got output f32:", res_f)
from scipy.special import softmax


print("max", np.max(res))
print("min", np.min(res))
for i in range(0,1000):
    v = res[i]
    if (v > 120):
        print("{}:{}".format(i, v), end=", ")
