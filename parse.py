#!/usr/bin/env python3

import tflite.Model
import tflite.BuiltinOperator
import numpy as np
import os
tflite_model_file = os.path.join("../models/", "mobilenet_v1_0.75_224_conv1.tflite")
tflite_model_buf = open(tflite_model_file, "rb").read()
model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)
np.set_printoptions(threshold=100000, precision=12, linewidth=160, floatmode='fixed')

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
    #fvv = list()
    #for v in vv:
    #    fv = (v - t.qnn_params['zero_point']) * t.qnn_params['scale']
    #    fvv.append(fv)
    #print(t.qnn_params)
    #print("f32 min, max: ", min(fvv), max(fvv))
    print("float[",vv.size,"] v =", np.array2string(vv, separator=","))
    print("--------------------")

def print_weight(id):
    print("--WEIGHTS------------------")
    t = get_tensors(model, subgraph, [id])[0]
    vv = get_tensor_value(t)
    print(t.qnn_params)
    ### OHWI to HWIO
    vv = vv.transpose(1,2,3,0).flatten()
    ### Use OHWI format
    #vv = vv.flatten()
    print("float[",vv.size,"] v =", np.array2string(vv, separator=","))
    print("--------------------")

subgraph = model.Subgraphs(0)
model_inputs = subgraph.InputsAsNumpy()
tid = 0
tn = subgraph.Tensors(tid).Name().decode("utf-8")
print(tid, tn)
tt = get_tensors(model, subgraph, [tid])
print(tt[0].qnn_params)
tv = get_tensor_value(tt[0])
vv=tv.flatten()
#print(vv)
#print("min, max: ", min(vv), max(vv))
#fvv = list()
#for v in vv:
#    fv = (v - tt[0].qnn_params['zero_point']) * tt[0].qnn_params['scale']
#    fvv.append(fv)
#print(fvv)
#print("min, max: ", min(fvv), max(fvv))


# print_data(id=3)
# print_weight(id=0)
# print_bias(id=1)

# print_data(id=6)
# print_weight(id=0)
# print_bias(id=2)
#
# print_data(id=3)
# print_weight(id=1)
# print_bias(id=5)
#
# print_data(id=4)


import tensorflow.lite as lite

def read_txt_img(fn):
    vv = list()
    with open(fn) as fp:
        for cnt, line in enumerate(fp):
            vv.append((int(line)-127.5)/255)
            #vv.append((int(line) - 128) / 256)
    return vv

img = np.array(read_txt_img("cat224-3.txt"), dtype="float32").reshape((1,224,224,3))
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

row = list()
rid = 0
for i in range(0, 112*112*24):
    if i%24 == 0:
        row.append(res[i])
        if len(row) == 112:
            print(rid, ":", min(row), max(row))
            row = list()
            rid +=1

print("--------------------")


print("Got input:", img.flatten()[0:4])
print("Got output:", res[0:10])

