#!/usr/bin/env python3

import tensorflow as tf
import numpy as np

def read_txt_img(fn):
    vv = list()
    with open(fn) as fp:
        for cnt, line in enumerate(fp):
            vv.append(int(line))
    return vv

def load_frozen_graph(frozen_graph_file):
    print("Loading frozen graph: {}".format(frozen_graph_file))
    with tf.io.gfile.GFile(frozen_graph_file, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")
    return graph
print(tf.__version__)
img = np.array(read_txt_img("cat224-3.txt"), dtype="float32").reshape((1,224,224,3))
img -= 127.5
img /= 255.0

print(img.shape)

model = "../models/mobilenet_v1_0.75_224_frozen.pb"

graph = load_frozen_graph(model)

in_t = graph.get_tensor_by_name("input:0")
#out_t = graph.get_tensor_by_name("MobilenetV1/MobilenetV1/Conv2d_0/Conv2D_Fold:0")
#out_t = graph.get_tensor_by_name("MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm_Fold/bias:0")
#out_t = graph.get_tensor_by_name("MobilenetV1/MobilenetV1/Conv2d_0/add_fold:0")
#out_t = graph.get_tensor_by_name("MobilenetV1/MobilenetV1/Conv2d_0/act_quant/FakeQuantWithMinMaxVars:0")
#out_t = graph.get_tensor_by_name("MobilenetV1/MobilenetV1/Conv2d_0/Relu6:0")
out_t = graph.get_tensor_by_name("MobilenetV1/MobilenetV1/Conv2d_1_depthwise/Relu6:0")



feed_dict = {}
feed_dict[in_t] = img
sess = tf.Session(graph=graph)
res = sess.run([out_t], feed_dict=feed_dict)[0]
sess.close()
print(res.shape)


#res /= 0.02352847

#for h in range(10,14):
#    print("h:", h)
#    for w in range(20,25):
#        print(res[0,h,w,0:10])

print("--------------------")
print("Got input:", img.flatten()[0:4])
print("Got output:", res.flatten()[0:10])



