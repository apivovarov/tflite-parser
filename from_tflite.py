"""Tensorflow lite to hexagon_nn converter"""
import numpy as np
import tflite.Model
from tflite.ActivationFunctionType import ActivationFunctionType
from tflite.Padding import Padding
import os
import shutil

np.set_printoptions(threshold=100000000, linewidth=160)


class TensorWrapper(object):
    """Tensor wrapper for TFLite Tensor"""
    def __init__(self, tensor_idx, tensor, buffer, c_type, qnn_params=None):
        self.tensor_idx = tensor_idx
        self.tensor = tensor
        self.buffer = buffer
        self.c_type = c_type
        self.qnn_params = qnn_params

class OperatorConverter(object):
    const_node_id = 0
    node_id = 1000

    """OperatorConverter class which handles TFLite ops to Hexagon NN ops conversion"""
    def __init__(self, model, subgraph, prog_name):

        try:
            from tflite.BuiltinOperator import BuiltinOperator
            from tflite.BuiltinOptions import BuiltinOptions
            from tflite.ActivationFunctionType import ActivationFunctionType
        except ImportError:
            raise ImportError("The tflite package must be installed")

        self.model = model
        self.subgraph = subgraph
        self.tensor_tab = {}
        self.stride_tab = {}
        self.float_tab = {}
        self.const_nodes = []
        self.nodes = []
        self.builtin_op_code = build_str_map(BuiltinOperator())
        self.activation_fn_type = build_str_map(ActivationFunctionType())
        self.builtin_options = build_str_map(BuiltinOptions())
        self.is_dequantize = False

        h_file_name = '{}.h'.format(prog_name)
        shutil.copy("header_templ.h", h_file_name)

        self.h_file = open(h_file_name, 'a')

        # Add more operators
        self.convert_map = {
            'CONV_2D': self.convert_conv2d,
            'DEPTHWISE_CONV_2D': self.convert_depthwise_conv2d,
            'RESHAPE': self.convert_reshape,
            'SOFTMAX': self.convert_softmax,
        }

    def __del__(self):
        self.close()

    def close(self):
        self.h_file.close()

    def h(self, *args):
        print(*args, file=self.h_file)

    def print_nn_nodes(self):
        self.h("void append_const_nodes(nn_id) {")
        for node in self.const_nodes:
            self.h("   ", node)
        self.h("}")
        self.h("")

        self.h("void append_nodes(nn_id) {")
        for node in self.nodes:
            self.h("   ", node)
        self.h("}")
        self.h("")

    def get_min_max(self, tensor):
        ctype, type_bytes = tensor.c_type
        qnn_params = tensor.tensor.Quantization()
        assert qnn_params is not None, "qnn_params is None"
        scale = float(qnn_params.ScaleAsNumpy())
        zero_point = int(qnn_params.ZeroPointAsNumpy())
        is_uint8 = ctype == "uint8_t"
        if is_uint8:
            min_v = -zero_point * scale
            max_v = (255 - zero_point) * scale
        else:
            max_v = 2147483647 * scale
            min_v = -max_v
        return min_v, max_v

    def nn_scalar(self, v, c_type):
        key = v
        if key in self.float_tab:
            print("Float, already exist:", self.float_tab[key])
            return self.float_tab[key]
        self.const_node_id += 1
        n_id = self.const_node_id
        ctype, type_bytes = c_type

        self.h("static {} data_for_op_{}[1] ALIGNED = {{".format(ctype, n_id))
        self.h("", v)
        self.h("};")
        self.h("")

        self.const_nodes.append("APPEND_CONST_NODE({},1,1,1,1,(const uint8_t*)data_for_op_{},{});".format(
            n_id, n_id, type_bytes
        ))

        print("Added Number, id:", self.const_node_id, ", v:", v)
        out_nodes = [(n_id, 0)]
        self.float_tab[key] = out_nodes
        return out_nodes

    def nn_stride(self, h, w):
        key = "{}x{}".format(h, w)
        if key in self.stride_tab:
            print("Stride, already exist:", self.stride_tab[key])
            return self.stride_tab[key]
        self.const_node_id += 1
        n_id = self.const_node_id

        self.const_nodes.append("APPEND_CONST_NODE({},1,{},{},1,(const uint8_t*)NULL,0);".format(
            n_id, h, w
        ))
        print("Added Stride, id:", self.const_node_id, ", h:", h, ", w:", w)
        out_nodes = [(n_id, 0)]
        self.stride_tab[key] = out_nodes
        return out_nodes

    def nn_new_const(self, tensor, val):
        tensor_id = tensor.tensor_idx
        if tensor_id in self.tensor_tab:
            print("Constant, already exist:", self.tensor_id[tensor_id])
            return self.tensor_id[tensor_id]
        self.const_node_id += 1
        n_id = self.const_node_id
        ctype, type_bytes = tensor.c_type
        is_quant = ctype != "float"
        if is_quant:
          min_v, max_v = self.get_min_max(tensor)

        ndim = len(val.shape)
        flat_v = val.flatten()
        assert ndim in [1,4], "Constant Value Shape should be 1 or 4"
        if ndim == 4:
          n, h, w, c = val.shape
        elif ndim == 1:
          n, h, w, c = 1, 1, 1, flat_v.size

        sz_bytes = type_bytes * flat_v.size

        # Data
        self.h("static {} data_for_op_{}[{}] ALIGNED = {{".format(ctype, n_id, flat_v.size))
        self.h("", np.array2string(flat_v, separator=",")[1:-1])
        self.h("};")
        self.h("")

        self.const_nodes.append("APPEND_CONST_NODE({},{},{},{},{},(const uint8_t*)data_for_op_{},{});".format(
            n_id, n,h,w,c, n_id, sz_bytes
        ))
        print("Added Constant, id:", self.const_node_id, ", type:", ctype, ", len:", len(val), ", tensor_id:", tensor_id)
        out_nodes = [(n_id, 0)]

        if is_quant:
            # Min, Max
            out_nodes += self.nn_scalar(min_v, ("float", 4))
            out_nodes += self.nn_scalar(max_v, ("float", 4))

        self.tensor_tab[tensor_id] = out_nodes
        return out_nodes

    def define_model_sizes(self, prefix, tensor):
        n,h,w,c = tensor.tensor.ShapeAsNumpy()
        _, type_bytes = tensor.c_type
        if self.is_dequantize and prefix == "OUT":
            type_bytes = 4

        self.h("//", prefix, "SIZE")
        self.h("#define {}_BATCH {}".format(prefix, n))
        self.h("#define {}_HEIGHT {}".format(prefix, h))
        self.h("#define {}_WIDTH {}".format(prefix, w))
        self.h("#define {}_DEPTH {}".format(prefix, c))
        self.h("#define {p}_LEN ({p}_BATCH * {p}_HEIGHT * {p}_WIDTH * {p}_DEPTH)".format(p=prefix))
        self.h("#define {}_ELEMENTSIZE {}".format(prefix, type_bytes))
        self.h("#define {p}_SIZE ({p}_LEN * {p}_ELEMENTSIZE)".format(p=prefix))
        self.h("")

    def nn_add_input(self, tensor):
        self.node_id += 1
        n_id = self.node_id
        tensor_id = tensor.tensor_idx
        n,h,w,c = tensor.tensor.ShapeAsNumpy()
        ctype, type_bytes = tensor.c_type
        is_quant = ctype != "float"
        if is_quant:
            min_v, max_v = self.get_min_max(tensor)

        self.h("// INPUT")
        # prep_input_arr - empty
        self.h("static hexagon_nn_input inputs_for_{}[] = {{".format(n_id))
        self.h("};")
        # prep_output_arr - shape + dtype
        self.h("static hexagon_nn_output outputs_for_{}[] = {{".format(n_id))
        self.h("    OUTPUT_4D({},{},{},{},{}),".format(
            n,h,w,c,type_bytes
        ))
        self.h("};")
        self.h("")

        self.nodes.append("APPEND_NODE(\"input\",{},OP_INPUT,NN_PAD_NA,inputs_for_{},0,outputs_for_{},1);".format(
            n_id, n_id, n_id
        ))

        print("Added input node, id:", self.node_id, "shape: , dtype: ", ", tensor_id:", tensor_id)
        out_nodes = [(n_id, 0)]

        if is_quant:
            # Min, Max
            out_nodes += self.nn_scalar(min_v, ("float", 4))
            out_nodes += self.nn_scalar(max_v, ("float", 4))

        self.tensor_tab[tensor_id] = out_nodes
        return out_nodes

    def nn_add_output(self, data_nodes):
        self.node_id += 1
        n_id = self.node_id
        #assert len(data_nodes) == 1, "OUTPUT length should be 1"
        # prep_input_arr - [node]
        self.h("// OUTPUT")
        self.h("static hexagon_nn_input inputs_for_{}[] = {{".format(n_id))
        self.h("    {{ .src_id = {}, .output_idx = {}, }},".format(data_nodes[0][0], data_nodes[0][1]))
        self.h("};")
        # prep_output_arr - empty
        self.h("static hexagon_nn_output outputs_for_{}[] = {{".format(n_id))
        self.h("};")
        self.h("")

        self.nodes.append(
            "APPEND_NODE(\"output\",{nid},OP_OUTPUT,NN_PAD_NA,inputs_for_{nid},1,outputs_for_{nid},0);"
            .format(nid=n_id)
        )

        print("Added output node, id:", self.node_id, ", node: ", data_nodes[0])

    def nn_requantize32_8(self, data_nodes, data_shape, min_v, max_v):
        assert len(data_nodes) == 3, "data_nodes size should be 3"
        min_nodes = self.nn_scalar(min_v, ("float", 4))
        max_nodes = self.nn_scalar(max_v, ("float", 4))
        self.node_id += 1
        n_id = self.node_id

        # prep_input_arr
        self.h("// Requantize")
        self.h("static hexagon_nn_input inputs_for_{}[] = {{".format(n_id))
        for node in data_nodes:
            self.h("    {{ .src_id = {}, .output_idx = {}, }},".format(node[0], node[1]))
        self.h("    {{ .src_id = {}, .output_idx = {}, }},".format(min_nodes[0][0], min_nodes[0][1]))
        self.h("    {{ .src_id = {}, .output_idx = {}, }},".format(max_nodes[0][0], max_nodes[0][1]))
        self.h("};")
        # prep_output_arr
        n, h, w, c = data_shape
        self.h("static hexagon_nn_output outputs_for_{}[] = {{".format(n_id))
        self.h("    OUTPUT_4D({},{},{},{},1),".format(n, h, w, c))
        self.h("    OUTPUT_4D(1,1,1,1,4),")
        self.h("    OUTPUT_4D(1,1,1,1,4),")
        self.h("};")
        self.h("")

        self.nodes.append(
            "APPEND_NODE(\"requantize\",{n_id},OP_Requantize_32to8,NN_PAD_NA,inputs_for_{n_id},5,outputs_for_{n_id},3);".format(
                n_id=n_id
            )
        )

        print("Requantize. id:", self.node_id, ", data_nodes: ", data_nodes, ", min_nodes:", min_nodes,
              "max_nodes: ", max_nodes)
        out_nodes = list(map(lambda i: (n_id, i), range(0, 3)))
        return out_nodes

    def nn_dequantize(self, data_nodes, data_shape):
        assert len(data_nodes) == 3, "data_nodes size should be 3"
        self.node_id += 1
        n_id = self.node_id

        # prep_input_arr
        self.h("// Requantize")
        self.h("static hexagon_nn_input inputs_for_{}[] = {{".format(n_id))
        for node in data_nodes:
            self.h("    {{ .src_id = {}, .output_idx = {}, }},".format(node[0], node[1]))
        self.h("};")
        # prep_output_arr
        n, h, w, c = data_shape
        self.h("static hexagon_nn_output outputs_for_{}[] = {{".format(n_id))
        self.h("    OUTPUT_4D({},{},{},{},4),".format(n, h, w, c))
        self.h("};")
        self.h("")

        self.nodes.append(
            "APPEND_NODE(\"dequantize\",{n_id},OP_Dequantize,NN_PAD_NA,inputs_for_{n_id},3,outputs_for_{n_id},1);".format(
                n_id=n_id
            )
        )

        print("Dequantize. id:", self.node_id, ", data_nodes: ", data_nodes)
        out_nodes = list(map(lambda i: (n_id, i), range(0, 1)))
        return out_nodes

    def nn_bias_add(self, data_nodes, bias_nodes, data_shape):
        self.node_id += 1
        n_id = self.node_id
        is_quant = len(data_nodes) == 3 and len(bias_nodes) == 3
        # prep_input_arr
        self.h("// Bias_add")
        self.h("static hexagon_nn_input inputs_for_{}[] = {{".format(n_id))
        self.h("    {{ .src_id = {}, .output_idx = {}, }},".format(data_nodes[0][0], data_nodes[0][1]))
        self.h("    {{ .src_id = {}, .output_idx = {}, }},".format(bias_nodes[0][0], bias_nodes[0][1]))
        if is_quant:
            self.h("    {{ .src_id = {}, .output_idx = {}, }},".format(data_nodes[1][0], data_nodes[1][1]))
            self.h("    {{ .src_id = {}, .output_idx = {}, }},".format(data_nodes[2][0], data_nodes[2][1]))
            self.h("    {{ .src_id = {}, .output_idx = {}, }},".format(bias_nodes[1][0], bias_nodes[1][1]))
            self.h("    {{ .src_id = {}, .output_idx = {}, }},".format(bias_nodes[2][0], bias_nodes[2][1]))
        self.h("};")
        # prep_output_arr
        n,h,w,c = data_shape
        self.h("static hexagon_nn_output outputs_for_{}[] = {{".format(n_id))
        self.h("    OUTPUT_4D({},{},{},{},4),".format(n,h,w,c))
        if is_quant:
            self.h("    OUTPUT_4D(1,1,1,1,4),")
            self.h("    OUTPUT_4D(1,1,1,1,4),")
        self.h("};")
        self.h("")

        in_len, out_len = 2, 1
        f_name = "OP_BiasAdd_f"
        if is_quant:
            in_len, out_len = 6, 3
            f_name = "OP_QuantizedBiasAdd_32p32to32"

        self.nodes.append(
            "APPEND_NODE(\"bias\",{nid},{f_name},NN_PAD_NA,inputs_for_{nid},{in_len},outputs_for_{nid},{out_len});"
            .format(nid=n_id, f_name=f_name, in_len=in_len, out_len=out_len)
        )

        print("bias_add. id: ", self.node_id, ", data_id: ", data_nodes, ", bias_nodes:", bias_nodes)
        out_nodes = list(map(lambda i: (n_id, i), range(0, out_len)))
        return out_nodes

    def nn_conv2d(self, data_nodes, weight_nodes, stride_nodes, conv_options, output_shape, is_dw):
        self.node_id += 1
        n_id = self.node_id
        is_quant = len(data_nodes) == 3

        padding = "NN_PAD_VALID"
        if conv_options.Padding() == Padding.SAME:
            padding = "NN_PAD_SAME"

        op_name = "OP_Conv2d_f"
        n_name = "conv2d"
        in_len, out_len = 3, 1
        if is_dw:
            op_name = "OP_DepthwiseConv2d_f_ref"
            n_name = "dw_conv2d"
        if is_quant:
            op_name = "OP_QuantizedConv2d_8x8to32"
            n_name = "qconv2d"
            in_len, out_len = 7, 3
            if is_dw:
                op_name = "OP_QuantizedDepthwiseConv2d_8x8to32"
                n_name = "dw_qconv2d"

        # prep_input_arr
        self.h("//", n_name)
        self.h("static hexagon_nn_input inputs_for_{}[] = {{".format(n_id))
        self.h("    {{ .src_id = {}, .output_idx = {}, }},".format(data_nodes[0][0], data_nodes[0][1]))
        self.h("    {{ .src_id = {}, .output_idx = {}, }},".format(weight_nodes[0][0], weight_nodes[0][1]))
        if is_quant:
            self.h("    {{ .src_id = {}, .output_idx = {}, }},".format(data_nodes[1][0], data_nodes[1][1]))
            self.h("    {{ .src_id = {}, .output_idx = {}, }},".format(data_nodes[2][0], data_nodes[2][1]))
            self.h("    {{ .src_id = {}, .output_idx = {}, }},".format(weight_nodes[1][0], weight_nodes[1][1]))
            self.h("    {{ .src_id = {}, .output_idx = {}, }},".format(weight_nodes[2][0], weight_nodes[2][1]))
        self.h("    {{ .src_id = {}, .output_idx = {}, }},".format(stride_nodes[0][0], stride_nodes[0][1]))
        self.h("};")
        # prep_output_arr
        n, h, w, c = output_shape
        self.h("static hexagon_nn_output outputs_for_{}[] = {{".format(n_id))
        self.h("    OUTPUT_4D({},{},{},{},4),".format(n, h, w, c))
        if is_quant:
            self.h("    OUTPUT_4D(1,1,1,1,4),")
            self.h("    OUTPUT_4D(1,1,1,1,4),")
        self.h("};")
        self.h("")

        self.nodes.append(
            "APPEND_NODE(\"{n_name}\",{n_id},{op_name},{padding},inputs_for_{n_id},{in_len},outputs_for_{n_id},{out_len});".format(
                n_name=n_name, n_id=n_id, op_name=op_name, padding=padding, in_len=in_len, out_len=out_len
            )
        )

        print("nn_conv2d. id:", self.node_id ,", data_nodes: ", data_nodes, ", weight_nodes:", weight_nodes, "stride_nodes: ", stride_nodes)
        out_nodes = list(map(lambda i: (n_id, i), range(0, out_len)))
        return out_nodes

    def nn_fused_activation(self, data_nodes, fused_activation_fn, data_shape, c_type):
        self.node_id += 1
        n_id = self.node_id
        is_quant = len(data_nodes) == 3
        if fused_activation_fn == ActivationFunctionType.RELU6:
            threshold_node = self.nn_scalar(6.0, ("float", 4))[0]
            print("Relu6. id:", self.node_id, ", data_id: ", data_nodes, ", threshold_node:", threshold_node)
        elif fused_activation_fn == ActivationFunctionType.RELU:
            print("Relu. id:", self.node_id, ", data_id: ", data_nodes)
        else:
            raise ValueError("Activation function RELU_N1_TO_1, TAHN, SIGN_BIT are not supported")

        in_len, out_len = 0, 0
        # prep_input_arr
        if fused_activation_fn == ActivationFunctionType.RELU6:
            self.h("// ReluX")
        else:
            self.h("// Relu")
        self.h("static hexagon_nn_input inputs_for_{}[] = {{".format(n_id))
        for data_node in data_nodes:
            self.h("    {{ .src_id = {}, .output_idx = {}, }},".format(data_node[0], data_node[1]))
            in_len += 1
        if threshold_node is not None:
            self.h("    {{ .src_id = {}, .output_idx = {}, }},".format(threshold_node[0], threshold_node[1]))
            in_len += 1
        self.h("};")

        # prep_output_arr
        n, h, w, c = data_shape
        _, type_bytes = c_type
        self.h("static hexagon_nn_output outputs_for_{}[] = {{".format(n_id))
        self.h("    OUTPUT_4D({},{},{},{},{}),".format(n, h, w, c, type_bytes))
        out_len += 1
        if len(data_nodes) == 3:
            self.h("    OUTPUT_4D(1,1,1,1,4),")
            self.h("    OUTPUT_4D(1,1,1,1,4),")
            out_len += 2
        self.h("};")
        self.h("")

        f_name = "Relu"
        if fused_activation_fn == ActivationFunctionType.RELU6:
            f_name = "ReluX"
        if is_quant:
            f_name = "Quantized" + f_name + "_8"
        else:
            f_name = f_name + "_f"
        f_name = "OP_" + f_name

        self.nodes.append(
            "APPEND_NODE(\"relu\",{nid},{f_name},NN_PAD_NA,inputs_for_{nid},{in_len},outputs_for_{nid},{out_len});"
            .format(nid=n_id, f_name=f_name, in_len=in_len, out_len=out_len)
        )

        out_nodes = list(map(lambda i: (n_id, i), range(0, out_len)))
        return out_nodes

    def nn_reshape(self, data_nodes, data_shape, target_shape):
        n, h, w, c = data_shape
        if n == 1 and h == 1 and w == 1 and c == target_shape[-1]:
            print("Skip reshape")
            return data_nodes
        raise ValueError("Reshape is not implemented yet")

    def nn_softmax(self, data_nodes, data_shape):
        in_len = out_len = len(data_nodes)
        assert in_len in [1, 3], "Softmax data_nodes size should be 1 or 3"
        self.node_id += 1
        n_id = self.node_id
        f_name = "OP_Softmax_f"
        type_bytes = 4
        is_quant = in_len == 3
        if is_quant:
            f_name = "OP_QuantizedSoftmax_8"
            type_bytes = 1

        # prep_input_arr
        self.h("// Softmax")
        self.h("static hexagon_nn_input inputs_for_{}[] = {{".format(n_id))
        for data_node in data_nodes:
            self.h("    {{ .src_id = {}, .output_idx = {}, }},".format(data_node[0], data_node[1]))
        self.h("};")

        # prep_output_arr
        n, h, w, c = data_shape
        self.h("static hexagon_nn_output outputs_for_{}[] = {{".format(n_id))
        self.h("    OUTPUT_4D({},{},{},{},{}),".format(n, h, w, c, type_bytes))
        if is_quant:
            self.h("    OUTPUT_4D(1,1,1,1,4),")
            self.h("    OUTPUT_4D(1,1,1,1,4),")
        self.h("};")
        self.h("")

        self.nodes.append(
            "APPEND_NODE(\"softmax\",{nid},{f_name},NN_PAD_NA,inputs_for_{nid},{in_len},outputs_for_{nid},{out_len});"
            .format(nid=n_id, f_name=f_name, in_len=in_len, out_len=out_len)
        )
        out_nodes = list(map(lambda i: (n_id, i), range(0, out_len)))
        return out_nodes


    def check_unsupported_ops(self):
        """Check unsupported TFLite ops in our converter."""
        unsupported_ops_set = set()

        for op_idx in range(self.subgraph.OperatorsLength()):
            op = self.subgraph.Operators(op_idx)
            op_code_str = self.get_op_code_str(op)
            if op_code_str not in self.convert_map:
                unsupported_ops_set.add(op_code_str)

        if unsupported_ops_set:
            msg = 'The following operators are not supported in frontend ' \
                  'TFLite: {}'
            ops = str(list(unsupported_ops_set)).strip('[,]')
            raise ValueError(msg.format(ops))


    def convert_op_to_hexagon_nn(self):
        """Convert TFLite ops to hexagon_nn"""
        for op_idx in range(self.subgraph.OperatorsLength()):
            op = self.subgraph.Operators(op_idx)
            op_code_str = self.get_op_code_str(op)
            output_tensors = self.get_output_tensors(op)

            ret = self.convert_map[op_code_str](op)

        if self.is_dequantize and len(ret) == 3:
            out_shape = output_tensors[0].tensor.ShapeAsNumpy()
            ret = self.nn_dequantize(ret, out_shape)
        assert len(output_tensors) == 1, "Last Operator should have one output tensor"
        return ret


    def get_op_code_str(self, op):
        """Get TFLite ops string representation"""
        try:
            from tflite.BuiltinOperator import BuiltinOperator
        except ImportError:
            raise ImportError("The tflite package must be installed")

        op_code_list_idx = op.OpcodeIndex()
        op_code_id = self.model.OperatorCodes(op_code_list_idx).BuiltinCode()
        op_code_str = self.builtin_op_code[op_code_id]
        if op_code_id == BuiltinOperator.CUSTOM:
            # Custom operator
            raise NotImplementedError("Custom operators are currently not supported")
        return op_code_str

    def get_input_tensors(self, op):
        operator_inputs = op.InputsAsNumpy()
        return self.get_tensors(operator_inputs)

    def get_output_tensors(self, op):
        operator_outputs = op.OutputsAsNumpy()
        return self.get_tensors(operator_outputs)

    def get_tensors(self, tensors_idx_list):
        """Get tensor wrapper list from given TFLite tensor index list"""
        return_list = list()
        for tensor_idx in tensors_idx_list:
            if tensor_idx < 0:
                return_list.append(TensorWrapper(tensor_idx, 0, 0))
                continue

            tensor = self.subgraph.Tensors(tensor_idx)
            buffer_idx = tensor.Buffer()
            buffer = self.model.Buffers(buffer_idx)
            c_type = self.get_c_type(tensor.Type())

            # Check if the tensors are quantized. Parse if yes.
            qnn_params = None
            tflite_qnn_params = tensor.Quantization()
            if tflite_qnn_params is not None:
                qnn_params = dict()
                scale = float(tflite_qnn_params.ScaleAsNumpy())
                zero_point = int(tflite_qnn_params.ZeroPointAsNumpy())
                min_v = float(tflite_qnn_params.MinAsNumpy())
                max_v = float(tflite_qnn_params.MaxAsNumpy())
                # Check that the scale and zero points are valid.
                if scale != 0 or zero_point != 0:
                    qnn_params['scale'] = scale
                    qnn_params['zero_point'] = zero_point
                if min_v != 0 or max_v != 0:
                    qnn_params['min'] = min_v
                    qnn_params['max'] = max_v
            return_list.append(TensorWrapper(tensor_idx, tensor, buffer, c_type, qnn_params))
        return return_list

    def get_tensor_value(self, tensor_wrapper):
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
        raise NotImplementedError("Tensor type {} is currently not supported"
                                  .format(str(tensor_wrapper.tensor.Type())))

    def get_c_type(self, tensor_type):
        """Get tensor type string representation when given TFLite tensor type"""
        try:
            from tflite.TensorType import TensorType
        except ImportError:
            raise ImportError("The tflite package must be installed")

        if tensor_type == TensorType.UINT8:
            return ("uint8_t", 1)
        if tensor_type == TensorType.FLOAT32:
            return ("float", 4)
        if tensor_type == TensorType.INT32:
            return ("int32_t", 4)
        raise NotImplementedError("Tensor type {} is currently not supported"
                                  .format(str(tensor_type)))

    def convert_conv2d(self, op):
        """Convert TFLite conv2d"""
        return self.convert_conv(op, "conv2d")

    def convert_depthwise_conv2d(self, op):
        """Convert TFLite depthwise conv2d"""
        return self.convert_conv(op, "depthwise")

    def convert_conv(self, op, conv_type):
        """convolution implementation."""
        try:
            from tflite.BuiltinOptions import BuiltinOptions
            from tflite.ActivationFunctionType import ActivationFunctionType
            from tflite.TensorType import TensorType
            from tflite.Operator import Operator
            from tflite.Conv2DOptions import Conv2DOptions
            from tflite.DepthwiseConv2DOptions import DepthwiseConv2DOptions
            from tflite.Padding import Padding
        except ImportError:
            raise ImportError("The tflite package must be installed")

        assert isinstance(op, Operator)
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) >= 2, "input tensors length should be >= 2"

        weight_tensor = input_tensors[1]

        # weight tensor type should be UINT8 (quantization) or FLOAT32
        weight_tensor_type = weight_tensor.tensor.Type()
        assert weight_tensor_type in (TensorType.UINT8, TensorType.FLOAT32)

        # in_expr = self.get_expr(input_tensor_idx)
        weight_value = self.get_tensor_value(weight_tensor)

        # TFLite kernel layout:
        # convolution:
        # OC KH KW IC, we require KH KW IC OC (HWIO)
        # depthwise convolution:
        # 1 KH KW C(input_c * depth_multiplier), we require
        # KH KW IC M (depth_multiplier) (HWOI)
        weight_value = weight_value.transpose((1, 2, 3, 0))
        weight_nodes = self.nn_new_const(weight_tensor, weight_value)


        #return
        # End of Constant nodes generation pass

        # Non-Constant nodes generation Pass
        input_tensor = input_tensors[0]
        input_tensor_idx = input_tensor.tensor_idx

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "output tensors length should be 1"
        output_tensor = output_tensors[0]
        output_tensor_idx = output_tensor.tensor_idx
        output_tensor_shape = output_tensor.tensor.ShapeAsNumpy()

        is_dw_conv = False
        if conv_type == 'conv2d':
            assert op.BuiltinOptionsType() == BuiltinOptions.Conv2DOptions
            op_options = op.BuiltinOptions()
            conv_options = Conv2DOptions()
            conv_options.Init(op_options.Bytes, op_options.Pos)
        elif conv_type == 'depthwise':
            is_dw_conv = True
            assert op.BuiltinOptionsType() == BuiltinOptions.DepthwiseConv2DOptions
            op_options = op.BuiltinOptions()
            conv_options = DepthwiseConv2DOptions()
            conv_options.Init(op_options.Bytes, op_options.Pos)
            depth_multiplier = conv_options.DepthMultiplier()
        else:
            raise ValueError(
                'Operator {} is not supported for frontend TFLite.'.format(conv_type))

        stride_h = conv_options.StrideH()
        stride_w = conv_options.StrideW()
        stride_nodes = self.nn_stride(stride_h, stride_w)

        dilation_h = conv_options.DilationHFactor()
        dilation_w = conv_options.DilationWFactor()

        _, input_h, input_w, input_c = input_tensor.tensor.ShapeAsNumpy()

        if is_dw_conv:
            # TFLite depthwise convolution kernel layout is:
            # 1 KH KW C(input_c * depth_multiplier)
            _, kernel_h, kernel_w, in_channels = weight_tensor.tensor.ShapeAsNumpy()
            assert in_channels == input_c * depth_multiplier
        else:
            output_channels, kernel_h, kernel_w, _ = weight_tensor.tensor.ShapeAsNumpy()

        if input_tensor_idx not in self.tensor_tab:
            raise ValueError("Can not find tensor_id {} in tensor_tab".format(input_tensor_idx))
        data_nodes = self.tensor_tab[input_tensor_idx]
        conv_out_nodes = self.nn_conv2d(
            data_nodes, weight_nodes, stride_nodes, conv_options, output_tensor_shape, is_dw_conv)
        op_out_nodes = conv_out_nodes
        # if we have bias
        if len(input_tensors) == 3:
            bias_tensor = input_tensors[2]
            bias_tensor_type = bias_tensor.tensor.Type()
            assert bias_tensor_type in (TensorType.INT32, TensorType.FLOAT32)
            bias_value = self.get_tensor_value(bias_tensor)
            bias_nodes = self.nn_new_const(bias_tensor, bias_value)
            bias_out_nodes = self.nn_bias_add(conv_out_nodes, bias_nodes, output_tensor_shape)
            op_out_nodes = bias_out_nodes
        # If we have fused activations
        fused_activation_fn = conv_options.FusedActivationFunction()
        if fused_activation_fn == ActivationFunctionType.NONE:
            # 32 to 8 bit case
            if output_tensor.c_type[1] == 1 and output_tensor.qnn_params is not None:
                out_min_v = output_tensor.qnn_params['min']
                out_max_v = output_tensor.qnn_params['max']
                op_out_nodes = self.nn_requantize32_8(op_out_nodes, output_tensor_shape, out_min_v, out_max_v)
        else:
            op_out_nodes = self.nn_fused_activation(
                op_out_nodes, fused_activation_fn, output_tensor_shape, output_tensor.c_type)


        self.tensor_tab[output_tensor_idx] = op_out_nodes
        return op_out_nodes

    def convert_reshape(self, op):
        """Convert TFLite reshape"""
        try:
            from tflite.BuiltinOptions import BuiltinOptions
            from tflite.Operator import Operator
            from tflite.ReshapeOptions import ReshapeOptions
        except ImportError:
            raise ImportError("The tflite package must be installed")

        assert isinstance(op, Operator)
        input_tensors = self.get_input_tensors(op)
        assert input_tensors, "input tensors should not be empty"
        input_tensor = input_tensors[0]
        input_tensor_idx = input_tensor.tensor_idx
        input_tensor_shape = input_tensor.tensor.ShapeAsNumpy()

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "output tensors length should be 1"
        output_tensor = output_tensors[0]
        output_tensor_idx = output_tensor.tensor_idx

        assert op.BuiltinOptionsType() == BuiltinOptions.ReshapeOptions
        op_options = op.BuiltinOptions()
        reshape_options = ReshapeOptions()
        reshape_options.Init(op_options.Bytes, op_options.Pos)
        target_shape = reshape_options.NewShapeAsNumpy()

        data_nodes = self.tensor_tab[input_tensor_idx]

        out_nodes = self.nn_reshape(data_nodes, input_tensor_shape, target_shape)

        self.tensor_tab[output_tensor_idx] = out_nodes
        return out_nodes

    def convert_softmax(self, op):
        """Convert TFLite softmax"""
        try:
            from tflite.Operator import Operator
        except ImportError:
            raise ImportError("The tflite package must be installed")

        assert isinstance(op, Operator)
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 1, "input tensors length should be 1"

        input_tensor = input_tensors[0]
        input_tensor_idx = input_tensor.tensor_idx

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "output tensors length should be 1"
        output_tensor = output_tensors[0]
        output_tensor_idx = output_tensor.tensor_idx
        output_tensor_shape = output_tensor.tensor.ShapeAsNumpy()

        data_nodes = self.tensor_tab[input_tensor_idx]

        out_nodes = self.nn_softmax(data_nodes, output_tensor_shape)

        self.tensor_tab[output_tensor_idx] = out_nodes
        return out_nodes


def build_str_map(obj):
    """Build string map of TFLite enum int value

    Parameters
    ----------
    obj:
        TFLite class which contains enum int value, such as BuiltInOptions

    Returns
    -------
        String representation map of TFLite class enum int value
    """
    ret = {}
    for field_name in dir(obj):
        if not field_name.startswith('_'):
            field_value = getattr(obj, field_name)
            if isinstance(field_value, int):
                ret[field_value] = field_name
    return ret



def get_tensor_name(subgraph, tensor_idx):
    """Get the tensor name.

    Parameters
    ----------
    subgraph:
        tflite.Subgraph.Subgraph

    tensor:
        tensor index in subgraph

    Returns
    -------
        tensor name in UTF-8 encoding
    """
    return subgraph.Tensors(tensor_idx).Name().decode("utf-8")


def from_tflite(model, prog_name): #, shape_dict, dtype_dict):
    """Convert from tflite model to Hexagon NN C program.

    Parameters
    ----------
    model:
        tflite.Model.Model

    shape_dict : dict of str to int list/tuple
        Input shapes of the model.

    dtype_dict : dict of str to str
        Input types of the model.

    """
    try:
        import tflite.Model
        import tflite.SubGraph
        import tflite.BuiltinOperator
    except ImportError:
        raise ImportError("The tflite package must be installed")
    assert isinstance(model, tflite.Model.Model)

    # keep the same as tflite
    assert model.SubgraphsLength() == 1, "only support one subgraph (main subgraph)"
    subgraph = model.Subgraphs(0)

    # model inputs / outputs
    model_inputs = subgraph.InputsAsNumpy()
    model_outputs = subgraph.OutputsAsNumpy()
    assert model_inputs.size == 1, "Model should have only one input"
    assert model_outputs.size == 1, "Model should have only one output"

    #for model_input in model_inputs:
    #     nn_add_input(model_input)
    #     model_input_name = get_tensor_name(subgraph, model_input)
    #     shape = shape_dict[model_input_name] if model_input_name in shape_dict else None
    #     dtype = dtype_dict[model_input_name] if model_input_name in dtype_dict else "float32"
    #     #exp_tab.set_expr(model_input_name, _expr.var(model_input_name, shape=shape, dtype=dtype))

    # op code in model
    op_converter = OperatorConverter(model, subgraph, prog_name)
    op_converter.is_dequantize = True
    op_converter.check_unsupported_ops()

    in_tensor = op_converter.get_tensors(model_inputs)[0]
    out_tensor = op_converter.get_tensors(model_outputs)[0]

    op_converter.define_model_sizes("IN", in_tensor)
    op_converter.define_model_sizes("OUT", out_tensor)

    op_converter.nn_add_input(in_tensor)

    output_nodes = op_converter.convert_op_to_hexagon_nn()

    op_converter.nn_add_output(output_nodes)

    op_converter.print_nn_nodes()

    print("tensor_tab:")
    print(op_converter.tensor_tab)

    op_converter.close()


def main():
    tflite_model_file = os.path.join("../models/", "mobilenet_v1_0.75_224_conv1.tflite")
    tflite_model_buf = open(tflite_model_file, "rb").read()
    model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)
    prog_name = "qmn2"
    from_tflite(model, prog_name)


if __name__== "__main__":
    main()
