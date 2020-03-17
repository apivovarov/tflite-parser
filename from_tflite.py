
"""Tensorflow lite frontend."""
import math
import numpy as np
import tflite.Model


class TensorWrapper(object):
    """Tensor wrapper for TFLite Tensor"""
    def __init__(self, tensor_idx, tensor, buffer, qnn_params=None):
        self.tensor_idx = tensor_idx
        self.tensor = tensor
        self.buffer = buffer
        self.qnn_params = qnn_params

class OperatorConverter(object):
    const_node_id = 0
    node_id = 1000
    is_const = True

    """OperatorConverter class which handles TFLite ops to Hexagon NN ops conversion"""
    def __init__(self, model, subgraph, tensor_tab):

        try:
            from tflite.BuiltinOperator import BuiltinOperator
            from tflite.BuiltinOptions import BuiltinOptions
            from tflite.ActivationFunctionType import ActivationFunctionType
        except ImportError:
            raise ImportError("The tflite package must be installed")

        self.model = model
        self.subgraph = subgraph
        self.tensor_tab = tensor_tab
        self.stride_tab = {}
        self.float_tab = {}
        self.builtin_op_code = build_str_map(BuiltinOperator())
        self.activation_fn_type = build_str_map(ActivationFunctionType())
        self.builtin_options = build_str_map(BuiltinOptions())

        # Add more operators
        self.convert_map = {
            'CONV_2D': self.convert_conv2d,
            'DEPTHWISE_CONV_2D': self.convert_depthwise_conv2d,
        }

    def nn_float(self, v):
        key = v
        if key in self.float_tab:
            print("Float, already exist:", self.float_tab[key])
            return self.float_tab[key]
        self.const_node_id += 1
        print("Added Float, id:", self.const_node_id, ", v:", v)
        out_nodes = [(self.const_node_id, 0)]
        self.float_tab[key] = out_nodes
        return out_nodes

    def nn_stride(self, h, w):
        key = "{}x{}".format(h, w)
        if key in self.stride_tab:
            print("Stride, already exist:", self.stride_tab[key])
            return self.stride_tab[key]
        self.const_node_id += 1
        print("Added Stride, id:", self.const_node_id, ", h:", h, ", w:", w)
        out_nodes = [(self.const_node_id, 0)]
        self.stride_tab[key] = out_nodes
        return out_nodes


    def nn_new_const(self, tensor_id, val, dtype):
        self.const_node_id += 1
        print("Added Constant, id:", self.const_node_id, ", type:", dtype, ", len:", len(val), ", tensor_id:", tensor_id)
        out_nodes = [(self.const_node_id, 0)]
        self.tensor_tab[tensor_id] = out_nodes
        return out_nodes

    def nn_add_input(self, tensor_id):
        self.node_id += 1
        # prep_input_arr - empty
        # prep_output_arr - shape + dtype
        print("Added input node, id:", self.node_id, "shape: , dtype: ", ", tensor_id:", tensor_id)
        out_nodes = [(self.node_id, 0)]
        self.tensor_tab[tensor_id] = out_nodes
        return out_nodes

    def nn_add_output(self, tensor_id):
        if tensor_id not in self.tensor_tab:
            raise ValueError("Can not find tensor_id {} in tensor_tab".format(tensor_id))
        node = self.tensor_tab[tensor_id]
        self.node_id += 1
        # prep_input_arr - [node]
        # prep_output_arr - empty
        print("Added output node, id:", self.node_id, ", node: ", node)

    def nn_bias_add(self, data_id, bias_nodes):
        self.node_id += 1
        print("bias_add. id: ", self.node_id, ", data_id: ", data_id, ", bias_nodes:", bias_nodes)
        out_nodes = [(self.node_id, 0)]
        return out_nodes

    def nn_conv2d(self, data_nodes, weight_nodes, stride_nodes, params):
        self.node_id += 1
        print("nn_conv2d. id:", self.node_id ,", data_nodes: ", data_nodes, ", weight_nodes:", weight_nodes, "stride_nodes: ", stride_nodes, ", params:", params)
        out_nodes = [(self.node_id, 0)]
        return out_nodes

    def nn_fused_activation(self, data_id, fused_activation_fn):
        self.node_id += 1
        from tflite.ActivationFunctionType import ActivationFunctionType
        if fused_activation_fn == ActivationFunctionType.RELU6:
            threshold_nodes = self.nn_float(6.0)
            print("Relu6. id:", self.node_id, ", data_id: ", data_id, ", threshold_nodes:", threshold_nodes)
        elif fused_activation_fn == ActivationFunctionType.RELU:
            print("Relu. id:", self.node_id, ", data_id: ", data_id)
        else:
            raise ValueError("Activation function RELU_N1_TO_1, TAHN, SIGN_BIT are not supported")
        out_nodes = [(self.node_id, 0)]
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

    def convert_constants_to_hexagon_nn(self):
        for tensor_id in range(self.subgraph.TensorsLength()):
            tensor = self.get_tensors([tensor_id])[0]
            if tensor.buffer.DataIsNone():
                continue
            tensor_type_str = self.get_tensor_type_str(tensor.tensor.Type())
            self.nn_new_const(tensor_id, self.get_tensor_value(tensor),
                              dtype=tensor_type_str)

    def convert_op_to_hexagon_nn(self):
        """Convert TFLite ops to hexagon_nn"""
        for op_idx in range(self.subgraph.OperatorsLength()):
            op = self.subgraph.Operators(op_idx)
            op_code_str = self.get_op_code_str(op)
            output_tensors = self.get_output_tensors(op)

            ret = self.convert_map[op_code_str](op)

            # if len(output_tensors) == 1:
            #     tensor_idx = output_tensors[0].tensor_idx
            #     self.set_expr(get_tensor_name(self.subgraph, tensor_idx), ret)
            # else:
            #     for idx, output_tensor in enumerate(output_tensors):
            #         self.set_expr(get_tensor_name(self.subgraph, output_tensor.tensor_idx),
            #                               ret[idx])

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

            # Check if the tensors are quantized. Parse if yes.
            qnn_params = None
            tflite_qnn_params = tensor.Quantization()
            if tflite_qnn_params is not None:
                scale = float(tflite_qnn_params.ScaleAsNumpy())
                zero_point = int(tflite_qnn_params.ZeroPointAsNumpy())
                # Check that the scale and zero points are valid.
                if scale != 0 or zero_point != 0:
                    qnn_params = dict()
                    qnn_params['scale'] = scale
                    qnn_params['zero_point'] = zero_point
            return_list.append(TensorWrapper(tensor_idx, tensor, buffer, qnn_params))
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

    def get_tensor_type_str(self, tensor_type):
        """Get tensor type string representation when given TFLite tensor type"""
        try:
            from tflite.TensorType import TensorType
        except ImportError:
            raise ImportError("The tflite package must be installed")

        if tensor_type == TensorType.UINT8:
            return "uint8"
        if tensor_type == TensorType.FLOAT32:
            return "float32"
        if tensor_type == TensorType.INT32:
            return "int32"
        if tensor_type == TensorType.INT64:
            return "int64"
        raise NotImplementedError("Tensor type {} is currently not supported"
                                  .format(str(tensor_type)))

    def has_same_qnn_params(self, lhs_tensor, rhs_tensor):
        lhs_scale = lhs_tensor.qnn_params['scale']
        rhs_scale = rhs_tensor.qnn_params['scale']
        lhs_zero_point = lhs_tensor.qnn_params['zero_point']
        rhs_zero_point = rhs_tensor.qnn_params['zero_point']
        lhs_scale_value = lhs_scale
        rhs_scale_value = rhs_scale
        lhs_zero_point_value = lhs_zero_point
        rhs_zero_point_value = rhs_zero_point
        return lhs_scale_value == rhs_scale_value and \
                lhs_zero_point_value == rhs_zero_point_value

    def is_quantized(self, op):
        """Check if an input tensor is quantized."""
        try:
            from tflite.Operator import Operator
        except ImportError:
            raise ImportError("The tflite package must be installed")

        assert isinstance(op, Operator)
        input_tensors = self.get_input_tensors(op)
        first_tensor = input_tensors[0]
        return first_tensor.qnn_params is not None

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
        weight_tensor_idx = weight_tensor.tensor_idx

        # Constant nodes generation pass
        if self.is_const:
            # Stride const tensor
            if conv_type == 'conv2d':
                assert op.BuiltinOptionsType() == BuiltinOptions.Conv2DOptions
                op_options = op.BuiltinOptions()
                conv_options = Conv2DOptions()
                conv_options.Init(op_options.Bytes, op_options.Pos)
            elif conv_type == 'depthwise':
                assert op.BuiltinOptionsType() == BuiltinOptions.DepthwiseConv2DOptions
                op_options = op.BuiltinOptions()
                conv_options = DepthwiseConv2DOptions()
                conv_options.Init(op_options.Bytes, op_options.Pos)
            else:
                raise ValueError(
                    'Operator {} is not supported for frontend TFLite.'.format(conv_type))

            stride_h = conv_options.StrideH()
            stride_w = conv_options.StrideW()
            self.nn_stride(stride_h, stride_w)

            # RELU6 consts value
            fused_activation_fn = conv_options.FusedActivationFunction()
            if fused_activation_fn == ActivationFunctionType.RELU6:
                self.nn_float(6.0)

            # weight tensor type should be UINT8 (quantization) or FLOAT32
            weight_tensor_type = weight_tensor.tensor.Type()
            assert weight_tensor_type in (TensorType.UINT8, TensorType.FLOAT32)
            weight_tensor_type_str = self.get_tensor_type_str(weight_tensor_type)

            # in_expr = self.get_expr(input_tensor_idx)
            weight_value = self.get_tensor_value(weight_tensor)

            # TFLite kernel layout:
            # convolution:
            # OC KH KW IC, we require KH KW IC OC (HWIO)
            # depthwise convolution:
            # 1 KH KW C(input_c * depth_multiplier), we require
            # KH KW IC M (depth_multiplier) (HWOI)
            weight_value = weight_value.transpose((1, 2, 3, 0))
            self.nn_new_const(weight_tensor_idx, weight_value, dtype=weight_tensor_type_str)

            if len(input_tensors) == 3:
                bias_tensor = input_tensors[2]
                bias_tensor_idx = bias_tensor.tensor_idx
                bias_tensor_type = bias_tensor.tensor.Type()
                # bias tensor type should be INT32 (quantization) or FLOAT32
                assert bias_tensor_type in (TensorType.INT32, TensorType.FLOAT32)
                bias_tensor_type_str = self.get_tensor_type_str(bias_tensor_type)
                self.nn_new_const(bias_tensor_idx, self.get_tensor_value(bias_tensor),
                                 dtype=bias_tensor_type_str)
            return
        # End of Constant nodes generation pass

        # Non-Constant nodes generation Pass
        input_tensor = input_tensors[0]
        input_tensor_idx = input_tensor.tensor_idx

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "output tensors length should be 1"
        output_tensor = output_tensors[0]
        output_tensor_idx = output_tensor.tensor_idx
        output_tensor_type = output_tensor.tensor.Type()
        output_tensor_type_str = self.get_tensor_type_str(output_tensor_type)

        is_depthwise_conv = False
        if conv_type == 'conv2d':
            assert op.BuiltinOptionsType() == BuiltinOptions.Conv2DOptions
            op_options = op.BuiltinOptions()
            conv_options = Conv2DOptions()
            conv_options.Init(op_options.Bytes, op_options.Pos)
        elif conv_type == 'depthwise':
            is_depthwise_conv = True
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
        padding = conv_options.Padding()
        fused_activation_fn = conv_options.FusedActivationFunction()

        _, input_h, input_w, input_c = input_tensor.tensor.ShapeAsNumpy()

        if is_depthwise_conv:
            # TFLite depthwise convolution kernel layout is:
            # 1 KH KW C(input_c * depth_multiplier)
            _, kernel_h, kernel_w, in_channels = weight_tensor.tensor.ShapeAsNumpy()
            assert in_channels == input_c * depth_multiplier
        else:
            output_channels, kernel_h, kernel_w, _ = weight_tensor.tensor.ShapeAsNumpy()

        dilated_kernel_h = dilation_h * (kernel_h - 1) + 1
        dilated_kernel_w = dilation_w * (kernel_w - 1) + 1

        params = {'kernel_size': [kernel_h, kernel_w],
                  'strides': [stride_h, stride_w],
                  'dilation': [dilation_h, dilation_w],
                  'padding': [0, 0],
                  'data_layout': 'NHWC'}

        if is_depthwise_conv:
            params['channels'] = int(in_channels)
            params['groups'] = int(in_channels)
            params['kernel_layout'] = 'HWOI'
        else:
            params['channels'] = int(output_channels)
            params['kernel_layout'] = 'HWIO'


        if input_tensor.qnn_params:
            qnn_conv2d_params = dict(params)
            qnn_conv2d_params['input_zero_point'] = input_tensor.qnn_params['zero_point']
            qnn_conv2d_params['kernel_zero_point'] = weight_tensor.qnn_params['zero_point']
            qnn_conv2d_params['out_dtype'] = 'int32'
            qnn_conv2d_params['input_scale'] = input_tensor.qnn_params['scale']
            qnn_conv2d_params['kernel_scale'] = weight_tensor.qnn_params['scale']
            #out = _qnn.opconv2d(in_expr, weight_expr, **qnn_conv2d_params)
        #else:
        if input_tensor_idx not in self.tensor_tab:
            raise ValueError("Can not find tensor_id {} in tensor_tab".format(input_tensor_idx))
        data_nodes = self.tensor_tab[input_tensor_idx]
        weight_nodes = self.tensor_tab[weight_tensor_idx]
        conv_out_nodes = self.nn_conv2d(data_nodes, weight_nodes, stride_nodes, params)
        op_out_nodes = conv_out_nodes
        # if we have bias
        if len(input_tensors) == 3:
            bias_tensor = input_tensors[2]
            bias_tensor_idx = bias_tensor.tensor_idx
            bias_nodes = self.tensor_tab[bias_tensor_idx]
            bias_out_nodes = self.nn_bias_add(conv_out_nodes, bias_nodes)
            op_out_nodes = bias_out_nodes
        # If we have fused activations
        if fused_activation_fn != ActivationFunctionType.NONE:
            fused_act_out_nodes = self.nn_fused_activation(bias_out_nodes, fused_activation_fn)
            op_out_nodes = fused_act_out_nodes

        self.tensor_tab[output_tensor_idx] = op_out_nodes
        return op_out_nodes


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


def from_tflite(model): #, shape_dict, dtype_dict):
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

    tensor_tab = {}
    #for model_input in model_inputs:
    #     nn_add_input(model_input)
    #     model_input_name = get_tensor_name(subgraph, model_input)
    #     shape = shape_dict[model_input_name] if model_input_name in shape_dict else None
    #     dtype = dtype_dict[model_input_name] if model_input_name in dtype_dict else "float32"
    #     #exp_tab.set_expr(model_input_name, _expr.var(model_input_name, shape=shape, dtype=dtype))

    # op code in model
    op_converter = OperatorConverter(model, subgraph, tensor_tab)
    op_converter.check_unsupported_ops()

    op_converter.is_const = True
    op_converter.convert_op_to_hexagon_nn()

    for model_input in model_inputs:
          op_converter.nn_add_input(model_input)

    op_converter.is_const = False
    op_converter.convert_op_to_hexagon_nn()

    for model_output in model_outputs:
          op_converter.nn_add_output(model_output)

    print(tensor_tab)

    # params and outputs
    # params = {k:_nd.array(np.array(v)) for k, v in exp_tab.params.items()}
    # outputs = [exp_tab.get_expr(get_tensor_name(subgraph, i)) for i in model_outputs]
    # outputs = outputs[0] if len(outputs) == 1 else _expr.Tuple(outputs)
    # func = _expr.Function(analysis.free_vars(outputs), outputs)
    # mod = _module.Module.from_expr(func)
    # return mod, params

import os

tflite_model_file = os.path.join("../models/", "mobilenet_v1_0.75_224_conv1.tflite")
tflite_model_buf = open(tflite_model_file, "rb").read()
model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)
from_tflite(model)
