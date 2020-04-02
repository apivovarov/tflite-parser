# Hexagon NN Converter

# Intro
This project allows to:
- Convert tflite model to Hexagon C program which uses NNLib API
- Compile converted model for variety of Hexagon Devices / platforms
- Test compiled model on the device

# Convert from TFLite
To convert tflite model to Hexagon C program which uses NNLib API use `from_tflite.py` script.
The scrip supports both quantized and float models.
In order to get reasonable compiled model performance it is recommended to use **quantized** tflite model!
```
./from_tflite.py <tflite_model>
```
This script will generate `dlr_hexagon_model.h` file.
The file contains model graph assembled using NNLib API and model weights.

# Compile converted model
To compile converted model use script `compile_deploy_run.py`.
This script can compile the model for Android or Linux platforms (`dlr_hexagon_model_so`). 
Also it can compile Hexagon NNLib for particular Hexagon DSP model (`libhexagon_nn_skel.so`).
See list of supported Hexagon devices in Appendix A
```
# Android aarch64, SoC QCS605 (V65)
compile_deploy_run.py -T qcs605 -N

# Linux aarch64
compile_deploy_run.py -T qcs605 -N -L

# Linux gnueabihf (32 bit HF)
compile_deploy_run.py -T qcs605 -N -L -32

```
If you repeatedly compile models for particular Hexagon device you can skip Hexagon NNLib compilation by adding option `-M`
```
# skip libhexagon_nn_skel.so recompilation for QCS605 (V65)
compile_deploy_run.py -T qcs605 -N -M
```

# Deploy and run compiled model
If edge device is accessible from the host via `adb` (locally or remotely)
the compiled model can be deployed and executed on the device. 
Remove option `-N` from `compile_deploy_run.py` commands above.
If multiple devices is connected you can specify device number using environment variable `ADB_DEVICE`.
If device is connected to remote server you can specify ADB Server using `ADB_SERVER`.
```
export ADB_SERVER=10.150.22.44
export ADB_DEVICE=34201010ff

compile_deploy_run.py -T qcs605 
```

# Appendix A - Supported devices
List of supported Hexagon devices:
```
sdm835
sdm820
sdm660
sdm845
sdm670
sdm710
qcs605
sm8150
sm6150
qcs405
sxr1130
sm7150
sm6125
sm8250
rennell
saipan
```

# Appendix B - Supported models
Hexagon converter supports almost all quantized TFLite models from [tfhub](https://tfhub.dev/)
such as Mobilenet v1, v2 and Inception v1, v2, v3 (except of v4).

List of supported ops:
```
CONV_2D
DEPTHWISE_CONV_2D
RESHAPE
SOFTMAX
AVERAGE_POOL_2D
MAX_POOL_2D
ADD
MUL
CONCATENATION
```
