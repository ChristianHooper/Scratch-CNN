import numpy as np
import matplotlib.pyplot as plt
import time
import math
from typing import Callable
from pathlib import Path
from PIL import Image


class Convolution():
    def __init__(self,
        layer_number:int,
        forward_data:np.ndarray,
        input_shape:tuple,
        kernel:int,
        output_multiple:int,
        stride:int,
        padding:int,
        bias:int
    ):
        self.layer_number = layer_number
        self.X = forward_data
        self.N, self.C_i, self.H, self.H = input_shape
        self.C_o = self.C_i * output_multiple
        self.fl = kernel
        self.s = stride
        self.p = padding
        print(np.floor(self.H + (2*self.p) - self.fl)/self.s)

        # Container for storing data input for calculation $\lfloor{\frac{length+2p}{s}}\rfloor$
        self.X_c = np.zeros(
            ( self.N,
            self.C_o,
            int((self.H + (2*self.p))/self.s),
            int((self.H + (2*self.p))/self.s)
            )
        )
        self.b = np.full(self.C_o, bias)
        self.W = self.set_weights()
        
        print(f'Data: {self.X.shape}')
        print(f'Data Padded: {self.X_c.shape}')
        print(f'Weights: {self.W.shape}')
        print(f'Bias:  {self.b.shape}')


    def set_weights(self) -> np.ndarray :
        """
        Creates Xavier-scaled weights for the convolution kernels.

        Returns
        -------
        np.ndarray
            Tensor shaped (out_channels, in_channels, kernel_h, kernel_w).
        """
        fan_in  = self.C_i * self.fl * self.fl # In-channels, kernel_h,
        fan_out = self.C_o  * self.fl * self.fl # Out-channels, kernel_h, kernel_w
        limit   = (-np.sqrt(6.0 / (fan_in + fan_out)), np.sqrt(6.0 / (fan_in + fan_out)))
        return  rng.uniform(*limit, size=(self.C_o, self.C_i, self.fl, self.fl)) #(2, 1, 5, 5)


class Activation():
    forward_activation = {
        'relu':     lambda x: np.max(x),
        'gelu':     lambda x: 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * x + 0.044715 * x**3)),
        'sigmoid':  lambda x: 1 / (1 + np.exp(-x)),
        'softmax':  lambda x: (np.exp(x[i]))/(sum(np.exp(x))) # TODO: work out i
    }

    def __init__(self, function_type:str='relu', ):
        self.f_forward = forward_activation[function_type]

    # Activation function for forwards pass
    def forward(self, data:np.ndarray) -> np.ndarray:
        return self.f_forward(data)


class Pooling():
    def __init__(self):
        None


class Head():
    def __init__(self):
        None


if __name__ == "__main__":
    rng = np.random.default_rng()

    # Data extracting for directories to train CNN
    test_directory = Path("../data/test_data_256x256")
    test_target_directory = Path("../data/test_data_256x256/target")

    test_paths = sorted(test_directory.glob("*.png"))
    test_target_paths = sorted(test_target_directory.glob("*.png"))
    total_paths = test_paths + test_target_paths

    # Class information extracted from data
    k_0, k_1 = len(test_paths), len(test_target_paths) # Images in class 0
    K = np.array([k_0, k_1]) # All class lengths

    # One-hot data evaluation metric
    evaluation = np.zeros((len(K), sum(K)))
    evaluation[0, 0:k_0] = 1 # Class 0 one-hot
    evaluation[1, k_0:] = 1  # Class 1 one-hot
    print("EVAL\n", evaluation)

    data_raw = np.array([[np.array(Image.open(p).convert("L"), dtype=np.float32) / 255.0] for p in total_paths])
    print("SHAPE: ", data_raw.shape)

    kernel_size = 5
    output_increase = 2
    convolution_stride = 1
    padding_thickness = kernel_size//2
    bias_base = 0.1

    # Creates first layer of network
    cl_0 = Convolution(
        layer_number = 0,
        forward_data = data_raw,
        input_shape = data_raw.shape,
        kernel = kernel_size,
        output_multiple = output_increase,
        stride = convolution_stride,
        padding = padding_thickness,
        bias = bias_base
    )



    begin_time = time.perf_counter()





