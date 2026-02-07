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
        bias:int,
        test:bool=False
    ):
        self.layer_number = layer_number
        self.X = forward_data
        self.N, self.C_i, self.H, self.W = input_shape
        self.C_o = self.C_i * output_multiple
        self.fl = kernel
        self.s = stride
        self.p = padding
        self.test = test

        # Container for storing data input for calculation $\lfloor{\frac{length+2p}{s}}\rfloor$
        self.X_c = np.zeros(
            ( self.N,
            self.C_i,
            int((self.H + (2*self.p))),
            int((self.W + (2*self.p)))
            )
        )
        self.b = np.full(self.C_o, bias)
        self.U = self.set_weights(test=self.test)

        print(f'Data: {self.X.shape}')
        print(f'Data Padded: {self.X_c.shape}')
        print(f'Weights: {self.U.shape}')
        print(f'Bias:  {self.b.shape}')

    def forward(self):
        N, C_i, C_o, H, W = self.N, self.C_i, self.C_o, self.H, self.W
        fl, s, p, b = self.fl, self.s, self.p, self.b

        # How many spaces the kernel slides across of the x&y-axis
        H_o, W_o = ((H+2*p-fl)//s)+1, ((W+2*p-fl)//s)+1

        # Places inputs into padded container
        self.X_c[:, :, p:p+H, p:p+W] = self.X # Padded inputs
        U = self.U # Weights
        output = np.zeros((N, C_o, H_o, W_o))
        print("\n")
        print(f"_o ({H_o}, {W_o})")
        print(f'Outputs: {output.shape}')

        for i, X in enumerate(self.X_c): # Extracts all feature maps for a single image set
            for o in range(C_o):
                for y in range(H_o):
                    for v in range(W_o):
                        accumulation:float=0 # Accumulated pixel value for output
                        for j in range(C_i):

                            # Extracts the kernel size window for a single feature image
                            X_w = X[j, y*s:y*s+fl, v*s:v*s+fl]

                            # Multiplies and summates single weight window by the input window
                            single_window =  np.sum(U[o,j] * X_w)

                            # Places in single pixel accumulation for each operations
                            accumulation += single_window

                        # Places single pixel in output array and adds in bias
                        output[i, o, y, v] = accumulation + b[o]
        return output

        #for img_set in enumerate()

    def set_weights(self, test=False) -> np.ndarray :
        """
        Creates Xavier-scaled weights for the convolution kernels.

        Returns
        -------
        np.ndarray
            Tensor shaped (out_channels, in_channels, kernel_h, kernel_w).
        """
        if not test:
            fan_in  = self.C_i * self.fl * self.fl # In-channels, kernel_h,
            fan_out = self.C_o  * self.fl * self.fl # Out-channels, kernel_h, kernel_w
            limit   = (-np.sqrt(6.0 / (fan_in + fan_out)), np.sqrt(6.0 / (fan_in + fan_out)))
            print("LIMIT W", limit)
            return  rng.uniform(*limit, size=(self.C_o, self.C_i, self.fl, self.fl)) #(2, 1, 5, 5)
        else:
            print(f'LAYER {self.layer_number} CONVOLUTION FORWARD IS USING TEST WEIGHTS.')
            filter_matrices = np.zeros((self.C_o, self.C_i, self.fl, self.fl))
            for v in range(self.C_o):
                for u in range(self.C_i):
                    filter_matrices[v,u] = self.test_filter()
            return filter_matrices


    # Create an identify matrix for matrix multiplication
    def test_filter(self, order=2):
        matrix_space = np.zeros((self.fl, self.fl))
        pattern = {0:(0,0), 1:(-1,1), 2:(1,-2,1)}

        if order == 0: return matrix_space # Default return

        # How many indices to shift to the left in diagonal render
        shift = len(pattern[order]) - len(pattern[order-1])

        for i, row in enumerate(matrix_space):
            for n in range(len(pattern[order])):
                index = n + i - shift # Shift to correct placement
                if index >= len(row): index = (index-len(row))*-1 # Error correction bottom matrices
                row[index] = pattern[order][n] # Sets each indices
        return matrix_space

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


def graph_output(data, raw, layers=1):
        # Graph information
    fig, axes = plt.subplots(len(data)*layers, len(data[0])+1, figsize=(6, 20), constrained_layout=True)

    for n in range(layers):

        for r in range(len(axes)//layers):
            axes[r,0].imshow(raw[r,0], cmap='gray') # Maps original dataset to first row
            axes[r,0].axis('off')

            if n == 0:
                for c in range(len(axes[0])-1):
                    axes[r,c+1].imshow(data[r,c], cmap='gray') # Maps kernel outputs to successive rows
                    axes[r,c+1].axis('off')
    plt.show()


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
    print("EVAL: \n", evaluation)

    data_raw = np.array([[np.array(Image.open(p).convert("L"), dtype=np.float32) / 255.0] for p in total_paths])

    kernel_size = 5
    output_increase = 2
    convolution_stride = 1
    padding_thickness = kernel_size//2
    bias_base = 0.1

    # Creates first layer of network
    print("\n[Layer 0]")
    cl_0 = Convolution(
        layer_number = 0,
        forward_data = data_raw,
        input_shape = data_raw.shape,
        kernel = kernel_size,
        output_multiple = output_increase,
        stride = convolution_stride,
        padding = padding_thickness,
        bias = bias_base,
        test = True
    )
    output = cl_0.forward()
    graph_output(data=output, raw=data_raw, layers=1)

    begin_time = time.perf_counter()
