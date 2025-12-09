import numpy as np
import matplotlib.pyplot as plt
from typing import Callable
from pathlib import Path
from PIL import Image


class Convolution():
    def __init__(self, activation:Callable[[float], float], number_inputs:int=1, number_outputs:int=3, kernel_size:int=5, stride:int=1):
        self.activation = activation
        self.n_inputs = number_inputs
        self.n_outputs = number_outputs
        self.kernel_size = kernel_size # Must be odd
        self.stride = stride
        self.padding = kernel_size // 2

    def forward(self):

        f = self.activation
        i_n = self.n_inputs
        o_n = self.o_outputs
        k_s = self.kernel_size
        st  = self.stride
        pd  = self.p

        # TODO: fix code below with standard parameters

        out = np.zeros((f_n, d_s, d_s))

        for data in dataset:
            for index, k in enumerate(self.weights):
                #print(index)
                #print('l: ', l)
                for r in range(p, l-p, s):
                    #print('r: ', r)
                    for c in range(p, l-p, s):
                        #print('c: ', c)
                        #print('D: ', data[r:r+f_s, c:c+f_s])
                        #print('K: ', k)
                        #print('b: ', b)
                        #print('AP: ', np.sum(data[r:r+f_s, c:c+f_s] @ k) + b)
                        product = (np.sum(data[r:r+f_s, c:c+f_s] * k) + b) / f_s**2
                        out[index, r+p, c+p] = gelu(product)
        return out



if __name__ == "__main__":

    rng = np.random.default_rng()
    relu =    lambda x:     max(0.0, x)
    gelu =    lambda x:     0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * x + 0.044715 * x**3))
    sigmoid = lambda x:     1 / (1 + np.exp(-x))
    softmax = lambda x, i:  (np.exp(x[i]))/(sum(np.exp(x)))


    folder = Path("../data/test_data_256x256")
    paths = sorted(folder.glob("*.png"))
    dataset = np.stack([np.array(Image.open(p).convert("L"), dtype=np.float32) / 255.0 for p in paths]) # Converts images to grayscale dataset


    net = Convolution(relu,1,2,3,1)

    print(net.activation(-1.0))

