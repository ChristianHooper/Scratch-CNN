import numpy as np
import matplotlib.pyplot as plt
from typing import Callable
from pathlib import Path
from PIL import Image

rng = np.random.default_rng()
function = Callable[[float], float]

class Convolution():
    def __init__(self, activation:function, number_dataset:int, input_channels:int=1, output_channels:int=3, dimension:int=256, kernel_size:int=5, stride:int=1):
        self.activation = activation
        self.n_dataset = number_dataset
        self.in_channels = input_channels
        self.on_channels = output_channels
        self.d_inputs = dimension
        self.kernel_size = kernel_size # Must be odd
        self.stride = stride
        self.padding = kernel_size // 2
        self.weights = self.set_weights()
        print('shape_r: ', self.weights.shape)
        self.bias = 0.1


    def set_weights(self):
        fan_in  = self.in_channels * self.kernel_size * self.kernel_size # In-channels, kernel_h,
        fan_out = self.on_channels  * self.kernel_size * self.kernel_size # Out-channels, kernel_h, kernel_w
        #print(fan_in, fan_out)
        limit   = (-np.sqrt(6.0 / (fan_in + fan_out)), np.sqrt(6.0 / (fan_in + fan_out)))
        return  rng.uniform(*limit, size=(self.n_dataset, self.on_channels, self.kernel_size, self.kernel_size))


    def forward(self, data):

        f = self.activation
        i_n = self.in_channels
        o_n = self.on_channels
        i_d = self.d_inputs
        k_s = self.kernel_size
        st  = self.stride
        pd  = self.padding
        b   = self.bias

        # TODO: fix code below with standard parameters
        dm  = (self.d_inputs - (self.padding * 2)) // self.stride # How many times kernel will move across a dimension
        out = np.zeros((len(dataset), self.on_channels, self.d_inputs, self.d_inputs)) # (Number of image, output in layer, H data, W data)
        print("shape: ", out.shape)

        for d_i, data in enumerate(dataset):
            for k_i, k in enumerate(self.weights[d_i]):
                #print(index)
                print('w: ', k_i)
                for r in range(pd, dm, st):
                    #print('r: ', r)
                    for c in range(pd, dm, st):
                        #print('c: ', c)
                        #print('D: ', data[r:r+f_s, c:c+f_s])
                        #print('K: ', k)
                        #print('b: ', b)
                        #print('AP: ', np.sum(data[r:r+f_s, c:c+f_s] @ k) + b)
                        product = (np.sum(data[r:r+k_s, c:c+k_s] * k[d_i]) + b) / k_s**2
                        out[d_i, k_i, r, c] = f(product) # Non-linear activation function
        return out

# Feb 6th @ 0830
# The only thing that will get me through this life is the consistent effort of work in pursuit of knowing; I have seen that the life of comfort outside of these bonds and it holds not fulfillment, a meaningless existence.

if __name__ == "__main__":

    relu:function    = lambda x:     max(0.0, x)
    gelu:function    = lambda x:     0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * x + 0.044715 * x**3))
    sigmoid:function = lambda x:     1 / (1 + np.exp(-x))
    softmax:function = lambda x, i:  (np.exp(x[i]))/(sum(np.exp(x)))


    folder = Path("../data/test_data_256x256")
    paths = sorted(folder.glob("*.png"))
    dataset = np.stack([np.array(Image.open(p).convert("L"), dtype=np.float32) / 255.0 for p in paths]) # Converts images to grayscale dataset
    input_dimension = len(dataset[0])

    net = Convolution(
        activation=relu,
        number_dataset=len(dataset),
        input_channels=1,
        output_channels=2,
        dimension=input_dimension,
        kernel_size=3,
        stride=1
    )

    out = net.forward(dataset)
    print(f'Out Dim: {len(out)} | {len(out[0])}')

    '''
    # Graph information
    fig, axes = plt.subplots(len(out), len(out[0]), figsize=(12, 8), constrained_layout=True)
    fig, ((x0, x1, x2, x3), (x4, x5, x6, x7)) = plt.subplots(2, 4, figsize=(12, 8), constrained_layout=True)
    x0.imshow(image_array, cmap='gray')
    x1.imshow(out[0],      cmap='gray')
    x2.imshow(out[1],      cmap='gray')
    x3.imshow(out[2],      cmap='gray')

    #x4.imshow(edge_color,        cmap=None if channels==4 else 'grey')
    x5.imshow(inputs[0],     cmap='gray')
    x6.imshow(inputs[1],        cmap='gray')
    x7.imshow(inputs[2],        cmap='gray')
    plt.axis("off")
    plt.show()

    print(net.activation(-1.0))
    '''
