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
        self.kernel_size = kernel_size # Recommend odd integer
        self.stride = stride
        self.padding = kernel_size // 2
        self.weights = self.set_weights()
        #print('shape_r: ', self.weights.shape)
        self.bias = 0.1


    def set_weights(self) -> np.ndarray :
        fan_in  = self.in_channels * self.kernel_size * self.kernel_size # In-channels, kernel_h,
        fan_out = self.on_channels  * self.kernel_size * self.kernel_size # Out-channels, kernel_h, kernel_w
        #print(fan_in, fan_out)
        limit   = (-np.sqrt(6.0 / (fan_in + fan_out)), np.sqrt(6.0 / (fan_in + fan_out)))
        return  rng.uniform(*limit, size=(self.n_dataset, self.on_channels, self.kernel_size, self.kernel_size))


    def forward(self, data) -> np.ndarray:
        # Remaps variables to remove self class dictionary calls.
        f = self.activation
        k_s = self.kernel_size
        st  = self.stride
        pd  = self.padding
        b   = self.bias

        dm  = (self.d_inputs - (self.padding * 2)) // self.stride  # How many times kernel will move across a dimension
        out_conv = np.zeros((len(data), self.on_channels, self.d_inputs//st, self.d_inputs//st)) # (Number of image, output in layer, H data, W data)
        #print('OUTPUT: ', out_conv.shape)
        #print("shape: ", out.shape)
        print('Data: ', data.shape) # TODO: Get forward to work with second layer for more than one input
        for d_i, datum in enumerate(data): # Separates single data image to be passed through layer of neurons
            print(datum.shape)
            for k_i, k in enumerate(self.weights[d_i]): # Singles out weight/kernels for each neuron
                for r in range(pd, dm*st, st): # Singles out image row with consideration to stride length
                    for c in range(pd, dm*st, st): # Singles out column with consideration to stride
                        #print(datum[r:r+k_s, c:c+k_s])
                        product = (np.sum(datum[r:r+k_s, c:c+k_s] * k) + b) / k_s**2 # Pixel calculation

                        # Non-linear activation function per pixel; (pd//st+c//st) centers outputs
                        out_conv[d_i, k_i, r//st + pd//st, c//st + pd//st] = f(product)
        return out_conv


class Pooling():
    def forward(self, out_conv:np.ndarray, reduction:int=2) -> np.ndarray:
        o_d = len(out_conv[0,0]) # HxW of the pool inputs (convolution outputs)
        d = reduction # Downsample amount
        m = o_d//d # Filter movement length
        out = np.zeros((len(out_conv), len(out_conv[0]), m, m))

        for d_i, data in enumerate(out_conv): # Separates single data image for entire layer
            for k in range(len(out_conv[0])): # Runs though every kernel for neuron layer
                for r_i, r in enumerate(range(0, o_d, d)):
                    for c_i, c in enumerate(range(0, o_d, d)):
                        window = data[k, r:r+d, c:c+d] # Get window to downsample
                        out[d_i, k, r_i, c_i] = np.max(window) # Places downsampled pixel in new downsampled array
        return out


if __name__ == "__main__":

    relu:function    = lambda x:     max(0.0, x)
    gelu:function    = lambda x:     0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * x + 0.044715 * x**3))
    sigmoid:function = lambda x:     1 / (1 + np.exp(-x))
    softmax:function = lambda x, i:  (np.exp(x[i]))/(sum(np.exp(x)))


    folder = Path("../data/test_data_256x256")
    paths = sorted(folder.glob("*.png"))
    dataset = np.stack([np.array(Image.open(p).convert("L"), dtype=np.float32) / 255.0 for p in paths]) # Converts images to grayscale dataset
    input_dimension = len(dataset[0])
    pool = Pooling()

    # Layer 0
    net_0 = Convolution(
        activation=gelu,
        number_dataset=len(dataset),
        input_channels=1,
        output_channels=2,
        dimension=input_dimension,
        kernel_size=5,
        stride=1
    )
    out_conv_0 = net_0.forward(dataset)
    #print('PNEW :', dataset.shape)
    out_0 = pool.forward(out_conv_0)
    #print('EW :', out_0.shape)
    print('OO: ', len(out_0[0,0]))
    '''
    # Layer 1
    net_1 = Convolution(
        activation=gelu,
        number_dataset = 5,
        input_channels = 2,
        output_channels = 4,
        dimension = 128,
        kernel_size=5,
        stride=1
    )
    out_1 = net_1.forward(out_0)
    #print('A NEW :', out_conv_1.shape)
    #out_1 = pool.forward(out_conv_1)
    print('NEW :', out_1.shape)
    '''



    #print('OUTPUT: ', out_conv.shape)
    #out = pool.forward(out_conv)
    #print(f'Out Dim: {out.shape}')

    # Graph information
    fig, axes = plt.subplots(len(out_0), len(out_0[0])+1, figsize=(12, 8), constrained_layout=True)
    #print('NEW SHAPE: ', axes.shape)

    for r in range(len(axes)):
        axes[r,0].imshow(dataset[r], cmap='gray') # Maps original dataset to first row
        axes[r,0].axis('off')
        for c in range(len(axes[0])-1):
            axes[r,c+1].imshow(out_0[r,c], cmap='gray') # Maps kernel outputs to successive rows
            axes[r,c+1].axis('off')

    #axes[:,:].axis("off")
    plt.show()

