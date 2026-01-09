"""
Minimal NumPy CNN with a runnable test pipeline.

The module includes a convolution layer, a pooling layer, and an optional
Cython-backed pooling path for faster downsampling.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Callable
from pathlib import Path
from PIL import Image


try: # Configures cython functions
    import fast_ops
    HAS_FAST_OPS = True
except ImportError:
    HAS_FAST_OPS = False

rng = np.random.default_rng()
function = Callable[[float], float]

class Convolution():
    """
    Single convolution layer with square kernels, fixed padding, and stride.

    Stores weights, bias, and activation to produce feature maps from an input
    batch.
    """
    def __init__(self, activation:function, number_dataset:int, input_channels:int=1, output_channels:int=3, dimension:int=256, kernel_size:int=5, stride:int=1):
        """
        Initialize convolution parameters and weights.

        Parameters
        ----------
        activation : Callable[[float], float]
            Non-linear activation applied per output pixel.
        number_dataset : int
            Dataset size; kept for potential future batching logic.
        input_channels : int, default=1
            Number of channels in the input tensor.
        output_channels : int, default=3
            Number of convolution kernels / output channels.
        dimension : int, default=256
            Height/width of the square input.
        kernel_size : int, default=5
            Size of the square kernel (odd values recommended).
        stride : int, default=1
            Convolution stride.
        """
        self.activation = activation
        self.n_dataset = number_dataset
        self.in_channels = input_channels
        self.on_channels = output_channels
        self.d_inputs = dimension
        self.kernel_size = kernel_size # Recommend odd integer
        self.stride = stride
        self.padding = kernel_size // 2
        self.weights = self.set_weights()
        self.bias = 0.02


    def set_weights(self) -> np.ndarray :
        """
        Creates Xavier-scaled weights for the convolution kernels.

        Returns
        -------
        np.ndarray
            Tensor shaped (out_channels, in_channels, kernel_h, kernel_w).
        """
        fan_in  = self.in_channels * self.kernel_size * self.kernel_size # In-channels, kernel_h,
        fan_out = self.on_channels  * self.kernel_size * self.kernel_size # Out-channels, kernel_h, kernel_w
        limit   = (-np.sqrt(6.0 / (fan_in + fan_out)), np.sqrt(6.0 / (fan_in + fan_out)))
        return  rng.uniform(*limit, size=(self.on_channels, self.kernel_size, self.kernel_size)) #(2, 1, 5, 5)


    def forward(self, data) -> np.ndarray:
        """
        Compute the convolution output for a batch and apply activation.

        Parameters
        ----------
        data : np.ndarray
            Input shaped (batch, channels, height, width).

        Returns
        -------
        np.ndarray
            Activated feature maps shaped (batch, out_channels, out_h, out_w).
        """
        # Remaps variables to remove self class dictionary calls.
        f = self.activation
        k_s = self.kernel_size
        i_n = self.in_channels - 1 # Static needs to be fixed
        st  = self.stride
        pd  = self.padding # $frac{d+2p-k}{s}$
        n = self.d_inputs
        b   = self.bias
        out_conv = np.empty((len(data), self.on_channels, self.d_inputs//st, self.d_inputs//st)) # (Number of image, output in layer, H data, W data)
        print('data in:  ', data.shape)
        print('weight:   ', self.weights.shape)

        for i, image in enumerate(data):
            for datum in image:
                x = np.pad(datum, ((pd, pd,), (pd, pd)), mode='edge') # Pads input image (wrap, edge, constant(default))
                for d, w in enumerate(self.weights):
                    for r in range(0, n, st):
                        for c in range(0, n, st):
                            product = np.sum(w * x[r:r+k_s, c:c+k_s] + b)
                            #product = np.sum(w.T @ x[r:r+k_s, c:c+k_s])+b TODO: Proper equivalence
                            out_conv[i, d, r, c] = f(product)
        return out_conv



class Pooling():
    """Max-pooling layer with an optional Cython acceleration path."""
    def forward(self, out_conv:np.ndarray, reduction:int=2) -> np.ndarray:
        """
        Downsample feature maps with max-pooling.

        Parameters
        ----------
        out_conv : np.ndarray
            Activations shaped (batch, channels, height, width).
        reduction : int, default=2
            Pooling window size and stride.

        Returns
        -------
        np.ndarray
            Pooled tensor shaped (batch, channels, height/reduction, width/reduction).
        """
        pooled_input = np.ascontiguousarray(out_conv, dtype=np.float32)

        if HAS_FAST_OPS: return fast_ops.max_pool4d(pooled_input, reduction)

        o_d = len(pooled_input[0,0]) # HxW of the pool inputs (convolution outputs)
        d = reduction # Downsample amount
        m = o_d//d # Filter movement length
        out = np.zeros((len(pooled_input), len(pooled_input[0]), m, m), dtype=pooled_input.dtype)

        for d_i, data in enumerate(pooled_input): # Separates single data image for entire layer
            for k in range(len(pooled_input[0])): # Runs though every kernel for neuron layer
                for r_i, r in enumerate(range(0, o_d, d)):
                    for c_i, c in enumerate(range(0, o_d, d)):
                        window = data[k, r:r+d, c:c+d] # Get window to downsample
                        out[d_i, k, r_i, c_i] = np.max(window) # Places downsampled pixel in new downsampled array
        print('data out: ', out.shape)
        return out


if __name__ == "__main__":

    relu:function    = lambda x:     max(0.0, x)
    gelu:function    = lambda x:     0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * x + 0.044715 * x**3))
    sigmoid:function = lambda x:     1 / (1 + np.exp(-x))
    softmax:function = lambda x, i:  (np.exp(x[i]))/(sum(np.exp(x)))

    folder = Path("../data/test_data_256x256")
    paths = sorted(folder.glob("*.png"))
    # Converts images to grayscale dataset (n images, 1, 256, 256)
    dataset = np.stack([[np.array(Image.open(p).convert("L"), dtype=np.float32) / 255.0] for p in paths])
    input_dimension = len(dataset[0,0])
    dataset_number = len(dataset)
    pool = Pooling()
    print("Cython:", HAS_FAST_OPS)

    begin_time = time.perf_counter()

    # Training Layers
    # Layer 0
    net_0 = Convolution(
        activation=gelu,
        number_dataset=dataset_number,
        input_channels=1,
        output_channels=2,
        dimension=input_dimension,
        kernel_size=5,
        stride=1
    )
    out_conv_0 = net_0.forward(dataset)
    out_0 = pool.forward(out_conv_0)
    elapsed = time.perf_counter() - begin_time
    print(f'L0 Elapsed: {elapsed:.3f}\n')
    elapsed = time.perf_counter()

    # Layer 1
    net_1 = Convolution(
        activation=gelu,
        number_dataset = dataset_number,
        input_channels = net_0.on_channels,
        output_channels = int(net_0.on_channels * 2),
        dimension = len(out_0[0,0]),
        kernel_size=5,
        stride=1
    )
    out_conv_1 = net_1.forward(out_0)
    out_1 = pool.forward(out_conv_1)
    #print(out_1.shape)
    elapsed = time.perf_counter() - elapsed
    print(f'L1 Elapsed: {elapsed:.3f}\n')
    elapsed = time.perf_counter()

    # Layer 2
    net_2 = Convolution(
        activation=gelu,
        number_dataset = dataset_number,
        input_channels = net_1.on_channels,
        output_channels = int(net_1.on_channels * 2),
        dimension = len(out_1[0,0]),
        kernel_size=5,
        stride=1
    )
    out_conv_2 = net_2.forward(out_1)
    out_2 = pool.forward(out_conv_2)
    #print(out_2.shape)
    elapsed = time.perf_counter() - elapsed
    print(f'L2 Elapsed: {elapsed:.3f}\n')
    elapsed = time.perf_counter()

    # Layer 3
    net_3 = Convolution(
        activation=gelu,
        number_dataset = dataset_number,
        input_channels = net_2.on_channels,
        output_channels = int(net_2.on_channels * 2),
        dimension = len(out_2[0,0]),
        kernel_size=5,
        stride=1
    )
    out_conv_3 = net_3.forward(out_2)
    out_3 = pool.forward(out_conv_3)
    #print(out_3.shape)
    elapsed = time.perf_counter() - elapsed
    print(f'L3 Elapsed: {elapsed:.3f}\n')
    elapsed = time.perf_counter()

    # Layer 4
    net_4 = Convolution(
        activation=gelu,
        number_dataset = dataset_number,
        input_channels = net_3.on_channels,
        output_channels = int(net_3.on_channels * 2),
        dimension = len(out_3[0,0]),
        kernel_size=5,
        stride=1
    )
    out_conv_4 = net_4.forward(out_3)
    out_4 = pool.forward(out_conv_4)
    #print(out_4.shape)
    elapsed = time.perf_counter() - elapsed
    print(f'L4 Elapsed: {elapsed:.6f}\n')



    # Classifier Head (Uses flattened feature vector and computes k evidence scores)
    k = 2 # Number of classes
    b = np.zeros((k)) # Bias
    n, c, h, w = out_4.shape
    lr = 0.1 # Learning rate

    # This vector is the encoded version of the network for feature detection through probabilistic assignment, meaning parts of the vector classify for certain features
    #flatten = out_4.reshape(n, c * h * w) # Flatten final output into a single vector
    #wt = np.random.randn(k, len(flatten[1])) * 0.01 # Weights
    flatten = out_4.mean(axis=(2,3))   # (n, c)  global avg pool $F\in{\mathbb{R}^{(N,C,H,W)}}\to{G\in{\mathbb{R}^{(N,C)}}}$
    wt = np.random.uniform(-1, 1, (k, c))
    print('GAP:   ', flatten.shape)

    y = np.array([0, 0, 1, 0, 0]) # Truth classifications; Class 1 is Aiko all others class 0
    #print(f"Hot Array: {one_hot} \n Shape: {one_hot.shape}")

    # /////[HEAD]//////////////////////////////////////////////////////////////////////////////

    # Class votes through weighted sum of all features; largest logits is the winning class
    logits = flatten @ wt.T + b # [i, j] (image, class) image i from class j (5, 2)
    print('LOGIT: ', logits.shape)
    # Softmax for creating a probability distribution of logits raw scores
    logits_shift = logits - logits.max(axis=1, keepdims=True) # Shifts logit to avoid expo map issues
    # $\Large\frac{e^{zk-kmax}}{\sum_k{e^{zk-kmax}}}$
    probs = np.exp(logits_shift) # Numerator
    probs /= probs.sum(axis=1, keepdims=True) # Denominator
    print('PROB:  ', probs.shape)

    # Loss
    # $-\frac{1}{N}\sum^N_n=1{log(P_n,y_n)}$
    print("LOGITS: ", logits)
    print("Probs:", probs)
    #print('Arrange: ', np.arange(n))
    print('Correct: ', y)
    #print("Prob Selection: ", probs[np.arange(n), y])
    #print("Prob LOG: ", -np.log(probs[np.arange(n), y]))
    print("Prob LOSS: ", np.mean(-np.log(probs[np.arange(n), y])))
    loss = np.mean(-np.log(probs[np.arange(n), y] + 1e-12)) # Small float; never log 0 (negative reverses log output)

    # Loss function derivative with respects to logits: $\frac{1}{n}(p_{n,k}-Y_{n,k})$
    one_hot = np.array(((1-y), (y))).T # TODO: Add and reorganize input data & set one-hot to the data
    # print('One-Hot:', one_hot)
    G = (probs - one_hot) / n # Distribution of how each logits should move to reduce loss
    print('G: ', G)

    # Loss function derivate with respects to weights: $z=g\cdot{w^T}+\vec{h}\to{z'=h}$
    L_w = G.T @ flatten # Derivative of loss
    L_b = G.sum(axis=0) # Derivative of bias

    wt = L_w # Moves weights
    b =  L_b # Moves bias

    # /////[END HEAD]//////////////////////////////////////////////////////////////////////////////

    print("\n(L/w):\n", L_w)
    print("Weights:\n", wt)
    print("\n(L/b)\n':", L_b)
    print("Bias:\n", b)

    print(f'Total Time: {(time.perf_counter() - begin_time):.3f}')

    # /////[GRAPH]//////////////////////////////////////////////////////////////////////////////
    layer = 5
    d_n = dataset_number

    # Graph information
    fig, axes = plt.subplots(len(out_0)*layer, len(out_0[0])+1, figsize=(6, 20), constrained_layout=True)

    for n in range(layer):

        for r in range(len(axes)//layer):
            axes[r,0].imshow(dataset[r,0], cmap='gray') # Maps original dataset to first row
            axes[r,0].axis('off')

            if n == 0:
                for c in range(len(axes[0])-1):
                    axes[r,c+1].imshow(out_0[r,c], cmap='gray') # Maps kernel outputs to successive rows
                    axes[r,c+1].axis('off')

            if n == 1:
                for c in range(len(axes[0])):
                    axes[r+d_n,c].imshow(out_1[r,c], cmap='gray') # Maps kernel outputs to successive rows
                    axes[r+d_n,c].axis('off')

            if n == 2:
                for c in range(len(axes[0])):
                    axes[d_n*n+r,c].imshow(out_2[r,c], cmap='gray') # Maps kernel outputs to successive rows
                    axes[d_n*n+r,c].axis('off')

            if n == 3:
                for c in range(len(axes[0])):
                    axes[d_n*n+r,c].imshow(out_3[r,c], cmap='gray') # Maps kernel outputs to successive rows
                    axes[d_n*n+r,c].axis('off')

            if n == 4:
                for c in range(len(axes[0])):
                    axes[d_n*n+r,c].imshow(out_4[r,c], cmap='gray') # Maps kernel outputs to successive rows
                    axes[d_n*n+r,c].axis('off')
    plt.show()
