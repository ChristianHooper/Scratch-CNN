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
        test:bool=False,
        rng_obj=np.random.default_rng()
    ):
        self.layer_number = layer_number
        self.X = forward_data
        self.N, self.C_i, self.H, self.W = input_shape
        self.C_o = self.C_i * output_multiple
        self.fl = kernel
        self.s = stride
        self.p = padding
        self.test = test
        self.rng_obj = rng_obj

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
        self.Y = np.zeros((self.N, self.C_o, self.H//self.s, self.W//self.s)) # Forward output for back propagation

        print(f'Data: {self.X.shape}')
        print(f'Data Padded: {self.X_c.shape}')
        print(f'Weights: {self.U.shape}')
        print(f'Bias:  {self.b.shape}\n')

    def forward_convolution(self):
        N, C_i, C_o, H, W = self.N, self.C_i, self.C_o, self.H, self.W
        fl, s, p, b = self.fl, self.s, self.p, self.b

        # How many spaces the kernel slides across of the x&y-axis
        H_o, W_o = ((H+2*p-fl)//s)+1, ((W+2*p-fl)//s)+1

        # Places inputs into padded container
        self.X_c[:, :, p:p+H, p:p+W] = self.X # Padded inputs
        U = self.U # Weights
        output = np.zeros((N, C_o, H//s, W//s))

        for i, X in enumerate(self.X_c): # Extracts all feature maps for a single image set
            for o in range(C_o):
                for y in range(H_o):
                    for v in range(W_o):

                        accumulation:float=0 # Accumulated pixel value for output
                        for j in range(C_i):

                            # Extracts the kernel size window for a single feature image
                            X_w = X[j, y*s:y*s+fl, v*s:v*s+fl]

                            # Multiplies and summates all weight windows by the input window one at a time
                            single_window =  np.sum(U[o,j] * X_w)

                            # Places in single pixel accumulation for each operations
                            accumulation += single_window

                        # Places single pixel in output array and adds in bias
                        output[i, o, y, v] = accumulation + b[o]

        print("Convolution Shape: ", output.shape)
        self.Y[:] = output[:]
        self.X_c[:] = 0 # Clears padding
        return self.Y

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
            return  self.rng_obj.uniform(*limit, size=(self.C_o, self.C_i, self.fl, self.fl)) #(2, 1, 5, 5)
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
                if index >= len(row): index = (index-len(row)) * -1 # Error correction bottom matrices
                row[index] = pattern[order][n] # Sets each indices
        return matrix_space


class Activation():
    def __init__(self, shape:tuple, function_type:str='relu'):
        self.forward_activation_functions = {
            'relu':     lambda x: np.maximum(x, 0),
            'gelu':     lambda x: 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * x + 0.044715 * x**3)),
            'sigmoid':  lambda x: 1 / (1 + np.exp(-x)),
            #'softmax':  lambda x: (np.exp(x[i]))/(sum(np.exp(x))) # TODO: work out i
        }
        self.f_forward = self.forward_activation_functions[function_type]
        self.Y_a = np.zeros(shape)

    # Activation function for forwards pass
    def forward_activation(self, data:np.ndarray) -> np.ndarray:
        self.Y_a[:] = self.f_forward(data)[:]
        print("Activation Shape: ", self.Y_a.shape)
        return self.Y_a


class Pooling():
    def __init__(self, stride, input_dimensions):
        self.N, self.C_o, H, W = input_dimensions
        self.H_p, self.W_p = (((H-stride)//stride)+1, ((W-stride)//stride)+1)
        self.s_p = stride
        self.mask = np.zeros((self.N, self.C_o, H, W), dtype=bool)
        self.Y_p =  np.zeros((self.N, self.C_o, self.H_p, self.W_p))
        self.Y_d = np.zeros((self.N, self.C_o, H, W))

    def forward_max_pooling(self, data):
        N, C_o, H_p, W_p = self.N, self.C_o, self.H_p, self.W_p
        s_p = self.s_p
        Y_p, mask = self.Y_p, self.mask

        for i, X in enumerate(data):
            for j in range(C_o):
                for y in range(H_p):
                    for v in range(W_p):

                        # Window ranges for downsampling
                        a_h, z_h, a_w, z_w = y*s_p, y*s_p+s_p, v*s_p, v*s_p+s_p

                        # Gets window for downsampling
                        window = X[j, a_h:z_h, a_w:z_w]

                        # Selects the index of the highest value in the window matrices
                        m_i = np.unravel_index(np.argmax(window), window.shape)

                        # Finds the highest value position in window vector (If same default closest indices to 0)
                        # m_w = np.argmax(window)
                        # m_i = ((m_w >> 1) & 1, m_w & 1) # Only works if 2x2 (truth table)

                        # Downsample single pixel placement for output which is the highest value in window
                        Y_p[i, j, y, v] = window[m_i[0], m_i[1]]

                        # Mask placement for back-propagation
                        mask[i, j, y*s_p+m_i[0], v*s_p+m_i[1]] = True

        self.mask[:] = mask[:]
        self.Y_p[:]  = Y_p[:]
        print("Pooling Shape: ", self.Y_p.shape)
        print("Mask Shape: ", self.mask.shape)
        return self.Y_p

    def backwards_max_pooling(self, head_input):
        N, C_o, H, W = self.N, self.C_o, self.H_p, self.W_p
        s = self.s_p
        h_o, w_o = H//s, W//s

        for n in range(N):
            for c in range(C_o):
                for i in range(h_o):
                    for j in range(w_o):
                        r, co = i*s, j*s # Gets the input position respective to the output position
                        mask_m = self.mask[n, c, r:r+s, co:co+s] # Full size
                        #print(f"dx: {dx[n, c, r:r+d, co:co+d].shape}")
                        #print(f"do: {d_out[n,c,i,j].shape}")
                        #print(f"ma: {mask_m}")
                        # Keeps only activated position from the original input
                        # Takes the selected mask position and transfer it to upscaled matrices
                        self.Y_d[n, c, r:r+s, co:co+s] += head_input[n,c,i,j] * mask_m
                        # print("EVAL FRAME: ", mask_m / mask_m.sum())
        print("Shape Pool Back: ", self.Y_d.shape)
        print("Pool Back:\n", self.Y_d)
        return self.Y_d




class Head():
    def __init__(self, data, evaluation, learning_rate=0.1, bias=0.1):
        self.input = data
        self.input_d = np.zeros((data.shape))
        self.N, self.C_o, self.H, self.W = data.shape
        self.Y = evaluation
        self.K = len(evaluation) # Number of classes for evaluation
        self.lr = learning_rate # Learning rate

        # Global Average Pooling (GAP)
        self.G = np.zeros((self.N, self.C_o))
        self.G_d = np.zeros((self.N, self.C_o))

        # Linear Classifier Head (Logits) containers
        self.Z = np.zeros((self.N, self.K))
        self.Z_d = np.zeros((self.N, self.K))
        self.U = np.random.uniform(-1, 1, (self.K, self.C_o))
        self.U_d = self.U.copy()
        self.B = np.full(self.K, bias)
        self.B_d = self.B.copy()

        # Soft-max (probability normalization)
        self.P = np.zeros((self.N, self.K))

        # What the model predicts in correct
        self.prediction = np.zeros((self.N, self.K), dtype=bool)

        # Cross-Entropy Loss
        self.L:float

    # Global avg pool $F\in{\mathbb{D}^{(N,C,H,W)}}\to{G\in{\mathbb{R}^{(N,C)}}}$
    def forward_gap(self):
        self.G[:] = self.input.mean(axis=(2, 3))[:]
        print("GAP Shape: ", self.G.shape)

    def forward_logits(self):
        self.Z[:] = (self.G @ self.U.T + self.B)[:]
        print("Logits Shape: ", self.Z.shape)

    def forward_softmax(self):
        self.P[:] = np.array([[
            (np.exp(self.Z[n,k] - np.max(self.Z[n]))) /
            (np.sum(np.exp(self.Z[n,:] - np.max(self.Z[n]))))
            for k in range(self.K)]
            for n in range(self.N)])[:]
        print("Soft Shape: ", self.P.shape)
        print("\nClassification:\n", self.P)
        print("Evaluation:\n", self.Y)
        self.forward_prediction()

    def forward_prediction(self):
        self.prediction[:] = (self.P.argmax(axis=1) == self.Y.argmax(axis=1))[:, None]
        print("Correct Predictions:\n", self.prediction, "\n")

    def forward_loss(self):
        self.L = -np.mean(np.sum(self.Y * (np.log(self.P)+1e-12), axis=1))
        print("Loss: ", self.L)

    # Start of the back-propagation functions below: $\frac{dx}{dy}f(h(g(x)))=f'(h(g(x))*h(g(x)*g'(x)$
    # Calculates the derivative of the loss w.r.t logit scores
    def backward_loss_softmax(self):
        self.Z_d[:] = ((1/self.N) * (self.P - self.Y))[:]
        print("Shape: ", self.Z_d.shape)
        print("Loss Derivative w.r.t Logits:\n", self.Z_d, "\n")

    def backward_loss_weights(self):
        self.U_d[:] = (self.Z_d.T @ self.G)[:]
        print("Shape: ", self.U_d.shape)
        print("Loss Derivative w.r.t Weights:\n", self.U_d, "\n")
        print("Old Weights:\n", self.U, "\n")
        #self.U = self.U_d * self.lr # Updates weights

    def backward_loss_bias(self):
        self.B_d[:] = self.Z_d.sum(axis=0)[:]
        print("Loss Derivative w.r.t Bias:\n", self.B_d, "\n")
        print("Old Bias:\n", self.B, "\n")
        #self.B = self.B_d * self.lr

    def backward_loss_gap(self):
        self.G_d[:] = (self.Z_d @ self.U)[:]
        print("Loss Derivative w.r.t GAP:\n", self.G_d, "\n")
        print("Old GAP:\n", self.G, "\n")

    def backward_update_head(self):
        self.U -= self.lr * self.U_d
        self.B -= self.lr * self.B_d

    # Preps data structure to exit head and continue to push into layers back-propagation
    def backward_loss_exit(self):
        self.input_d[:] = self.G_d[:, :, None, None] / (self.H * self.W)
        print("Loss Derivative Shape w.r.t Head Input:\n", self.input_d.shape, "\n")


# Prints out a copy of the original input image and one feature map along the convolution process
def graph_output(data, raw, layers=1, forward_render=False):
    f_r = 1 if forward_render == 1 else 0
    fm_n = 0
    # Graph information
    fig, axes = plt.subplots(len(raw)*layers+f_r, len(data)+1, figsize=(15, 10), constrained_layout=True)
    print("Graphing Shape: ", axes.shape)
    for n in range(layers):
        for r in range(len(raw)):
            axes[r+(n*2),0].imshow(raw[r,0], cmap='gray') # Maps original dataset to first row
            axes[r+(n*2),0].axis('off')

            for c in range(len(data[0])):
                #print(f'PLACEMENT: ({r+(n*2)}, {c+1}) ' )
                axes[r+(n*2),c+1].imshow(data[n][c][r][fm_n], cmap='gray') # Maps kernel outputs to successive rows
                axes[r+(n*2),c+1].axis('off')
    plt.show()


def clock(marker:str):
    global past_time
    print(f'{marker}: {(time.perf_counter() - past_time):.2f}\n')
    past_time = time.perf_counter()


if __name__ == "__main__":
    rng = np.random.default_rng()
    #np.set_printoptions(suppress=True, precision=2)

    # Data extracting for directories to train CNN
    test_directory = Path("../data/test_data_256x256")
    test_target_directory = Path("../data/test_data_256x256/target")

    test_paths = sorted(test_directory.glob("*.png"))
    test_target_paths = sorted(test_target_directory.glob("*.png"))
    total_paths = [test_paths[0], test_target_paths[0]] # TODO: Change TO: test_paths + test_target_paths

    # Class information extracted from data
    k_0, k_1 = 1, 1 # TODO: Change TO: len(test_paths), len(test_target_paths) Images in class 0
    K = np.array([k_0, k_1]) # All class lengths

    # One-hot data evaluation metric
    evaluation = np.zeros((len(K), sum(K)))
    evaluation[0, 0:k_0] = 1 # Class 0 one-hot
    evaluation[1, k_0:] = 1  # Class 1 one-hot

    data_raw = np.array([[np.array(Image.open(p).convert("L"), dtype=np.float32) / 255] for p in total_paths])

    # //////////[LAYER VARIABLES]////////////////////////////////////////////////////////////////////////////////////////////////////

    # General convolution forward parameters
    # If kernel is even convolution feature map output increased by one on HxW
    kernel_size = 3
    output_increase = 2
    convolution_stride = 1
    padding_thickness = kernel_size//2
    testing = True

    # Activation function base parameters
    bias_base = 0.1
    activation = 'relu'

    # Polling base parameters
    pooling_stride = 2

    begin_time = time.perf_counter()
    past_time  = time.perf_counter()

    # //////////[LAYER & HEAD INITIALIZATION]////////////////////////////////////////////////////////////////////////////////////////////////////

    # Creates layer 0 of network
    cl_0 = Convolution(
        layer_number = 0,
        forward_data = data_raw,
        input_shape = data_raw.shape,
        kernel = kernel_size,
        output_multiple = output_increase,
        stride = convolution_stride,
        padding = padding_thickness,
        bias = bias_base,
        test = testing,
        rng_obj=rng)

    # Creates layer 0 activation function class
    al_0 = Activation(
        cl_0.Y.shape,
        activation)

    # Create layer 0 max pooling
    pl_0 = Pooling(
        stride=pooling_stride,
        input_dimensions=al_0.Y_a.shape)

    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////

    # Creates layer 1 of network
    cl_1 = Convolution(
        layer_number = 1,
        forward_data = pl_0.Y_p,
        input_shape = pl_0.Y_p.shape,
        kernel = kernel_size,
        output_multiple = output_increase,
        stride = convolution_stride,
        padding = padding_thickness,
        bias = bias_base,
        test = testing,
        rng_obj=rng)

    # Creates layer 1 activation function class
    al_1 = Activation(
        cl_1.Y.shape,
        activation)

    # Create layer 1 max pooling
    pl_1 = Pooling(
        stride=pooling_stride,
        input_dimensions=al_1.Y_a.shape)

    # //////////////////////////////////////////////////////////////////////////////////////////////////////////////

    # Creates layer 2 of network
    cl_2 = Convolution(
        layer_number = 2,
        forward_data = pl_1.Y_p,
        input_shape = pl_1.Y_p.shape,
        kernel = kernel_size,
        output_multiple = output_increase,
        stride = convolution_stride,
        padding = padding_thickness,
        bias = bias_base,
        test = testing,
        rng_obj=rng )

    # Creates layer 2 activation function class
    al_2 = Activation(
        cl_2.Y.shape,
        activation )

    # Create layer 2 max pooling
    pl_2 = Pooling(
        stride=pooling_stride,
        input_dimensions=al_2.Y_a.shape )

    head = Head(pl_2.Y_p, evaluation) # Creates class for processing head layer

    # ////////////[NETWORK PROCESSING]//////////////////////////////////////////////////////////////////////////////////////////////////

    print("\n[Layer 0]")
    cl_f_0 = cl_0.forward_convolution() # Convolution forward pass
    al_f_0 = al_0.forward_activation(cl_f_0) # Activation function forward pass
    pl_f_0 = pl_0.forward_max_pooling(al_f_0) # Max-pooling forward pass
    clock("L0 Convolution Time")

    print("\n[Layer 1]")
    cl_f_1 = cl_1.forward_convolution() # Convolution forward pass
    al_f_1 = al_1.forward_activation(cl_f_1) # Activation function forward pass
    pl_f_1 = pl_1.forward_max_pooling(al_f_1) # Max-pooling forward pass
    clock("L1 Convolution Time")

    print("\n[Layer 2]")
    cl_f_2 = cl_2.forward_convolution() # Convolution forward pass
    al_f_2 = al_2.forward_activation(cl_f_2) # Activation function forward pass
    pl_f_2 = pl_2.forward_max_pooling(al_f_2) # Max-pooling forward pass
    clock("L2 Convolution Time")

    print("[HEAD]")
    head.forward_gap()
    head.forward_logits()
    head.forward_softmax()
    head.forward_loss()
    clock("Head Time")

    print("[BACK-PROP HEAD]")
    head.backward_loss_softmax()
    head.backward_loss_weights()
    head.backward_loss_bias()
    head.backward_loss_gap()
    head.backward_update_head()
    head.backward_loss_exit()

    print("[BACK-PROP LAYERS]")
    pl_b_2 = pl_2.backwards_max_pooling(head.input_d)

    print("Total Time: ", time.perf_counter() - begin_time, "\n")



    # ////////////[VISUALIZATION]//////////////////////////////////////////////////////////////////////////////////////////////////

    to_graph = [[cl_f_0, al_f_0, pl_f_0],
                [cl_f_1, al_f_1, pl_f_1],
                [cl_f_2, al_f_2, pl_f_2],
                [pl_b_2, pl_b_2, pl_b_2]
    ]
    graph_output(data=to_graph, raw=data_raw, layers=1)
