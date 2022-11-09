# Convolutional architecture for ANN with MNIST

Type of layer:
- **Conv layer** *Conv(filter, in => out, σ = identity; stride = 1, pad = 0, dilation = 1, groups = 1, [bias, init])* :
"Filter" is a tuple of integers specifying the size of the convolutional kernel; in and out specify the number of input and output channels.
Image data should be stored in WHCN order (width, height, channels, batch).

- **Maxpool layer** *MaxPool(window::NTuple; pad=0, stride=window)*: replaces all pixels in a block of size window with one. Expects as input an array with *ndims(x) == N+2*, i.e. channel and batch dimensions, after the N feature dimensions, where *N = length(window)*. By default the window size is also the stride in each dimension. The keyword pad accepts the same options as for the Conv layer, including *SamePad()*.

- **Softmax** *(softmax(x; dims = 1))* : turns input array x into probability distributions that sum to 1 along the dimensions specified by dims:
*softmax(x; dims = 1) = exp.(x) ./ sum(exp.(x), dims = dims)*

# Detailed explanation:
**Layer 1**
*Conv((3, 3), 1=>16, pad=(1,1), relu)*

The first layer can be broken down as follows:

(3,3) is the convolution filter size (3x3) that will slide over the image detecting new features.

1=>16 is the network input and output size. The input size is 1 recalling that one batch is of size 28x28x1x128. The output size is 16 meaning we’ll create 16 new channels for every training digit in the batch.

pad=(1,1) This pads a single layer of zeros around the images meaning that the dimensions of the convolution output can remain at 28x28.

relu is our activation function.

The output from this layer only can be viewed with model[1](train_set[1][1]) and has the dimensions 28×28×16×128.

**Layer 2**
*x -> maxpool(x, (2,2))*

Convolutional layers are generally followed by a maxpool layer. In our case the parameter (2,2) is the window size that slides over x reducing it to half the size whilst retaining the most important feature information for learning.

The output from this layer only can be viewed with model[1:2](train_set[1][1]) and has the output dimensions 14×14×16×128.

**Layer 3**
*Conv((3, 3), 16=>32, pad=(1,1), relu)*

This is the second convolution operating on the output from layer 2.

Conv((3, 3), is the same filter size as before.

16=>32 This time the input is 16 (from layer 2). The output size of the layer will be 32.

The padding, filter size and activation remains the same as before.

The output from this layer only can be viewed with model[1:3](train_set[1][1]) and has the output dimensions 14×14×32×128.

**Layer 4**
*x -> maxpool(x, (2,2))*

Maxpool reduces the dimensionality in half again whilst retaining the most important feature information for learning.

The output from this layer only can be viewed with model[1:4](train_set[1][1]) and has the output dimensions 7×7×32×128.

**Layers 5 & 6**
*Conv((3, 3), 32=>32, pad=(1,1), relu)*
*x -> maxpool(x, (2,2))*

Perform a final convolution and maxpool. The output from layer 6 is 3×3×32×128

**Layer 7**
*x -> reshape(x, :, size(x, 4))*

The reshape layer effectively flattens the data from 4-dimensions to 2-dimensions suitable for the dense layer and training.

The output from this layer only can be viewed with model[1:7](train_set[1][1]) and has the output dimensions 288×128. If you’re wondering where 288 comes from, it is determined by multiplying the output of layer 6; i.e. 3x3x32.

**Layer 8**
*Dense(288, 10)*

Our final training layer takes the input of 288 and outputs a size of 10x128.

(10 for 10 digits 0-9)

**Layer 9**
*softmax*

Outputs probabilities between 0 and 1 of which digit the model has predicted.