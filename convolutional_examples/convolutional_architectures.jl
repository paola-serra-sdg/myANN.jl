using MLDatasets
using BenchmarkTools
using Flux
using Flux: @epochs, onehotbatch, onecold, logitcrossentropy, train!, throttle, flatten


train_x, train_y = MLDatasets.MNIST.traindata(Float32)
test_x, test_y = MLDatasets.MNIST.testdata(Float32)

train_y, test_y = onehotbatch(train_y, 0:9), onehotbatch(test_y, 0:9)

train_data = [(train_x, train_y)]
test_data = [(test_x, test_y)]


# We will use a simple convolutional architecture withthree iterations of Conv -> ReLU -> MaxPool, 
# followed by a final Dense layer that feeds into a softmax probability output.
model = Chain(
    # First convolution, operating upon a 28x28 image
    Conv((3, 3), 1=>16, pad=(1,1), relu),
    x -> maxpool(x, (2,2)),

    # Second convolution, operating upon a 14x14 image
    Conv((3, 3), 16=>32, pad=(1,1), relu),
    x -> maxpool(x, (2,2)),

    # Third convolution, operating upon a 7x7 image
    Conv((3, 3), 32=>32, pad=(1,1), relu),
    x -> maxpool(x, (2,2)),

    # Reshape 3d tensor into a 2d one, at this point it should be (3, 3, 32, N)
    # which is where we get the 288 in the `Dense` layer below:
    x -> reshape(x, :, size(x, 4)),
    Dense(288, 10),

    # Finally, softmax to get nice probabilities
    softmax,
)
params = Flux.params(model)

optimiser = ADAM()
loss(x,y) = logitcrossentropy(model(x), y)

evalcb = () -> @show(loss(train_x, train_y))

num_epochs = 10
@epochs num_epochs train!(loss, params, train_data, optimiser, cb = throttle(evalcb, 5))