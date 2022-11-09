# Necessary packages
using Flux
using Statistics
using MLDatasets
using BenchmarkTools
using Flux: @epochs, onehotbatch, onecold, logitcrossentropy, train!, throttle, flatten

# Convert your data to Float32
# ibm_data = Float32.(ibm_data)

# # Reshape your data to the general RNN input format in Flux
# # Note that this assumes a sequence length of 5000, i.e. the full sequence, which is not necessarily a good idea
# X = [[x] for x in ibm_data[1:end-1]]
# y = ibm_data[2:end]


# train_x, train_y = MLDatasets.MNIST.traindata(Float32)
# test_x, test_y = MLDatasets.MNIST.testdata(Float32)

# train_y, test_y = onehotbatch(train_y, 0:9), onehotbatch(test_y, 0:9)

# train_data = [(train_x, train_y)]
# test_data = [(test_x, test_y)]




# Create the RNN model
myrnn = Chain(RNN(1, 32), Dense(32, 1))

# Choose an optimizer
opt = ADAM(1e-2)

# Keep track of parameters for update
ps = Flux.params(myrnn)

# Define a loss function
function loss(X, y)
    myrnn(X[1]) # Warm up model
    # Compute loss
    mean(abs2(myrnn(x)[1] - y) for (x, y) in zip(X[2:end], y[2:end]))
end
for epoch in 1:10 # Train the RNN for 10 epochs
    Flux.reset!(myrnn) # Reset RNN
    gs = gradient(ps) do # Compute gradients
        loss(X, y)        
    end
    Flux.update!(opt, ps, gs) # Update parameters
end