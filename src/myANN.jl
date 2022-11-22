module myANN

using Statistics: mean, std
using MLDatasets
using Flux 
using Flux: @epochs, onehotbatch, onecold, logitcrossentropy, train!, throttle, flatten
using Images
using Flux.Data: DataLoader

export get_data, split_train_test, standardize

include("load_data.jl")


end
