module myANN

using Statistics: mean, std
using MLDatasets
using Flux 
using Flux: @epochs, onehotbatch, onecold, logitcrossentropy, train!, throttle, flatten
using Images
using Flux.Data: DataLoader

export get_data, split_train_test, standardize, process_images

include("load_data.jl")
include("preprocessing_data.jl")


end
