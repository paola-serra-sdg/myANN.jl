module myANN

using Statistics: mean, std
using MLDatasets
using Flux 
using Flux: @epochs, onehotbatch, onecold, logitcrossentropy, train!, throttle, flatten, params, loadparams!
using Images
using Flux.Data: DataLoader

export get_data, split_train_test, standardize, process_images, accuracy, is_best

include("load_data.jl")
include("metrics.jl")
include("preprocessing_data.jl")

end
