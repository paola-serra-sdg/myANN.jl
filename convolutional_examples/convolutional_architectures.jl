using MLDatasets
using Flux 
using Flux: @epochs, onehotbatch, onecold, logitcrossentropy, train!, throttle, flatten
using Statistics: mean, std
using Images
using ImageMagick


function get_data(path::String)
    dirs = readdir(path; join=true)
    images = []
    labels = []

    for dir in dirs
        els = readdir(dir; join=true)
        imgs = load.(els)
        if occursin("bee", dir)
            labs = zeros(length(els))
        else 
            labs = ones(length(els))
        end
        imgs = cat(imgs..., dims=3)
        push!(labels,labs)
        push!(images,imgs)
    end
    
    # Put all images in a vector and add channel dimension
    images = Flux.unsqueeze(cat(images..., dims=3), 3)
    
    # Standardize images
    images = standardize(images)

    # All labels in a vector
    labels = cat(labels..., dims=1)

    return images, labels

end


function standardize(images::Array)
    Float64.(images)
    m = mean(images, dims=(1,2))
    s = std(images, dims=(1,2))
    st_imgs = (images.-m)/s
    return st_imgs
end

function split_train_test(images::Array, labels::Array, ratio=0.7)
    ind = Int(ratio*length(images))
    train_x = images[:,:,:,1:ind]
    train_y = labels[1:ind]
    test_x = images[:,:,:,ind:length(images)]
    test_y = labels[ind:length(labels)]
    return train_x, train_y, test_x, test_y
end


images, labels = get_data("preprocessed_data")

x_train, y_train, x_test, y_test = split_train_test(images, labels)

train_loader = DataLoader((x_train, y_train))


model = Chain(
        Conv((3, 3), 1=>32, pad=(1,1), relu),
        MaxPool((2,2)),
        Conv((3, 3), 32=>64, pad=(1,1), relu),
        MaxPool((2,2)),
        Conv((3, 3), 64=>128, pad=(1,1), relu),
        MaxPool((2,2)),
        flatten,
        Dense(1152, 2),
        softmax)


params = Flux.params(model)

optimiser = ADAM()
loss(x,y) = logitcrossentropy(model(x), y)

evalcb = () -> @show(loss(x_train, y_train))

num_epochs = 10
@epochs num_epochs train!(loss, params, train_loader, optimiser, cb = throttle(evalcb, 5))