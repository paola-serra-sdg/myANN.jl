using Flux: onehotbatch
using Statistics: mean, std

# Take as input the path in which the data are stored and create a vector of images of size (length, width, n_channels, n_images)
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
    
    # From graytype vector to Float64 vector
    images = real.(images)

    # Standardize images
    images = standardize(images)

    # All labels in a vector
    labels = cat(labels..., dims=1)

    # I need a onehot vector to pass the labels to the model
    labels = onehotbatch(labels, 0:1)

    return images, labels

end

# Standardize my data
function standardize(images::Array)
    m = mean(images, dims=(1,2))
    s = std(images, dims=(1,2))    
    st_imgs = (images.-m)./s
    return st_imgs
end

# Split the data in train and test set
function split_train_test(images::Array, labels::Any, ratio=0.7)
    ind = trunc(Int, ratio*(size(images)[4]))
    train_x = images[:,:,:,1:ind]
    train_y = labels[:,1:ind]
    test_x = images[:,:,:,ind:(size(images)[4])]
    test_y = labels[:,ind:(size(labels)[2])]
    return train_x, train_y, test_x, test_y
end