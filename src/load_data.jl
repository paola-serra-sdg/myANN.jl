using Flux: onehotbatch, OneHotMatrix
using Statistics: mean, std


function get_data(path::String)
    dirs = readdir(path; join=true)
    images = []
    labels = []
    i = 0

    for dir in dirs
        els = readdir(dir; join=true)
        imgs = load.(els)
        labs = repeat([i], length(els))
        i = i+1
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
    cold_labels = cat(labels..., dims=1)

    # I need a onehot vector to pass the labels to the model
    labels = onehotbatch(cold_labels, 0:(length(dirs)-1))

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
function split_train_test(images::Array, labels::OneHotMatrix, ratio=0.7)
    ind = trunc(Int, ratio*(size(images)[4]))
    train_x = images[:,:,:,1:ind]
    train_y = labels[:,1:ind]
    test_x = images[:,:,:,ind:(size(images)[4])]
    test_y = labels[:,ind:(size(labels)[2])]
    return train_x, train_y, test_x, test_y
end