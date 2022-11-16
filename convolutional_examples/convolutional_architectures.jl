using MLDatasets
using Flux 
using Flux: @epochs, onehotbatch, onecold, logitcrossentropy, train!, throttle, flatten
using Statistics: mean, std


function get_data(path)
    dirs= readdir(path; join = true)
    images =[]
    labels=[]

    for dir in dirs
        els= readdir(dir; join= true)
        imgs = load.(els)
        if  occursin("bee",dir)
            labs= zeros(length(els)) 
        else 
            labs= ones(length(els))
        end
        push!(labels,labs)
        push!(images,imgs)
    end
    
    
    images= Flux.unsqueeze(cat(images..., dims= 3), 3)
    images= standardize(images)
    labels=cat(labels..., dims= 1)
    
    return images, labels
end


function standardize(images)
    m= mean(images,dims= (1,2))
    s= std(images,dims=(1,2))
    st_imgs= (images.-m)/s

end

function split_traintest(images,labels,ratio=0.7)
    ind = Int(ratio*length(images))
    train_x = images[:,:,:,1:ind]
    train_y = labels[1:ind]
    test_x = images[:,:,:,ind:length(images)]
    test_y = labels[ind:length(labels)]
end






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