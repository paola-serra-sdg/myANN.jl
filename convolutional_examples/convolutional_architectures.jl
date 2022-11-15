using MLDatasets
using BenchmarkTools
using Flux
using Flux: @epochs, onehotbatch, onecold, logitcrossentropy, train!, throttle, flatten
using DataFrames
using Plots
using Images
using Lathe.preprocess: TrainTestSplit
using MLDataUtils
using IterTools

# image_dog = readdir(raw"C:\\Users\\garavagliam\\Desktop\\Repo_SDG\\myANN.jl\\data\\images\\dog")
# image_cat = readdir(raw"C:\\Users\\garavagliam\\Desktop\\Repo_SDG\\myANN.jl\\data\\images\\cat")

function resize_and_grayify(directory, im_name, width::Int64, height::Int64)
    resized_gray_img = Gray.(load(directory * "/" * im_name)) |> (x -> imresize(x, width, height))
    try
        save("preprocessed_" * directory * "/" * im_name, resized_gray_img)
    catch e
        if isa(e, SystemError)
            mkdir("preprocessed_" * directory)
            save("preprocessed_" * directory * "/" * im_name, resized_gray_img)
        end
    end
end

function process_images(directory, width::Int64, height::Int64)
    files_list = readdir(directory)
    map(x -> resize_and_grayify(directory, x, width, height), files_list)
end

n_resolution = 90

begin
    process_images("data/bee1", n_resolution, n_resolution)
    process_images("data/bee2", n_resolution, n_resolution)
    process_images("data/wasp1", n_resolution, n_resolution)
    process_images("data/wasp2", n_resolution, n_resolution)
end

begin
    bee1_dir = readdir("preprocessed_data/bee1")
    bee2_dir = readdir("preprocessed_data/bee2")
    wasp1_dir = readdir("preprocessed_data/wasp1")
    wasp2_dir = readdir("preprocessed_data/wasp2")
end

begin
    # we load the pre-proccessed images
    bees1 = load.("preprocessed_data/bee1/" .* bee1_dir)
    bees2 = load.("preprocessed_data/bee2/" .* bee2_dir)
    wasp1 = load.("preprocessed_data/wasp1/" .* wasp1_dir)
    wasp2 = load.("preprocessed_data/wasp2/" .* wasp2_dir)
end

bees = vcat(bees1, bees2)

images=[]

wasps = vcat(wasp1, wasp2)

data = vcat(bees, wasps)

begin
    labels = vcat([0 for _ in 1:length(bees)], [1 for _ in 1:length(wasps)])
    (x_train, y_train), (x_test, y_test) = splitobs(shuffleobs((data, labels)), at = 0.7)
end

function make_minibatch(X, Y, idxs)
    X_batch = Array{Float32}(undef, size(X[1])..., 1, length(idxs))
    for i in eachindex(idxs)
        X_batch[:, :, :, i] = Float32.(X[idxs[i]])
    end
    Y_batch = onehotbatch(Y[idxs], 0:1)
    return (X_batch, Y_batch)
end

begin
    # here we define the train and test sets.
    batchsize = 128
    mb_idxs = partition(1:length(x_train), batchsize)
    train_set = [make_minibatch(x_train, y_train, i) for i in mb_idxs]
    test_set = make_minibatch(x_test, y_test, 1:length(x_test))
end

model = Chain(
        Conv((3, 3), 1=>32, pad=(1,1), relu),
        MaxPool((2,2)),
        Conv((3, 3), 32=>64, pad=(1,1), relu),
        MaxPool((2,2)),
        Conv((3, 3), 64=>128, pad=(1,1), relu),
        MaxPool((2,2)),
        flatten,
        Dense(15488, 2),
        softmax)

begin
    train_loss = Float64[]
    test_loss = Float64[]
    acc = Float64[]
    ps = Flux.params(model)
    opt = ADAM()
    L(x, y) = Flux.crossentropy(model(x), y)
    L((x,y)) = Flux.crossentropy(model(x), y)
    accuracy(x, y, f) = mean(Flux.onecold(f(x)) .== Flux.onecold(y))
            
    function update_loss!()
        push!(train_loss, mean(L.(train_set)))
        push!(test_loss, mean(L(test_set)))
        push!(acc, accuracy(test_set..., model))
        @printf("train loss = %.2f, test loss = %.2f, accuracy = %.2f\n", train_loss[end], test_loss[end], acc[end])
    end
end

@epochs 10 Flux.train!(L, ps, train_set, opt;
               cb = Flux.throttle(update_loss!, 8))

begin
    plot(train_loss, xlabel="Iterations", title="Model Training", label="Train loss", lw=2, alpha=0.9)
    plot!(test_loss, label="Test loss", lw=2, alpha=0.9)
    plot!(acc, label="Accuracy", lw=2, alpha=0.9)
end