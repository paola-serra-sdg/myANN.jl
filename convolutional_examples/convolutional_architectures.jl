using MLDatasets
using BenchmarkTools
using Flux
using Flux: @epochs, onehotbatch, onecold, logitcrossentropy, train!, throttle, flatten
using DataFrames
using Plots
using Images
using Lathe.preprocess: TrainTestSplit
using MLDataUtils

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
    process_images("data/wasp2", n_resolution, n_resolution);
end

begin
    bee1_dir = readdir("preprocessed_10_bees_vs_wasps/data/bee1")
    bee2_dir = readdir("preprocessed_10_bees_vs_wasps/data/bee2")
    wasp1_dir = readdir("preprocessed_10_bees_vs_wasps/data/wasp1")
    wasp2_dir = readdir("preprocessed_10_bees_vs_wasps/data/wasp2");
end;

# begin
#     # we load the pre-proccessed images
#     dog = load.(raw"C:\\Users\\garavagliam\\Desktop\\Repo_SDG\\myANN.jl\\data\\preprocess_data\\dog" .* dog_dir)
#     cat = load.(raw"C:\\Users\\garavagliam\\Desktop\\Repo_SDG\\myANN.jl\\data\\preprocess_data\\cat" .* cat_dir)
# end;

# data = vcat(dog, cat)

# images=[]

# for i in image_dog
#     append!(images,1)
# end
# for i in image_cat
#     append!(images,0)
# end

# df2 = DataFrame([image_cat],[:x1]) 
# df1 = DataFrame([image_dog], [:x1])
# dff = vcat(df1, df2)
# di = DataFrame([images], [:x2])

# df = hcat(dff,di)


# # img= load("C:\\Users\\garavagliam\\Desktop\\Repo_SDG\\myANN.jl\\data\\images\\dog_0002.jpg")
# # plot(img)
# # size(img)
# #labels= ["dog","cat"]

# begin
#     labels = vcat([0 for _ in 1:length(dog)], [1 for _ in 1:length(cat)])
#     (x_train, y_train), (x_test, y_test) = splitobs(shuffleobs((data, labels)), at = 0.7)
# end;

# (x_train, y_train), (x_test, y_test) = splitobs(shuffleobs((dff, di)), at = 0.7)
# #train, test = TrainTestSplit(df,0.7)

# # minibatch
# function make_minibatch(X, Y, idxs)
#     X_batch = Array{Float32}(undef, size(X[1])..., 1, length(idxs))
#     for i in 1:length(idxs)
#         X_batch[:, :, :, i] = Float32.(X[idxs[i]])
#     end
#     Y_batch = onehotbatch(Y[idxs], 0:1)
#     return (X_batch, Y_batch)
# end

# begin
#     # here we define the train and test sets.
#     batchsize = 128
#     mb_idxs = partition(1:length(x_train), batchsize)
#     train_set = [make_minibatch(x_train, y_train, i) for i in mb_idxs]
#     test_set = make_minibatch(x_test, y_test, 1:length(x_test))
# end

# # We will use a simple convolutional architecture withthree iterations of Conv -> ReLU -> MaxPool, 
# # followed by a final Dense layer that feeds into a softmax probability output.
# model = Chain(
#     # First convolution, operating upon a 28x28 image
#     Conv((3, 3), 1=>32, pad=(1,1), relu),
#     MaxPool((2,2)),

#     # Second convolution, operating upon a 14x14 image
#     Conv((3, 3), 32=>64, pad=(1,1), relu),
#     MaxPool((2,2)),

#     # Third convolution, operating upon a 7x7 image
#     Conv((3, 3), 64=>128, pad=(1,1), relu),
#     MaxPool((2,2)),

#     # Reshape 3d tensor into a 2d one, at this point it should be (3, 3, 32, N)
#     # which is where we get the 288 in the `Dense` layer below:
#     flatten,
#     Dense(1190, 2),

#     # Finally, softmax to get nice probabilities
#     softmax)

# begin
#     train_loss = Float64[]
#     test_loss = Float64[]
#     acc = Float64[]
#     ps = Flux.params(model)
#     opt = ADAM()
#     L(x, y) = Flux.crossentropy(model(x), y)
#     L((x,y)) = Flux.crossentropy(model(x), y)
#     accuracy(x, y, f) = mean(Flux.onecold(f(x)) .== Flux.onecold(y))
    
#     function update_loss!()
#         push!(train_loss, mean(L.(train_set)))
#         push!(test_loss, mean(L(test_set)))
#         push!(acc, accuracy(test_set..., model))
#         @printf("train loss = %.2f, test loss = %.2f, accuracy = %.2f\n", train_loss[end], test_loss[end], acc[end])
#     end
# end

# @epochs 10 Flux.train!(L, ps, train_set, opt;
#                cb = Flux.throttle(update_loss!, 8))

# # begin
# #     plot(train_loss, xlabel="Iterations", title="Model Training", label="Train loss", lw=2, alpha=0.9)
# #     plot!(test_loss, label="Test loss", lw=2, alpha=0.9)
# #     plot!(acc, label="Accuracy", lw=2, alpha=0.9)
# # end