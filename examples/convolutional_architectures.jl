using MLDatasets
using Flux 
using Flux: @epochs, onehotbatch, onecold, logitcrossentropy, train!, throttle, flatten
using Statistics: mean, std
using Images
using ImageMagick
using Flux.Data: DataLoader
using Plots


images, labels = get_data("preprocessed_data");

x_train, y_train, x_test, y_test = split_train_test(images, labels);

train_data = DataLoader((x_train, y_train); batchsize = 32);
test_data = DataLoader((x_test, y_test); batchsize = 32);

model = Chain(
        Conv((3, 3), 1=>32, pad=(1,1), relu),
        MaxPool((2,2)),
        Conv((3, 3), 32=>64, pad=(1,1), relu),
        MaxPool((2,2)),
        Conv((3, 3), 64=>128, pad=(1,1), relu),
        MaxPool((2,2)),
        flatten,
        Dense(6272, 2),
        softmax)

model = cpu(model)

params = Flux.params(model)

optimiser = ADAM()
loss(x,y) = logitcrossentropy(model(x), y)

#evalcb = () -> @show(loss(x_train, y_train))

num_epochs = 5
# @epochs num_epochs train!(loss, params, train_data, optimiser) #, cb = throttle(evalcb, 1000))
# @epochs num_epochs train!(loss, params, test_data, optimiser)


epochs = Int64[]
loss_on_train = Float32[]
loss_on_test = Float32[]
accuracy_train = Float32[]
accuracy_test = Float32[]

for epoch in 1:5
    Flux.train!(loss, params, test_data, optimiser)
    # we record our training loss
    push!(epochs, epoch)
    push!(loss_on_test, loss(x_test, y_test))
    push!(loss_on_train, loss(x_train, y_train))
    push!(accuracy_train, MLJBase.accuracy(x_train, y_train))
    push!(accuracy_test, MLJBase.accuracy(x_test, y_test))
end

plot(epochs, loss_on_train, lab="Training", c=:black, lw=2);
plot!(epochs, loss_on_test, lab="Testing", c=:teal, ls=:dot);
yaxis!("Loss", :log);
xaxis!("Training epoch")


plot(epochs, accuracy_train, lab="Training", c=:black, lw=2);
plot!(epochs, accuracy_tets, lab="Testing", c=:teal, ls=:dot);
yaxis!("Accuracy", :log);
xaxis!("Training epoch")


