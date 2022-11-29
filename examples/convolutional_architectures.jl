using Flux 
using Flux: @epochs, onehotbatch, onecold, logitcrossentropy, train!, throttle, flatten
using Statistics: mean, std
using Images
using Flux.Data: DataLoader
using myANN
using Plots


images, labels = get_data("preprocessed_data");

x_train, y_train, x_test, y_test = split_train_test(images, labels);

train_data = DataLoader((x_train, y_train); batchsize = 32, shuffle = true);
test_data = DataLoader((x_test, y_test); batchsize = 32, shuffle = true);

model = Chain(
        Conv((3, 3), 1=>32, pad=(1,1), relu),
        MaxPool((2,2)),
        flatten,
        Dense(7200, 3),
        softmax)

model = cpu(model)

params = Flux.params(model)

optimiser = ADAM()
loss(x,y) = logitcrossentropy(model(x), y)

# Training and plotting
epochs = Int64[]
loss_on_train = Float64[]
loss_on_test = Float64[]
acc = Float64[]

for epoch in 1:5
    Flux.train!(loss, params, train_data, optimiser)
end

w = Flux.params(model);
Flux.loadparams!(model, w);

for epoch in 1:5
    # we record our training loss
    push!(epochs, epoch)
    push!(loss_on_train, loss(x_train, y_train))
    push!(loss_on_test, loss(x_test, y_test))
    push!(acc, accuracy(y_test, model(x_test)))
end


plot(epochs, loss_on_train, lab="Training", c=:black, lw=2);
plot!(epochs, loss_on_test, lab="Test", c=:green, lw=2);
title!("Dense architecture");
yaxis!("Loss", :log);
xaxis!("Training epoch");
savefig("conv_loss");

plot(epochs, acc, lab="Training", c=:black, lw=2, ylims = (0,1));
title!("Dense architecture");
yaxis!("Accuracy", :log);
xaxis!("Training epoch");
savefig("conv_accuracy");


