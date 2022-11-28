using Flux
using Flux: @epochs, onehotbatch, onecold, logitcrossentropy, train!, throttle, flatten, loadmodel!
using Statistics: mean, std
using Images
using Flux.Data: DataLoader
using myANN
using Plots


images, labels, cold_labels = get_data("preprocessed_data");

x_train, y_train, x_test, y_test = split_train_test(images, labels);

train_data = DataLoader((x_train, y_train));
test_data = DataLoader((x_test, y_test));


model = Chain(
        flatten,
        Dense(400, 128, relu),
        Dense(128, 64, relu),
        Dense(64, 32, relu),
        Dense(32, 2),
        softmax
)

model = cpu(model)

params = Flux.params(model)

optimiser = ADAM(0.01)
loss(x,y) = logitcrossentropy(model(x), y)

# Training and plotting
epochs = Int64[]
loss_on_train = Float64[]
loss_on_test = Float64[]
acc = Float64[]

for epoch in 1:20
    Flux.train!(loss, params, train_data, optimiser)
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
xaxis!("Training epoch")

plot(epochs, acc, lab="Training", c=:black, lw=2, ylims = (0,1));
title!("Dense architecture");
yaxis!("Accuracy", :log);
xaxis!("Training epoch")