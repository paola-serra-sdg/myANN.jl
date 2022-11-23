using Flux: @epochs, onehotbatch, onecold, logitcrossentropy, train!, throttle, flatten, onehotbatch
using Statistics: mean, std
using Images
using Flux.Data: DataLoader
using myANN
using MLJBase


images, labels = get_data("preprocessed_data");

x_train, y_train, x_test, y_test = split_train_test(images, labels);

data = DataLoader((x_train, y_train));

# Perchè va anche con la dimensione dei canali?
model = Chain(
        flatten,
        Dense(3600, 128, relu),
        Dense(128, 64, relu),
        Dense(64, 32, relu),
        Dense(32, 2)
)

params = Flux.params(model)

optimiser = ADAM()
loss(x,y) = logitcrossentropy(model(x), y)

#evalcb = () -> @show(loss(x_train, y_train))

# num_epochs = 5
# @epochs num_epochs train!(loss, params, data, optimiser, cb = throttle(evalcb, 5))



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
    push!(accuracy_train, MLJBase.accuracy(y_train, x_train))
    push!(accuracy_test, MLJBase.accuracy(y_test, x_test))
end

plot(epochs, loss_on_train, lab="Training", c=:black, lw=2);
plot!(epochs, loss_on_test, lab="Testing", c=:teal, ls=:dot);
yaxis!("Loss", :log);
xaxis!("Training epoch")


plot(epochs, accuracy_train, lab="Training", c=:black, lw=2);
plot!(epochs, accuracy_tets, lab="Testing", c=:teal, ls=:dot);
yaxis!("Accuracy", :log);
xaxis!("Training epoch")
