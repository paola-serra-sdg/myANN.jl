using Flux: @epochs, onehotbatch, onecold, logitcrossentropy, train!, throttle, flatten, onehotbatch
using Statistics: mean, std
using Images
using Flux.Data: DataLoader
using myANN

# COME FACCIO A FARE IL MODULE????


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

evalcb = () -> @show(loss(x_train, y_train))

num_epochs = 5
@epochs num_epochs train!(loss, params, data, optimiser, cb = throttle(evalcb, 5))