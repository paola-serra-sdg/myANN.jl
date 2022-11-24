using Flux
using Flux: @epochs, onehotbatch, onecold, logitcrossentropy, train!, throttle, flatten, onehotbatch, loadmodel!
using Statistics: mean, std
using Images
using Flux.Data: DataLoader
using myANN
using Plots
using BSON: @save, @load

images, labels, labs = get_data("preprocessed_data");

x_train, y_train, x_test, y_test = split_train_test(images, labels);

train_data = DataLoader((x_train, y_train));
test_data = DataLoader((x_test, y_test));


model = Chain(
        flatten,
        Dense(3600, 128, relu),
        Dense(128, 64, relu),
        Dense(64, 32, relu),
        Dense(32, 2)
)

model = cpu(model)

params = Flux.params(model)

optimiser = ADAM(0.01)
loss(x,y) = logitcrossentropy(model(x), y)

# evalcb = throttle(30) do 
#     @save "model_checkpoint.bson" model
# end;

# num_epochs = 5
# @epochs num_epochs train!(loss, params, data, optimiser, cb = throttle(evalcb, 5))

function accuracy(y_true::Any, y_pred::Any)
    s = 0
    y_true = onecold(y_true, 0:1)
    y_pred = onecold(y_pred, 0:1)
    for i in size(y_true)
        if y_true[i] == y_pred[i]
            s = s+1
        end
    end
    return s/(size(y_true)[1])
end


epochs = Int64[]
loss_on_train = Float32[]
acc = Float32[]



for epoch in 1:5
    Flux.train!(loss, params, train_data, optimiser, cb = evalcb)
    # we record our training loss
    push!(epochs, epoch)
    push!(loss_on_train, loss(x_train, y_train))
    push!(acc, accuracy(y_test, model(x_test)))
end


plot(epochs, loss_on_train, lab="Training", c=:black, lw=2);
title!("Dense architecture");
yaxis!("Loss", :log);
xaxis!("Training epoch")

plot(epochs, acc, lab="Training", c=:black, lw=2);
title!("Dense architecture");
yaxis!("Accuracy", :log);
xaxis!("Training epoch")