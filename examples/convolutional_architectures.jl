using Flux 
using Flux: @epochs, onehotbatch, onecold, logitcrossentropy, train!, throttle, flatten
using Statistics: mean, std
using Images
using Flux.Data: DataLoader
using myANN
using Plots

# Get our data
images, labels = get_data("preprocessed_data");

# Train and test splitting
x_train, y_train, x_test, y_test = split_train_test(images, labels);

# Loading
train_data = DataLoader((x_train, y_train); batchsize = 32, shuffle = true);
test_data = DataLoader((x_test, y_test); batchsize = 32, shuffle = true);

# Convolutional architecture
model = Chain(
        Conv((3, 3), 1=>8, pad=(1,1), relu),
        MaxPool((2,2)),
        flatten,
        Dense(1800, 3))

# Save our model on CPU
model = cpu(model)

params = Flux.params(model)

optimiser = ADAM()
loss(x,y) = logitcrossentropy(model(x), y)

# Training and plotting
epochs = Int64[]
loss_on_train = Float64[]
loss_on_test = Float64[]
acc = Float64[]
best_params = Float32[]

for epoch in 1:20
    Flux.train!(loss, params, train_data, optimiser)
    push!(epochs, epoch)
    push!(loss_on_train, loss(x_train, y_train))
    push!(loss_on_test, loss(x_test, y_test))
    push!(acc, accuracy(y_test, model(x_test)))
    @show loss(x_train, y_train)
    @show loss(x_test, y_test)
    if epoch > 1
        if is_best(loss_on_test[epoch-1], loss_on_test[epoch])
            best_params = params
        end
    end
end

# Extract and add new trained parameters
if isempty(best_params)
    best_params = params
end

Flux.loadparams!(model, best_params);

# Visualization
plot(epochs, loss_on_train, lab="Training", c=:black, lw=2);
plot!(epochs, loss_on_test, lab="Test", c=:green, lw=2);
title!("Convolutional architecture");
yaxis!("Loss");
xaxis!("Training epoch");
savefig("conv_loss");

plot(epochs, acc, lab="Accuracy", c=:green, lw=2, ylims = (0,1));
title!("Convolutional architecture");
yaxis!("Accuracy");
xaxis!("Training epoch");
savefig("conv_accuracy");
