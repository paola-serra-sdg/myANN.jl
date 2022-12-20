using Flux
using Flux: @epochs, onehotbatch, onecold, logitcrossentropy, train!, throttle, flatten
using Statistics: mean, std
using Images
using Flux.Data: DataLoader
using myANN
using Plots
using Random
using ParametricMachinesDemos

# Get our data
images, labels = get_data("preprocessed_data");

# Train and test splitting
x_train, y_train, x_test, y_test = split_train_test(images, labels);

# Loading
train_data = DataLoader((x_train, y_train); batchsize = 32, shuffle = true);
test_data = DataLoader((x_test, y_test); batchsize = 32, shuffle = true);

# Dimensions represent the channel in input and output of each layer in the machine
dimensions = [4,4,4,4];

# Define the parametric machine
machine = ConvMachine(dimensions, sigmoid; pad=(0,0,0,0))

# Model
model = Flux.Chain(machine, flatten, Dense(14400,3)) |> f64 
model = cpu(model)

# Parameters
params = Flux.params(model)

optimiser = ADAM(0.01)

loss(x,y) = logitcrossentropy(model(x), y)

# Training and plotting
epochs = Int64[]
loss_on_train = Float64[]
loss_on_test = Float64[]
acc = Float64[]
best_params = Float32[]

for epoch in 1:10

    # Train
    Flux.train!(loss, params, train_data, optimiser)

    # Show the sum of the gradients
    gs = gradient(params) do
        loss(x_train, y_train)
    end
    @show sum(first(gs))
    
    # Saving losses and accuracies for visualization
    push!(epochs, epoch)
    push!(loss_on_train, loss(x_train, y_train))
    push!(loss_on_test, loss(x_test, y_test))
    push!(acc, accuracy(y_test, model(x_test)))
    @show loss(x_train, y_train)
    @show loss(x_test, y_test)

    # Saving the best parameters
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
plot(epochs, loss_on_train, lab="Training", c=:black, lw=2, ylims = (0,2));
plot!(epochs, loss_on_test, lab="Test", c=:green, lw=2, ylims = (0,2));
title!("Conv parametric machine architecture");
yaxis!("Loss");
xaxis!("Training epoch");
savefig("convPM_loss.png");

plot(epochs, acc, lab="Training", c=:black, lw=2, ylims = (0,1));
title!("Conv parametric machine architecture");
yaxis!("Accuracy");
xaxis!("Training epoch");
savefig("convPM_accuracy.png");




