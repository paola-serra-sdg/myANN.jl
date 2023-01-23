using DelimitedFiles
using Flux
using Flux: @epochs, onehotbatch, onecold, logitcrossentropy, train!, throttle, flatten
using Statistics: mean, std
using Images
using Flux.Data: DataLoader
using myANN
using Plots
using Random
using ParametricMachinesDemos


train = readdlm("ischemie/data/ECG200_TRAIN.txt")
test = readdlm("ischemie/data/ECG200_TEST.txt")

y_train = train[:, 1]
y_test = test[:,1]
y_train = onehotbatch(y_train, (-1,1))
y_test = onehotbatch(y_test, (-1,1))

x_train = permutedims(train[:, 2:end], (2,1))
x_test = permutedims(test[:, 2:end], (2,1))


# Loading
train_data = DataLoader((x_train, y_train); batchsize = 32, shuffle = true);
test_data = DataLoader((x_test, y_test); batchsize = 32, shuffle = true);

dimensions = [32,16,8];

# Define the parametric machine
machine = DenseMachine(dimensions, sigmoid);

model = Flux.Chain(Dense(96, 32), machine, Dense(sum(dimensions), 2)) |> f64;

model = cpu(model)

# Parameters
params = Flux.params(model)

optimiser = ADAM(0.01)

loss(x,y) = logitcrossentropy(model(x), y)

# Training and plotting
epochs = Int64[]
loss_on_train = Float64[]
loss_on_test = Float64[]
acc_train = Float64[]
acc_test = Float64[]
best_params = Float32[]

for epoch in 1:200

    # Train
    Flux.train!(loss, params, train_data, optimiser)

    # Show the sum of the gradients
    gs = gradient(params) do
        loss(x_train, y_train)
    end
    @show [norm(g,2) for g in gs] # gradient's norm
    
    # Saving losses and accuracies for visualization
    push!(epochs, epoch)
    push!(loss_on_train, loss(x_train, y_train))
    push!(loss_on_test, loss(x_test, y_test))
    push!(acc_train, accuracy(y_train, model(x_train)))
    push!(acc_test, accuracy(y_test, model(x_test)))
    @show loss(x_train, y_train)
    @show loss(x_test, y_test)

    # Saving the best parameters
    if epoch > 1
        if is_best(loss_on_test[epoch-1], loss_on_test[epoch])
            best_params = params
        end
    end
end

@show maximum(acc_train)
@show maximum(acc_test)

# Extract and add new trained parameters
if isempty(best_params)
    best_params = params
end

Flux.loadparams!(model, best_params);


# Visualization
# plot(epochs, loss_on_train, lab="Training", c=:black, lw=2, ylims = (0,20));
# plot!(epochs, loss_on_test, lab="Test", c=:green, lw=2, ylims = (0,20));
# title!("Ischemie - dense machine");
# yaxis!("Loss");
# xaxis!("Training epoch");
# savefig("ischemie_dense_loss.png");

# plot(epochs, acc_train, lab="Training", c=:black, lw=2, ylims = (0,1));
# plot!(epochs, acc_test, lab="Test", c=:green, lw=2, ylims = (0,1));
# title!("Ischemie - dense machine");
# yaxis!("Accuracy");
# xaxis!("Training epoch");
# savefig("ischemie_dense_accuracy.png");