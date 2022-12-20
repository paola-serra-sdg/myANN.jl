using Flux
using Statistics: mean
using Plots
using ParametricMachinesDemos


# Sinusoidal data
t = -pi:0.1:pi;

minibatch = 32;
x = zeros(length(t), 1, minibatch);
y = zeros(length(t), 1, minibatch);

for i in 1:minibatch
    v = @. sin(t) + 0.1 * (rand() - 0.5)
    x[:, 1, i] .= v
    y[:, 1, i] .= circshift(v, 10)
end

# To Float32 
x = Float32.(x);

train_data = DataLoader((x, y); batchsize = 32, shuffle = true);

dimensions = [1, 4, 4, 4];

machine = RecurMachine(dimensions, sigmoid; pad=3, timeblock=5)

model = Flux.Chain(machine, Conv((1,), sum(dimensions) => 1)) |> f64

opt = ADAM(0.1);

params = Flux.params(model);

# Loss function
loss(x, y) = Flux.Losses.mse(model(x), y)

epochs = Int64[]
loss_on_train = Float64[]
loss_on_test = Float64[]
acc = Float64[]
best_params = Float32[]

for epoch in 1:20

    # Train
    Flux.train!(loss, params, train_data, optimiser)

    # Show the sum of the gradients
    gs = gradient(params) do
        loss(x, y)
    end
    @show sum(first(gs))

    # Saving loss for visualization
    push!(epochs, epoch)
    push!(loss_on_train, loss(x, y))
    @show loss(x, y)

    # Saving the best parameters
    if epoch > 1
        if is_best(loss_on_train[epoch-1], loss_on_train[epoch])
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
title!("Recurrent architecture");
yaxis!("Loss");
xaxis!("Training epoch");
savefig("recurrent_loss");