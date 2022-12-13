using Flux
using Statistics: mean
using Plots

function my_loss(x, y)
    Flux.Losses.mse(x, y)
end

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

dimensions = [300,200,200,200];

machine = RecurMachine(dimensions, sigmoid; pad=3, timeblock=5)

model = Flux.Chain(machine, Conv((1,), sum(dimensions) => 1)) |> f64

opt = ADAM(0.1);

params = Flux.params(model);

epochs = Float64[]
l = Float64[]

# Training 
for i in 1:5
    gs = gradient(ps) do
        loss(x, y)
    end
    Flux.Optimise.update!(opt, ps, gs)
    if i % 1 == 0
        # @show loss(x, y)
        push!(epochs, i)
        push!(l, loss(x, y))
    end
end

# Visualization
plot(epochs, l, lab="Loss", c=:green, lw=2);
title!("Recurrent parametric machine architecture");
yaxis!("Loss", :log);
xaxis!("Training epoch");
savefig("recurrentPM_loss");