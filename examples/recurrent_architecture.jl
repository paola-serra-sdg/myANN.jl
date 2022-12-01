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

model = Chain(RNN(63, 32), Dense(32, 63));

opt = ADAM(0.1);

params = Flux.params(model);

epochs = Float64[]
l = Float64[]

# Training 
for epoch in 1:50
    Flux.reset!(model)
    gs = gradient(params) do 
        my_loss(model(x), y)  
    end
    push!(epochs, epoch)
    push!(l, my_loss(model(x), y))
    Flux.update!(opt, params, gs)
end

# Visualization
plot(epochs, l, lab="Loss", c=:green, lw=2);
title!("Recurrent architecture");
yaxis!("Loss", :log);
xaxis!("Training epoch");
savefig("recurrent_loss");