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
x1 = x_train[:,:,:,1:200]
y1 = y_train[:,1:200]
x2 = x_test[:,:,:,1:50]
y2 = y_test[:,1:50]

dimensions = [300,200,200,200];

# Define the parametric machine

machine = ConvMachine(dimensions, sigmoid; pad=(0,0,0,0))

model = Flux.Chain(machine, flatten, Dense(810000,3), softmax) |> f64 

ps = Flux.params(model)
opt = ADAM(0.1)
loss(x,y) = logitcrossentropy(model(x), y)

# check that learning happens correctly
epochs = Int64[]
loss_train = Float64[]
loss_test = Float64[]
acc = Float64[]

for i in 1:3
    gs = gradient(ps) do
        loss(x1, y1)
    end
    Flux.Optimise.update!(opt, ps, gs)
    if i % 1 == 0
        # @show loss(x_train, y_train)
        # @show loss(x_test, y_test)
        push!(epochs, i)
        push!(loss_train, loss(x1, y1))
        push!(loss_test, loss(x2, y2))
        push!(acc, accuracy(y2, model(x2)))
    end
end

plot(epochs, loss_train, lab="Training", c=:black, lw=2, ylims = (0,2));
plot!(epochs, loss_test, lab="Test", c=:green, lw=2, ylims = (0,2));
title!("Conv parametric machine");
yaxis!("Loss", :log);
xaxis!("Training epoch");
savefig("conv_PM.png");

plot(epochs, acc, lab="Training", c=:black, lw=2, ylims = (0,1));
title!("Conv parametric machine architecture");
yaxis!("Accuracy", :log);
xaxis!("Training epoch");
savefig("convPM_accuracy.png");





