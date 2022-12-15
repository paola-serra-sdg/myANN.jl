using ParametricMachinesDemos, Flux
  
poly(x, y) = (2x-1)^2 + 2y + x * y - 3
poly((x, y)) = poly(x, y)
rg = 0:0.01:1
N = length(rg)
flat = hcat(repeat(rg, inner=N), repeat(rg, outer=N))
truth = map(poly, eachrow(flat))

## Let us generate a `6 x 6` trainig grid.

N_train = 6
rg_train = range(0, 1, length=N_train)
X_train = hcat(repeat(rg_train, inner=N_train), repeat(rg_train, outer=N_train))
Y_train = map(poly, eachcol(X_train))

dimensions = [2, 4, 4, 4]

machine = DenseMachine(dimensions, sigmoid)

model = Flux.Chain(machine, Dense(sum(dimensions), 1)) |> f64

function loss(X_train, Y_train)
    Flux.Losses.mse(model(X_train), Y_train)
end

opt = ADAM(0.1)
ps = Flux.params(model)

# check that learning happens correctly

N = 14
rg = range(0, 1, length=N)
X = hcat(repeat(rg, inner=N), repeat(rg, outer=N)) 
Y = map(poly, eachcol(X)) 

for i in 1:10000
    gs = gradient(ps) do
        loss(X_train, Y_train)
    end
    Flux.Optimise.update!(opt, ps, gs)
    if i % 500 == 0
        @show loss(X_train, Y_train)
        @show loss(X, Y)
    end
end







t = -pi:0.1:pi

minibatch = 32
x = zeros(length(t), 1, minibatch)
y = zeros(length(t), 1, minibatch)
shift = 10
for i in 1:minibatch
    v = @. sin(t) + 0.1 * (rand() - 0.5)
    x[:, 1, i] .= v
    y[:, 1, i] .= circshift(v, 10)
end

dimensions = [1, 4, 4, 4]



















t = -pi:0.1:pi

minibatch = 32
x = zeros(length(t), 1, minibatch)
y = zeros(length(t), 1, minibatch)
shift = 10
for i in 1:minibatch
    v = @. sin(t) + 0.1 * (rand() - 0.5)
    x[:, 1, i] .= v
    y[:, 1, i] .= circshift(v, 10)
end

dimensions = [1, 4, 4, 4]

## RecurMachine

machine = RecurMachine(dimensions, sigmoid; pad=3, timeblock=5)

model = Flux.Chain(machine, Conv((1,), sum(dimensions) => 1)) |> f64

model(x)

function loss(X_train, Y_train)
    Flux.Losses.mse(model(X_train), Y_train)
end

loss(x, y)

opt = ADAM(0.1)
ps = Flux.params(model)

# check that learning happens correctly

for i in 1:1000
    gs = gradient(ps) do
        loss(x, y)
    end
    Flux.Optimise.update!(opt, ps, gs)
    if i % 10 == 0
        @show loss(x, y)
    end
end