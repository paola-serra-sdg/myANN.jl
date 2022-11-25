using myANN
using Test
using Flux: onehotbatch

@testset "myANN.jl" begin
    # Accuracy
    a = onehotbatch((0,1,1,0,1), 0:1)
    b = onehotbatch((0,1,1,0,0), 0:1)
    @test accuracy(a,b) == 0.8
    c = onehotbatch((1,1,1,1,1), 0:1)
    d = onehotbatch((1,1,1,1,1), 0:1)
    @test accuracy(c,d) == 1
    e = onehotbatch((0,0,0,0,0), 0:1)
    f = onehotbatch((1,1,1,1,1), 0:1)
    @test accuracy(e,f) == 0
end
