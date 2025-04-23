@testset "trace                                  " begin
    # see Trefthen and Bau for this test case
    Random.seed!(1)
    m = 200
    A = 2*Matrix{Float64}(I, m, m) + 0.5*randn(m, m)/sqrt(m)
    x_ex = randn(m)
    b = A*x_ex

    # construct trace
    t = GMRES.GMRESTrace(x_ex)
    t isa GMRES.GMRESTrace{Vector{Float64}}

    # loop over same system solving each one
    maxiters = [10, 8, 10]
    for (i, maxiter) in enumerate(maxiters)
        x = zeros(m)
        gmres!(x, A, b; rel_rtol=1e-7, maxiter=maxiter, verbose=false, trace=t)
    end

    # check trace holds onto information between runs
    @test length(t) == 3
    @test t[1] == t[3]
    @test t.its == maxiters
end
