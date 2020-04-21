@testset "restart                                " begin
    # see Trefthen and Bau for this test case
    Random.seed!(1)
    m = 200
    A = 2*Matrix{Float64}(I, m, m) + 0.5*randn(m, m)/sqrt(m)
    b = randn(m)
    x = zeros(m)
        
    # solve
    x, res_norm, it = gmres!(x, A, b; m = 2, rel_rtol=1e-7, maxiter=10, verbose=false)

    # check convergence rate
    expected = [4.0^(-n) for n = 1:10]
    @test res_norm/norm(b) < 1.1*(4.0^(-10.0))
    @test res_norm/norm(b) > 0.7*(4.0^(-10.0))
end

@testset "trefthen                               " begin
    # see Trefthen and Bau for this test case
    Random.seed!(1)
    m = 200
    A = 2*Matrix{Float64}(I, m, m) + 0.5*randn(m, m)/sqrt(m)
    b = randn(m)
    x = zeros(m)
        
    # solve
    x, res_norm, it = gmres!(x, A, b; rel_rtol=1e-7, maxiter=10, verbose=false)

    # check convergence rate
    expected = [4.0^(-n) for n = 1:10]
    @test res_norm/norm(b) < 1.1*(4.0^(-10.0))
    @test res_norm/norm(b) > 0.7*(4.0^(-10.0))
end

@testset "norm of error                          " begin
    # see Trefthen and Bau for this test case
    Random.seed!(1)
    m = 200
    A = 2*Matrix{Float64}(I, m, m) + 0.5*randn(m, m)/sqrt(m)
    x_ex = randn(m)
    b = A*x_ex
    x = zeros(m)
        
    # solve with large number of iterations
    x, r_norm, it = gmres!(x, A, b; rel_rtol=1e-7, maxiter=20, verbose=false)

    # norm of error
    @test norm(x - x_ex, 2)/norm(b) < 1e-7
    @test norm(A*x - b,  2)/norm(b) < 1e-7

    # try non zero initial vector
    x, r_norm, it = gmres!(randn(m), A, b; rel_rtol=1e-7, maxiter=20, verbose=false)

    # norm of error
    @test norm(x - x_ex, 2)/norm(b) < 1e-7
    @test norm(A*x - b,  2)/norm(b) < 1e-7
end

@testset "hookstep                               " begin
    # come up with a matrix of size m x m    
    Random.seed!(1)
    m = 50
    A = 2*Matrix{Float64}(I, m, m) + 0.5*randn(m, m)/sqrt(m)

    # exact solution
    x_ex = randn(m)

    # initial guess
    x = zeros(m)

    # right hand side
    b = A*x_ex

    # for large Δ (maximum norm of solution) the solution 
    # of min_x ||A*x - b|| s.t. ||x|| < Δ
    # is the exact solution, with small residual
    Δ = 100
    
    x, r_norm, it = gmres!(x, A, b, Δ; rel_rtol=1e-10, maxiter=20, verbose=false)

    @test r_norm/norm(b) < 1e-10
        
    # for medium Δ we need to respect the constraint
    Δ = 1

    # initial guess
    x = zeros(m)
    
    x, r_norm, it = gmres!(x, A, b, Δ; rel_rtol=1e-10, maxiter=50, verbose=false)
    @test norm(x) < (1 + 1e-15)*Δ

    # in addition, the minimum of ||A*x - b|| s.t. ||x|| < Δ
    # should roughly match with what we have if we do 
    # proper optimisation. To achieve this, however, we need to 
    # have enough iteration so that the basis vectors of the 
    # krylov subspace span the minimum of the problem
    using NLopt

    # function and constraint
    fun(x, grad) = (val = norm(A*x .- b); val)
    constr(res, x, grad) = (res[1] = norm(x) - Δ; res)

    # define optimisation problem
    opt = Opt(:LN_COBYLA, m)
    ftol_abs!(opt, 1e-12)
    min_objective!(opt, fun)
    inequality_constraint!(opt, constr, [1e-12])

    # solve it
    (optf, optx, ret) = optimize(opt, zeros(m))

    # this should be what we got before
    @test abs( norm(A*x.- b) - norm(A*optx .- b) ) < 1e-13
end