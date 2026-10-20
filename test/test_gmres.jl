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
    
    # (a complete Krylov space: rel_rtol = 0 runs all the m iterations)
    x, r_norm, it = gmres!(x, A, b, Δ; rel_rtol=0.0, maxiter=50, verbose=false)
    @test norm(x) < (1 + 1e-15)*Δ

    # in addition, the minimum of ||A*x - b|| s.t. ||x|| < Δ
    # should roughly match with what we have if we do 
    # proper optimisation. To achieve this, however, we need to 
    # have enough iteration so that the basis vectors of the 
    # krylov subspace span the minimum of the problem

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

# an operator that counts its products, for the tests of the hookstep
mutable struct Counted{M}
    A::M        # the matrix
    n::Int      # the products so far
end
Base.:*(C::Counted, x) = (C.n += 1; C.A*x)

@testset "hookstep, stop on the Newton residual  " begin
    Random.seed!(1)
    m = 50
    A = 2*Matrix{Float64}(I, m, m) + 0.5*randn(m, m)/sqrt(m)
    b = A*randn(m)
    C = Counted(A, 0)

    # Δ binds: the iteration stops when the unconstrained solution in the Krylov space meets the tolerance, well
    # before a complete space, with one product per iteration from a zero initial guess
    Δ = 1
    x, r_norm, it, K = gmres!(zeros(m), C, b, Δ; rel_rtol=1e-8, maxiter=50, verbose=false)
    @test it < 30
    @test C.n == it
    @test norm(x) < (1 + 1e-12)*Δ
    @test norm(K.H*(K.H\K.g) - K.g)/norm(b) < 1e-8
    @test abs(r_norm - norm(A*x - b))/norm(b) < 1e-10
    @test r_norm/norm(b) > 1e-2
end

@testset "hookstep for another radius            " begin
    Random.seed!(1)
    m = 50
    A = 2*Matrix{Float64}(I, m, m) + 0.5*randn(m, m)/sqrt(m)
    b = A*randn(m)

    # the hookstep for a smaller radius in the Krylov space of the first call (hookstep! has no access to A) is that
    # of a new call with the smaller radius, which stops at the same iteration
    x₁, r₁, it₁, K = gmres!(zeros(m), A, b, 1.0; rel_rtol=1e-8, maxiter=50, verbose=false)
    x₂, r₂ = hookstep!(zeros(m), K, 0.25)
    x₃, r₃, it₃, _ = gmres!(zeros(m), A, b, 0.25; rel_rtol=1e-8, maxiter=50, verbose=false)
    @test it₃ == it₁
    @test norm(x₂ - x₃) < 1e-12
    @test abs(r₂ - r₃)/norm(b) < 1e-12
    @test abs(r₂ - norm(A*x₂ - b))/norm(b) < 1e-10
    @test norm(x₂) < (1 + 1e-12)*0.25

    # from a nonzero initial guess the radius bounds the correction
    x₀ = 0.1*randn(m)
    x, r, it, K = gmres!(copy(x₀), A, b, 1.0; rel_rtol=1e-8, maxiter=50, verbose=false)
    @test K.x₀ == x₀
    x₄, r₄ = hookstep!(zeros(m), K, 0.25)
    @test norm(x₄ - x₀) < (1 + 1e-12)*0.25
    @test abs(r₄ - norm(A*x₄ - b))/norm(b) < 1e-10
end
