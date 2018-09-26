using GMRES
using Base.Test

@testset "trefthen                               " begin
    # see Trefthen and Bau for this test case
    srand(1)
    m = 200
    A = 2*eye(m) + 0.5*randn(m, m)/sqrt(m)
    b = randn(m)
        
    # solve
    opts = GMRESOptions(tol=1e-7, maxiter=10)
    x, convres = gmres!(A, deepcopy(b), opts)

    # check convergence rate
    expected = [4.0^(-n) for n = 1:10]
    @test all(convres.history .< expected*1.1)
    @test all(convres.history .> expected*0.7)

end

@testset "norm of error                          " begin
    # see Trefthen and Bau for this test case
    srand(1)
    m = 200
    A = 2*eye(m) + 0.5*randn(m, m)/sqrt(m)
    x_ex = randn(m)
    b = A*x_ex
        
    # solve with large number of iterations
    x, res = gmres!(A, deepcopy(b), 1e-7, 20, false)

    # norm of error
    @test norm(x - x_ex, 2)/norm(b) < 1e-7
    @test norm(A*x - b,  2)/norm(b) < 1e-7
end

@testset "stopping                               " begin
    # see Trefthen and Bau for this test case
    srand(1)
    m = 200
    A = 2*eye(m) + 0.5*randn(m, m)/sqrt(m)
    x_ex = randn(m)
    b = A*x_ex
        
    # hit tolerance
    opts = GMRESOptions(tol=1e-1, maxiter=20)
    x, convres = gmres!(A, deepcopy(b), opts)
    @test convres.status == :converged
    @test norm(A*x - b,  2)/norm(b) < 1e-1

    # hit iterations number
    for niter = 1:5
        opts = GMRESOptions(tol=1e-15, maxiter=niter)
        x, convres = gmres!(A, deepcopy(b), opts)
        @test convres.status == :maxiterreached
        @test length(convres.history) == niter
    end

    # cant set zero iterations
    @test_throws AssertionError GMRESOptions(tol=1e-15, maxiter=0)
end

@testset "verbose                                " begin
    srand(1)
    m = 200
    A = 2*eye(m) + 0.5*randn(m, m)/sqrt(m)
    x_ex = randn(m)
    b = A*x_ex
        
    # print output
    opts = GMRESOptions(tol=1e-100, maxiter=5, verbose=true)

    # run
    gmres!(A, deepcopy(b), opts)
end