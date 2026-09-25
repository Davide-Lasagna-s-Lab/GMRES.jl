@testset "arnoldi eigvals                        " begin
    n = 12
    A = randn(n, n)
    b = randn(n)

    vals, _ = GMRES.arnoldi_eigvals(A, b)
    true_vals = eigvals(A)

    key(v) = (real(v), imag(v))
    @test sort(vals; by = key) ≈ sort(true_vals; by = key) atol = 1e-8
end

@testset "eigvals                                " begin
    # see Trefthen and Bau for this test case
    Random.seed!(1)
    m = 200
    A = 2*Matrix{Float64}(I, m, m) + 0.5*randn(m, m)/sqrt(m)
    b = ones(m)

    # solve
    Λ, res = GMRES.arnoldi_eigvals(A, b, m=200)
    # evals = GMRES.eigvals(A, b, 4, 200)

    # test output
    @test length(Λ)   == 200
    @test length(res) == 200

    # test convergence
    real_eigvals = sort(LinearAlgebra.eigvals(A); by=real, rev=true)
    # for i = 1:4
    #     @test abs(real(evals[i][end]) - real(real_eigvals[i])) < 1e-7
    #     @test abs(imag(evals[i][end]) - imag(real_eigvals[i])) < 1e-7
    # end
    @test_broken all(abs(Λ .- real(real_eigvals)) .< 1e-7)
    @test_broken all(abs(Λ .- imag(real_eigvals)) .< 1e-7)
end
