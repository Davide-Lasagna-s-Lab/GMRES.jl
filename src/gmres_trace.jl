# Trace object to keep track of final state of GMRES solver between systems.

mutable struct GMRESTrace{X}
    its::Vector{Int}
    Qs::Vector{Vector{X}}
    Hs::Vector{Matrix{Float64}}

    GMRESTrace(x::X) where {X} = new{X}(Int[], Vector{X}[], Matrix{Float64}[])
end

Base.length(t::GMRESTrace) = length(t.its)
Base.getindex(t::GMRESTrace, i::Int) = (t.its[i], t.Qs[i], t.Hs[i])

function Base.push!(t::GMRESTrace, it, Q, H)
    push!(t.its, it)
    push!(t.Qs, Q)
    push!(t.Hs, H)
end
