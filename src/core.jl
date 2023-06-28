module NeocognitronCore

export Neuron, Layer, Neocognitron, sigmoid, Cell, SCell, CCell

mutable struct Neuron
    value::Float64
    weights::Vector{Float64}
end

struct Layer
    neurons::Vector{Neuron}
end

struct Neocognitron
    layers::Vector{Layer}
end

abstract type Cell end

mutable struct SCell <: Cell
    value::Float64
    weights::Vector{Float64}
    bias::Float64
end

mutable struct CCell <: Cell
    value::Float64
end

# Activation function
sigmoid(x) = 1.0 / (1.0 + exp(-x))

end  # module
