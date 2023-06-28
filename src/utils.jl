module Utils

using ..NeocognitronCore

# Function to create a neuron with random weights
function Neuron(n::Int)
    NeocognitronCore.Neuron(0.0, randn(n))
end

# Function to create a S-cell with random weights and bias
function SCell(n::Int)
    NeocognitronCore.SCell(0.0, randn(n), randn())
end

# Function to create a C-cell
function CCell()
    NeocognitronCore.CCell(0.0)
end

# Function to create a layer with a specified number of neurons, each with a specified number of weights
function Layer(num_neurons::Int, num_weights::Int)
    NeocognitronCore.Layer([Neuron(num_weights) for _ in 1:num_neurons])
end

# Function to create a Neocognitron with specified layer sizes
function Neocognitron(layer_sizes::Vector{Int})
    num_layers = length(layer_sizes)
    layers = Vector{NeocognitronCore.Layer}(undef, num_layers)
    layers[1] = Layer(layer_sizes[1], 0)  # Input layer neurons have no weights
    for i in 2:num_layers
        layers[i] = Layer(layer_sizes[i], layer_sizes[i-1])
    end
    NeocognitronCore.Neocognitron(layers)
end

# Function to create a Neocognitron from provided layers
function Neocognitron(layers::Vector{NeocognitronCore.Layer})
    NeocognitronCore.Neocognitron(layers)
end

end  # module
