module Forward

using ..NeocognitronCore
using ..Utils

# Activation function for S-cell
function activate_scell(scell::NeocognitronCore.SCell, input::Vector{Float64})
    NeocognitronCore.sigmoid(dot(scell.weights, input) + scell.bias)
end

# Activation function for C-cell
function activate_ccell(ccell::NeocognitronCore.CCell, scells::Vector{NeocognitronCore.SCell})
    ccell.value = maximum([scell.value for scell in scells])
end

# Forward propagation function
function forward(network::NeocognitronCore.Neocognitron, input::Vector{Float64})
    # Set the value of the input neurons to the input
    for (neuron, value) in zip(network.layers[1].neurons, input)
        neuron.value = value
    end

    # Propagate the values through the network
    for i in 2:length(network.layers)
        prev_layer = network.layers[i-1]
        for cell in network.layers[i].neurons
            activation = 0.0  # Initialize activation
            if cell isa NeocognitronCore.SCell
                activation = activate_scell(cell, [n.value for n in prev_layer.neurons])
            elseif cell isa NeocognitronCore.CCell
                activation = activate_ccell(cell, prev_layer.neurons)
            end
            cell.value = activation
        end
    end

    # Return the output of the network
    return [cell.value for cell in network.layers[end].neurons]
end

end  # module
