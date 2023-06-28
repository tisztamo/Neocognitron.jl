module Neocognitron

include("core.jl")
include("utils.jl")
include("forward.jl")

export Neuron, Layer, Neocognitron, sigmoid, forward

end  # module
