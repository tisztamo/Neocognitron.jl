using Test
using Neocognitron
using Neocognitron.Utils
using Neocognitron.Forward
using Neocognitron.NeocognitronCore

@testset "Neocognitron Tests" begin
    @testset "Neuron Creation" begin
        neuron = Utils.Neuron(5)
        @test neuron.value == 0.0
        @test length(neuron.weights) == 5
    end

    @testset "SCell Creation" begin
        scell = Utils.SCell(5)
        @test scell.value == 0.0
        @test length(scell.weights) == 5
    end

    @testset "CCell Creation" begin
        ccell = Utils.CCell()
        @test ccell.value == 0.0
    end

    @testset "Layer Creation" begin
        layer = Utils.Layer(5, 3)
        @test length(layer.neurons) == 5
        @test length(layer.neurons[1].weights) == 3
    end

    @testset "Neocognitron Creation" begin
        neocognitron = Utils.Neocognitron([2, 3, 2])
        @test length(neocognitron.layers) == 3
        @test length(neocognitron.layers[1].neurons) == 2
        @test length(neocognitron.layers[2].neurons) == 3
        @test length(neocognitron.layers[3].neurons) == 2
    end

    @testset "Forward Propagation" begin
        # Set up a specific Neocognitron with known weights and biases

        weights1 = [0.2, 0.8]
        weights2 = [0.6, 0.4]
        bias = 0.0
        scell1 = Utils.SCell(2)  # Fixed: Use Utils.SCell instead of Neocognitron.Utils.scell
        scell2 = Utils.SCell(2)  # Fixed: Use Utils.SCell instead of Neocognitron.Utils.scell

        ccell1 = Utils.CCell()  # Fixed: Use Utils.CCell instead of Neocognitron.Utils.ccell
        ccell2 = Utils.CCell()  # Fixed: Use Utils.CCell instead of Neocognitron.Utils.ccell
        layer1 = Utils.Layer(2, 2)  # Fixed: Use Utils.Layer instead of Neocognitron.Utils.layer
        layer2 = Utils.Layer(2, 2)  # Fixed: Use Utils.Layer instead of Neocognitron.Utils.layer

        neocognitron = Utils.Neocognitron([layer1, layer2])  # Fixed: Use Utils.Neocognitron instead of Neocognitron.Utils.neocognitron
        # Set up a specific input
        input = [0.5, 0.7]
    
        # Run forward propagation
        output = Forward.forward(neocognitron, input)
    
        # Test output
        # The neurons should output the result of the sigmoid function applied to the dot product of weights and input
        @test output[1] ≈ NeocognitronCore.sigmoid(dot([0.5, 0.7], input))
        @test output[2] ≈ NeocognitronCore.sigmoid(dot([0.2, 0.4], input))
        @test output[3] ≈ max(output[1:2])
    end
end
