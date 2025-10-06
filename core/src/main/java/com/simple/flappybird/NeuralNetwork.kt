package com.simple.flappybird

import kotlin.math.exp
import kotlin.random.Random

private fun sigmoid(f: Double): Double = 1.0 / (1.0 + exp(-f))
private fun sigmoidPrime(f: Double): Double {
    val sig = sigmoid(f)
    return sig * (1.0 - sig)
}

class ForwardPassResult(
    val output: List<Double>,
    val outputOfActivation: List<Double>,
)

private fun List<List<Double>>.toMutableList(): MutableList<MutableList<Double>> {
    val result  = mutableListOf<MutableList<Double>>()
    for (innerList in this) {
        result.add(innerList.toMutableList())
    }
    return result
}

/**
 * Represents a single dense layer with an optional activation of a neural network.
 */
class DenseLayer(
    val inputCount: Int,
    val outputCount: Int,
    val activation: ((Double) -> Double)? = ::sigmoid,
    val activationPrime: ((Double) -> Double)? = ::sigmoidPrime,
    val learningRate: Double = 0.002,
    val initialWeights: List<List<Double>> = emptyList(),
    val initialBiases: List<Double> = emptyList(),
) {
    private val weights = initialWeights.ifEmpty {
        List(outputCount) { _ ->
            List(inputCount) { _ ->
                val f = Random.nextDouble() - 0.5
                if (f == 0.0) {
                    0.01
                } else {
                    f
                }
            }
        }
    }.toMutableList()

    private val biases = initialBiases.ifEmpty {
        List(outputCount) { _ ->
            val f = Random.nextDouble() - 0.5
            if (f == 0.0) {
                0.01
            } else {
                f
            }
        }
    }.toMutableList()

    /**
     * Performs a forward pass through the layer.
     *
     * @param input a(l - 1).
     * @return The pair of matmul (W(l)*a(l-1) + b(l)) and sigmoid (sigmoid(matmul)) outputs.
     */
    fun forward(input: List<Double>): ForwardPassResult {
        assert(input.size == inputCount)

        val output = mutableListOf<Double>()
        val outputOfActivation = mutableListOf<Double>()
        for (i in 0..<outputCount) {
            var sum = 0.0
            for (j in 0..<inputCount) {
                sum += input[j] * weights[i][j]
            }
            output.add(sum)
            activation?.let {
                outputOfActivation.add(it(sum))
            }
        }

        assert(output.size == outputCount)
        return ForwardPassResult(output, outputOfActivation)
    }

    /**
     * Performs a backward pass through the layer.
     *
     * Notation: dx = delta L / delta x
     *
     * dz(l) = da(l) * (delta a(l) / delta z(l)) = da(l) * g'(z(l))
     * dw(l, i, j) = dz(l, i) * a(l-1, j)
     * db(l, i) = dz(l, i)
     * da(l-1, j) = sum(w(l, i, j) * dz(l, i))
     *
     * @param da The delta L / delta a(l)
     * @param input The a(l - 1)
     * @param forwardOutput The output of the forward pass for [input]
     */
    fun backward(da: List<Double>, input: List<Double>, forwardOutput: ForwardPassResult): List<Double> {
        assert(input.size == inputCount)
        assert(da.size == outputCount)
        val dz = mutableListOf<Double>()
        for (i in 0..<outputCount) {
            dz.add(da[i] * (activationPrime?.invoke(forwardOutput.output[i]) ?: 1.0))
        }
        val dw = mutableListOf<MutableList<Double>>()
        for (i in 0..<outputCount) {
            val row = mutableListOf<Double>()
            for (j in 0..<inputCount) {
                row.add(input[j] * dz[i])
            }
            dw.add(row)
        }
        val db = dz.toList()
        val daPrev = mutableListOf<Double>()
        for (j in 0..<inputCount) {
            var sum = 0.0
            for (i in 0..<outputCount) {
                sum += weights[i][j] * dz[i]
            }
            daPrev.add(sum)
        }

        for (j in 0..<inputCount) {
            for (i in 0..<outputCount) {
                weights[i][j] -= learningRate * dw[i][j]
            }
        }
        for (i in 0..<outputCount) {
            biases[i] -= learningRate * db[i]
        }

        return daPrev
    }
}

class NeuralNetwork(val layers: List<DenseLayer>) {
    constructor(vararg layers: DenseLayer) : this(layers.toList())
    constructor() : this(DenseLayer(inputCount = 3, outputCount = 3), DenseLayer(inputCount = 3, outputCount = 2, activation = null, activationPrime = null))

    fun predict(input: List<Double>): List<Double> {
        var a = input
        for (layer in layers) {
            val result = layer.forward(a)
            a = if (layer.activation != null) {
                result.outputOfActivation
            } else {
                result.output
            }
        }
        return a
    }

    private fun mse(y: List<Double>, yHat: List<Double>): Double {
        assert(y.size == yHat.size)
        var sum = 0.0
        for (i in y.indices) {
            sum += (y[i] - yHat[i]) * (y[i] - yHat[i])
        }
        return sum / y.size
    }

    private fun msePrime(y: List<Double>, yHat: List<Double>): List<Double> {
        assert(y.size == yHat.size)
        val result = mutableListOf<Double>()
        for (i in y.indices) {
            result.add(2.0 * (yHat[i] - y[i]))
        }
        return result
    }

    fun train(x: List<Double>, y: List<Double>) {
        var a = x
        val forwardOutputs = mutableListOf<ForwardPassResult>()
        val forwardInputs = mutableListOf<List<Double>>()
        for (layer in layers) {
            forwardInputs.add(a)
            val result = layer.forward(a)
            forwardOutputs.add(result)
            a = if (layer.activation != null) {
                result.outputOfActivation
            } else {
                result.output
            }
        }
        var da = msePrime(y, a)
        for (i in layers.size - 1 downTo 0) {
            val layer = layers[i]
            da = layer.backward(da, forwardInputs[i], forwardOutputs[i])
        }
    }
}
