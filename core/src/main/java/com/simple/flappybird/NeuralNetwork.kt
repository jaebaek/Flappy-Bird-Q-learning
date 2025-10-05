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

private fun List<List<Double>>.toMutableList(): MutableList<MutableList<Double>> =
    map { innerList -> innerList.toMutableList() }.toMutableList()

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
    constructor() : this(DenseLayer(inputCount = 7, outputCount = 3), DenseLayer(inputCount = 3, outputCount = 2, activation = null, activationPrime = null))

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

class Network(
    val inputCount: Int,
    val hiddenCount: Int,
    val outputCount: Int,
    val weightsHidden: DoubleArray,
    val biasesHidden: DoubleArray,
    val weightsOutput: DoubleArray,
    val biasesOutput: DoubleArray
) {
    companion object {
        fun sigmoid(f: Double): Double = 1.0 / (1.0 + exp(-f))
        fun sigmoidPrim(f: Double): Double {
            val sig = sigmoid(f)
            return sig * (1.0 - sig)
        }
    }

    fun predict(input: DoubleArray): DoubleArray {
        val hidden = DoubleArray(hiddenCount)
        val output = DoubleArray(outputCount)
        return predict(input, hidden, output)
    }

    fun predict(input: DoubleArray, hidden: DoubleArray, output: DoubleArray): DoubleArray {
        for (c in 0..<hiddenCount) {
            var sum = 0.0
            for (r in 0..<inputCount) {
                sum += input[r] * weightsHidden[r * hiddenCount + c]
            }

            hidden[c] = sigmoid(sum + biasesHidden[c])
        }

        for (c in 0..<outputCount) {
            var sum = 0.0
            for (r in 0..<hiddenCount) {
                sum += hidden[r] * weightsOutput[r * outputCount + c]
            }

            output[c] = sigmoid(sum + biasesOutput[c])
        }

        return output
    }
}

class Trainer(
    val network: Network,
    val hidden: DoubleArray,
    val output: DoubleArray,
    val gradHidden: DoubleArray,
    val gradOutput: DoubleArray
) {
    companion object {
        fun create(
            inputCount: Int, hiddenCount: Int, outputCount: Int
        ): Trainer {
            val weightsHidden =
                DoubleArray(inputCount * hiddenCount) { _ -> Random.nextDouble() - 0.5 }
            var biasesHidden = DoubleArray(hiddenCount)
            val weightsOutput =
                DoubleArray(hiddenCount * outputCount) { _ -> Random.nextDouble() - 0.5 }
            val biasesOutput = DoubleArray(outputCount)
            val network = Network(
                inputCount,
                hiddenCount,
                outputCount,
                weightsHidden,
                biasesHidden,
                weightsOutput,
                biasesOutput
            )
            val hidden = DoubleArray(hiddenCount)
            val output = DoubleArray(outputCount)
            val gradHidden = DoubleArray(hiddenCount)
            val gradOutput = DoubleArray(outputCount)
            return Trainer(network, hidden, output, gradHidden, gradOutput)
        }

        fun sigmoidPrim(f: Double) = f * (1.0 - f)
    }

    override fun toString(): String {
        return buildString {
            appendLine(network.biasesHidden.contentToString())
            appendLine(network.weightsHidden.contentToString())
            appendLine(hidden.contentToString())
            appendLine(network.biasesOutput.contentToString())
            appendLine(network.weightsOutput.contentToString())
            appendLine(output.contentToString())
        }
    }

    fun train(input: DoubleArray, y: DoubleArray, lr: Double) {
        network.predict(input, hidden, output)
        for (c in 0..<network.outputCount) {
            gradOutput[c] = (output[c] - y[c]) * sigmoidPrim(output[c])
        }

        for (r in 0..<network.hiddenCount) {
            var sum = 0.0
            for (c in 0..<network.outputCount) {
                sum += gradOutput[c] * network.weightsOutput[r * network.outputCount + c]
            }

            gradHidden[r] = sum * sigmoidPrim(hidden[r])
        }

        for (r in 0..<network.hiddenCount) {
            for (c in 0..<network.outputCount) {
                network.weightsOutput[r * network.outputCount + c] -= lr * gradOutput[c] * hidden[r]
            }
        }

        for (r in 0..<network.inputCount) {
            for (c in 0..<network.hiddenCount) {
                network.weightsHidden[r * network.hiddenCount + c] -= lr * gradHidden[c] * input[r]
            }
        }

        for (c in 0..<network.outputCount) {
            network.biasesOutput[c] -= lr * gradOutput[c]
        }

        for (c in 0..<network.hiddenCount) {
            network.biasesHidden[c] -= lr * gradHidden[c]
        }
    }
}
