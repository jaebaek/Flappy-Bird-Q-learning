package com.simple.flappybird

import kotlin.math.max
import kotlin.random.Random

/**
 * A simple Multi-Layer Perceptron (MLP) implementation from scratch.
 * This network has one hidden layer and is designed for the Flappy Bird DQN.
 *
 * Architecture:
 * - Input Layer: 8 neurons (for the state vector)
 * - Hidden Layer: 3 neurons (with ReLU activation)
 * - Output Layer: 2 neurons (for Q-values of "jump" and "stay")
 */
class MLP(
    private val inputSize: Int,
    private val hiddenSize: Int,
    private val outputSize: Int,
    private val learningRate: Double
) {
    // Weights and biases are initialized with small random values
    var weightsInputHidden: Array<DoubleArray> = Array(inputSize) { DoubleArray(hiddenSize) { Random.nextDouble(-0.5, 0.5) } }
    var biasHidden: DoubleArray = DoubleArray(hiddenSize) { Random.nextDouble(-0.5, 0.5) }
    var weightsHiddenOutput: Array<DoubleArray> = Array(hiddenSize) { DoubleArray(outputSize) { Random.nextDouble(-0.5, 0.5) } }
    var biasOutput: DoubleArray = DoubleArray(outputSize) { Random.nextDouble(-0.5, 0.5) }

    // Activation function (Rectified Linear Unit)
    private fun relu(x: Double): Double = max(0.0, x)
    private fun relu(arr: DoubleArray): DoubleArray = arr.map { relu(it) }.toDoubleArray()

    // Derivative of the ReLU function for backpropagation
    private fun reluDerivative(x: Double): Double = if (x > 0) 1.0 else 0.0

    /**
     * Performs a forward pass through the network to predict Q-values for a given state.
     */
    fun predict(inputs: DoubleArray): DoubleArray {
        // From input to hidden layer
        val hiddenInputs = DoubleArray(hiddenSize)
        for (j in 0 until hiddenSize) {
            var sum = 0.0
            for (i in 0 until inputSize) {
                sum += inputs[i] * weightsInputHidden[i][j]
            }
            hiddenInputs[j] = sum + biasHidden[j]
        }
        val hiddenOutputs = relu(hiddenInputs)

        // From hidden to output layer
        val finalInputs = DoubleArray(outputSize)
        for (j in 0 until outputSize) {
            var sum = 0.0
            for (i in 0 until hiddenSize) {
                sum += hiddenOutputs[i] * weightsHiddenOutput[i][j]
            }
            finalInputs[j] = sum + biasOutput[j]
        }
        // The output layer is linear (no activation) because Q-values are not bounded
        return finalInputs
    }

    /**
     * Trains the network using backpropagation.
     * It adjusts weights and biases to minimize the error between predicted and target Q-values.
     */
    fun train(inputs: DoubleArray, targets: DoubleArray) {
        // --- 1. FORWARD PASS (to get intermediate values) ---
        val hiddenInputs = DoubleArray(hiddenSize)
        for (j in 0 until hiddenSize) {
            var sum = 0.0
            for (i in 0 until inputSize) {
                sum += inputs[i] * weightsInputHidden[i][j]
            }
            hiddenInputs[j] = sum + biasHidden[j]
        }
        val hiddenOutputs = relu(hiddenInputs)

        val finalInputs = DoubleArray(outputSize)
        for (j in 0 until outputSize) {
            var sum = 0.0
            for (i in 0 until hiddenSize) {
                sum += hiddenOutputs[i] * weightsHiddenOutput[i][j]
            }
            finalInputs[j] = sum + biasOutput[j]
        }
        val predictedOutputs = finalInputs // Linear output

        // --- 2. BACKWARD PASS (calculate errors and update weights) ---

        // A. Calculate output layer error and gradients
        val outputErrors = DoubleArray(outputSize) { i -> targets[i] - predictedOutputs[i] }
        val outputGradients = outputErrors // Derivative of linear activation is 1

        // B. Update hidden-to-output weights
        for (i in 0 until hiddenSize) {
            for (j in 0 until outputSize) {
                val delta = learningRate * outputGradients[j] * hiddenOutputs[i]
                weightsHiddenOutput[i][j] += delta
            }
        }
        for (i in 0 until outputSize) {
            biasOutput[i] += learningRate * outputGradients[i]
        }

        // C. Calculate hidden layer error
        val hiddenErrors = DoubleArray(hiddenSize)
        for (i in 0 until hiddenSize) {
            var error = 0.0
            for (j in 0 until outputSize) {
                error += outputErrors[j] * weightsHiddenOutput[i][j]
            }
            hiddenErrors[i] = error
        }

        // D. Calculate hidden layer gradients
        val hiddenGradients = DoubleArray(hiddenSize) { i ->
            hiddenErrors[i] * reluDerivative(hiddenInputs[i])
        }

        // E. Update input-to-hidden weights
        for (i in 0 until inputSize) {
            for (j in 0 until hiddenSize) {
                val delta = learningRate * hiddenGradients[j] * inputs[i]
                weightsInputHidden[i][j] += delta
            }
        }
        for (i in 0 until hiddenSize) {
            biasHidden[i] += learningRate * hiddenGradients[i]
        }
    }
}
