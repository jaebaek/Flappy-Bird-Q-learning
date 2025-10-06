package com.simple.flappybird

import kotlin.random.Random

/**
 * Represents a single experience tuple (S, A, R, S').
 */
data class Experience(
    val state: List<Double>,
    val action: Int,
    val reward: Double,
    val nextState: List<Double>,
    val done: Boolean // True if the episode ended after this experience
)

/**
 * The Deep Q-Network Agent. It uses an MLP to approximate the Q-function.
 */
class DQNAgent(
    private val actionSize: Int,
    private val replayBufferSize: Int = 512,
    private val batchSize: Int = 128,
    private val gamma: Double = 0.9,       // Discount factor for future rewards
    private val learningRate: Double = 0.001,
) {
    private val nn = NeuralNetwork()

    // Epsilon-greedy strategy parameters
    var epsilon: Double = 1.0               // Exploration rate

    private val replayBuffer = mutableListOf<Experience>()
    private var timeStep = 0

    /**
     * Stores an experience in the replay buffer.
     */
    fun remember(state: List<Double>, action: Int, reward: Double, nextState: List<Double>, done: Boolean) {
        if (replayBuffer.size >= replayBufferSize) {
            replayBuffer.removeAt(0) // Remove oldest experience if buffer is full
        }
        replayBuffer.add(Experience(state, action, reward, nextState, done))
    }

    /**
     * Chooses an action for a given state using the epsilon-greedy policy.
     */
    fun chooseAction(state: List<Double>): Int {
        val randomDouble = Random.nextDouble()
        if (randomDouble <= epsilon) {
            if (randomDouble <= epsilon / 9.0) return 1
            return 0
        }
        // Exploit: choose the best action based on the main network's prediction
        val qValues = nn.predict(state)
        return qValues.indices.maxByOrNull { qValues[it] } ?: 0
    }

    /**
     * Samples a mini-batch from the replay buffer and trains the main network.
     */
    fun replay() {
        if (replayBuffer.size < batchSize) {
            return
        }

        val miniBatch = List(batchSize) { replayBuffer.random() }

        for (experience in miniBatch) {
            val (state, action, reward, nextState, done) = experience

            // Get the predicted Q-values for the next state from the STABLE target network
            val nextQValues = nn.predict(nextState)
            val maxNextQ = nextQValues.maxOrNull() ?: 0.0

            // Calculate the target Q-value for the current state and action
            // If the episode is done, the future reward is 0
            val targetQ = if (done) reward else reward + gamma * maxNextQ

            // Get the current Q-value predictions from the main network
            val currentQValues = nn.predict(state)

            // Create the target vector: it's the same as the current prediction,
            // but updated with the new Q-value for the action that was taken.
            val targetQValues = currentQValues.toMutableList()
            targetQValues[action] = targetQ

            // Train the main network on this single experience
            nn.train(state, targetQValues)
        }

        // Periodically update the target network to match the main network
        timeStep++
    }
}
