/*
 * Apache License 2.0
 *
 * Acknowledgement: Copied from https://github.com/kostasdrakonakis/flappybird
 *
 * Initial copyright:
 *
 * Copyright 2018 Konstantinos Drakonakis.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.simple.flappybird

import com.badlogic.gdx.ApplicationAdapter
import com.badlogic.gdx.Gdx
import com.badlogic.gdx.graphics.Color
import com.badlogic.gdx.graphics.Texture
import com.badlogic.gdx.graphics.g2d.BitmapFont
import com.badlogic.gdx.graphics.g2d.SpriteBatch
import com.badlogic.gdx.math.Intersector
import com.badlogic.gdx.math.Rectangle
import java.util.*
import kotlin.collections.toDoubleArray

class FlappyBird : ApplicationAdapter() {

    private lateinit var batch: SpriteBatch
    private lateinit var background: Texture
    private lateinit var gameOver: Texture
    private lateinit var birds: Array<Texture>
    private lateinit var topTubeRectangles: Array<Rectangle?>
    private lateinit var bottomTubeRectangles: Array<Rectangle?>
    private lateinit var font: BitmapFont
    private lateinit var topTube: Texture
    private lateinit var bottomTube: Texture
    private lateinit var random: Random

    private var speedUpFactor = 1

    private var flapState = 0
    private var timeFromLastFlap = 0.0f
    private var birdY: Float = 0f
    private val birdX: Float by lazy { gdxWidth / 4f - currentBird.width / 2f }
    private val currentBird by lazy { birds[flapState] }

    val agent = DQNAgent(stateSize = 8, actionSize = 2)
    var remainingEpisodes = 1000
    var reward = 0.1

    /**
     * Box surrounding the bird. It is used to detect collision.
     */
    private val birdBox: Rectangle
        get() = Rectangle(
            birdX, birdY,
            currentBird.width.toFloat(),
            currentBird.height.toFloat(),
        )

    private var velocity: Float = 0f
    private var score: Int = 0
    private var scoringTube: Int = 0
    private var gameState: GameStates = GameStates.PLAYING
    private val numberOfTubes: Int = 4
    private var gdxHeight: Int = 0
    private var gdxWidth: Int = 0
    private var topTubeWidth: Int = 0
    private var bottomTubeWidth: Int = 0
    private var topTubeHeight: Int = 0
    private var bottomTubeHeight: Int = 0

    /** X coordinates of tubes */
    private val tubeX = FloatArray(numberOfTubes)

    /** Y coordinates of the open space of the tubes */
    private val tubeOpenSpaceY = FloatArray(numberOfTubes)

    private fun getTopTubeY(i: Int) = gdxHeight / 2f + gap / 2 + tubeOpenSpaceY[i]
    private fun getBottomTubeY(i: Int) = gdxHeight / 2f - gap / 2 - bottomTubeHeight.toFloat() + tubeOpenSpaceY[i]

    private val gap by lazy { currentBird.height * 3.0f }

    private var distanceBetweenTubes: Float = 0.toFloat()

    override fun create() {
        batch = SpriteBatch()
        background = Texture("bg.png")
        gameOver = Texture("gameover.png")
        font = BitmapFont()
        font.color = Color.WHITE
        font.data.setScale(10f)

        birds = arrayOf(Texture("bird.png"), Texture("bird2.png"))

        gdxHeight = Gdx.graphics.height
        gdxWidth = Gdx.graphics.width

        topTube = Texture("toptube.png")
        bottomTube = Texture("bottomtube.png")
        random = Random()
        distanceBetweenTubes = gdxWidth * 3f / 4f
        topTubeRectangles = arrayOfNulls(numberOfTubes)
        bottomTubeRectangles = arrayOfNulls(numberOfTubes)

        topTubeWidth = topTube.width
        topTubeHeight = topTube.height
        bottomTubeWidth = bottomTube.width
        bottomTubeHeight = bottomTube.height

        startGame()
    }

    override fun render() {
        if (remainingEpisodes == 0) {
            val action = agent.chooseAction(getState())
            simulate(action == JUMP_ACTION)
            return
        }

        updateSpeedUpFactor()

        for (i in 0 until speedUpFactor) {
            val currentState = getState()

            val action = agent.chooseAction(currentState)
            simulate(action == JUMP_ACTION)

            agent.remember(currentState, action, reward, getState(), gameState == GameStates.GAME_OVER)
            agent.replay()

            if (gameState == GameStates.GAME_OVER) {
                println("Episode: $remainingEpisodes, Score: $score, Epsilon: ${agent.epsilon}")
                reset()
                remainingEpisodes--
            }
            if (remainingEpisodes == 0) return
        }
    }

    private fun updateSpeedUpFactor() {
        if (Gdx.input.justTouched()) {
            if (speedUpFactor >= 8) {
                speedUpFactor = 1
            } else {
                speedUpFactor *= 2
            }
        }
    }

    private fun getState(): DoubleArray {
        val relevantIndices = tubeX.indices
            .filter { tubeX[it] + topTubeWidth > birdX }
            .sortedBy { tubeX[it] }
        val i1 = relevantIndices[0]
        val i2 = relevantIndices[1]
        val topTubeY1 = getTopTubeY(i1)
        val topTubeY2 = getTopTubeY(i2)
        val bottomTubeY1 = getBottomTubeY(i1) + bottomTubeHeight.toFloat()
        val bottomTubeY2 = getBottomTubeY(i2) + bottomTubeHeight.toFloat()
        val state = listOf(birdX, birdY, tubeX[i1], topTubeY1, bottomTubeY1, tubeX[i2], topTubeY2, bottomTubeY2)
        return state.map { it.toDouble() }.toDoubleArray()
    }

    private fun simulate(jumpRequested: Boolean) {
        batch.begin()
        batch.draw(background, 0f, 0f, gdxWidth.toFloat(), gdxHeight.toFloat())

        assert(gameState == GameStates.PLAYING)
        reward = 0.1

        updateScore()
        if (jumpRequested) {
            jump()
        }
        drawTubes()
        updateBirdY()

        updateFlapState()

        batch.draw(currentBird, birdX, birdY)
        font.draw(batch, score.toString(), 100f, 200f)

        checkCrash()

        batch.end()
    }

    private fun updateScore() {
        if (tubeX[scoringTube] < birdX) {
            score++
            reward = 1.0
            if (scoringTube < numberOfTubes - 1) {
                scoringTube++
            } else {
                scoringTube = 0
            }
        }
    }

    private fun jump() { velocity = -0.2f * currentBird.height }

    private fun updateBirdY() {
        if (birdY > 0 && birdY < gdxHeight - currentBird.height) {
            velocity += GRAVITY
            birdY -= velocity
        } else {
            reward = -10.0
            gameState = GameStates.GAME_OVER
        }
    }

    private fun drawTubes() {
        for (i in 0 until numberOfTubes) {
            updateTubePosition(i)

            val topTubeY = getTopTubeY(i)
            val bottomTubeY = getBottomTubeY(i)
            batch.draw(topTube, tubeX[i], topTubeY)
            batch.draw(
                bottomTube, tubeX[i], bottomTubeY
            )
            topTubeRectangles[i] = Rectangle(
                tubeX[i], topTubeY, topTubeWidth.toFloat(), topTubeHeight.toFloat()
            )
            bottomTubeRectangles[i] = Rectangle(
                tubeX[i], bottomTubeY, bottomTubeWidth.toFloat(), bottomTubeHeight.toFloat()
            )
        }
    }

    private fun updateTubePosition(i: Int) {
        if (tubeX[i] < -topTubeWidth) {
            tubeX[i] += numberOfTubes * distanceBetweenTubes
            tubeOpenSpaceY[i] = (random.nextFloat() - 0.5f) * (gdxHeight.toFloat() - gap - 200f)
        } else {
            tubeX[i] = tubeX[i] - TUBE_VELOCITY
        }
    }

    private fun updateFlapState() {
        timeFromLastFlap += Gdx.graphics.deltaTime
        if (timeFromLastFlap >= 0.2f) {
            flapState = if (flapState == 0) 1 else 0
            timeFromLastFlap = 0.0f
        }
    }

    private fun reset() {
        score = 0
        scoringTube = 0
        velocity = 0f
        startGame()
    }

    private fun checkCrash() {
        for (i in 0 until numberOfTubes) {
            if (Intersector.overlaps(birdBox, topTubeRectangles[i])
                || Intersector.overlaps(birdBox, bottomTubeRectangles[i])
            ) {
                reward = -10.0
                gameState = GameStates.GAME_OVER
                return
            }
        }
    }

    private fun startGame() {
        birdY = gdxHeight / 2f - birds[0].height / 2f

        for (i in 0 until numberOfTubes) {
            tubeOpenSpaceY[i] = (random.nextFloat() - 0.5f) * (gdxHeight.toFloat() - gap - 200f)
            tubeX[i] = gdxWidth / 2f - topTubeWidth / 2f +
                TUBE_START_POS_FACTOR * gdxWidth.toFloat() + i * distanceBetweenTubes
            topTubeRectangles[i] = Rectangle()
            bottomTubeRectangles[i] = Rectangle()
        }
        gameState = GameStates.PLAYING
    }

    companion object {
        private const val GRAVITY = 2f
        private const val TUBE_VELOCITY = 4f
        private const val TUBE_START_POS_FACTOR = 0.5f
        private const val JUMP_ACTION = 1
        const val FPS = 30
    }
}
