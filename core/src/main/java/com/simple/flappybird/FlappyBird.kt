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
        for (i in 0 until speedUpFactor) {
            run()
        }
    }

    private fun run() {
        batch.begin()
        batch.draw(background, 0f, 0f, gdxWidth.toFloat(), gdxHeight.toFloat())

        if (gameState == GameStates.PLAYING) {
            updateScore()
            if (Gdx.input.justTouched()) {
                jump()
            }
            drawTubes()
            updateBirdY()
        } else if (gameState == GameStates.GAME_OVER) {
            reset()
        }

        updateFlapState()

        batch.draw(currentBird, birdX, birdY)
        font.draw(batch, score.toString(), 100f, 200f)

        checkCrash()

        batch.end()
    }

    private fun updateScore() {
        if (tubeX[scoringTube] < birdX) {
            score++
            if (scoringTube < numberOfTubes - 1) {
                scoringTube++
            } else {
                scoringTube = 0
            }
        }
    }

    private fun jump() { velocity = -0.2f * currentBird.height }

    private fun updateBirdY() {
        if (birdY > 0) {
            velocity += GRAVITY
            birdY -= velocity
        } else {
            gameState = GameStates.GAME_OVER
        }
    }

    private fun drawTubes() {
        for (i in 0 until numberOfTubes) {
            updateTubePosition(i)

            val topTubeY = gdxHeight / 2f + gap / 2 + tubeOpenSpaceY[i]
            val bottomTubeY = gdxHeight / 2f - gap / 2 - bottomTubeHeight.toFloat() + tubeOpenSpaceY[i]

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
        const val FPS = 30
    }
}
