package com.simple.flappybird

import com.badlogic.gdx.utils.Disposable
import kotlinx.coroutines.*
import kotlinx.coroutines.channels.Channel
import okhttp3.*
import okio.ByteString

class SimpleWebSocketListener : WebSocketListener() {
    val messageChannel = Channel<String>()

    override fun onOpen(webSocket: WebSocket, response: Response) {
        println("Connection Opened!")
    }

    override fun onMessage(webSocket: WebSocket, text: String) {
        println("Receiving: $text")
        if ("predict:" in text) {
            messageChannel.trySend(text.removePrefix("predict:"))
        }
        if (text == "Goodbye!") {
            webSocket.close(1000, "Client initiated close")
        }
    }

    override fun onMessage(webSocket: WebSocket, bytes: ByteString) {
        println("Receiving bytes: " + bytes.hex())
    }

    override fun onClosing(webSocket: WebSocket, code: Int, reason: String) {
        println("Closing: $code / $reason")
    }

    override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
        println("Error: " + t.message)
    }
}

fun main() {
    NeuralNetworkWrapper().test()
}

class NeuralNetworkWrapper: Disposable {
    private val client = OkHttpClient()

    private val listener = SimpleWebSocketListener()

    private val webSocket by lazy {
        val request = Request.Builder().url("ws://localhost:8765").build()
        client.newWebSocket(request, listener)
    }

    fun test() {
        // Use Coroutines for asynchronous sending
        runBlocking {
            launch {
                for (i in 1..5) {
                    val message = "Data packet $i"
                    println("Sending: $message")
                    webSocket.send(message)
                    delay(2000) // Wait 2 seconds before sending the next message
                }
                println("Sending: exit")
                webSocket.send("exit")
            }
        }

        // The client will shut down after sending the exit message and receiving a response.
        client.dispatcher.executorService.shutdown()
    }

    suspend fun predict(input: DoubleArray): DoubleArray {
        webSocket.send("predict:${input.joinToString(",")}")
        return listener.messageChannel.receive().split(",").map { it.toDouble() }.toDoubleArray()
    }

    fun train(input: DoubleArray, y: DoubleArray) {
        webSocket.send("train:${input.joinToString(",")},${y.joinToString(",")}")
    }

    override fun dispose() {
        webSocket.send("exit")
        client.dispatcher.executorService.shutdown()
    }
}
