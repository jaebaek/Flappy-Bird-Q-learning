import tensorflow as tf
import numpy as np
import asyncio
import websockets
from keras import Sequential

PREDICT_TAG = "predict:"
PREDICT_TAG_LEN = len(PREDICT_TAG)
TRAIN_TAG = "train:"
TRAIN_TAG_LEN = len(TRAIN_TAG)


def create_and_compile_model():
    """Returns the compiled (but untrained) model.

    Returns:
        tf.keras.Model: The model that will be trained to predict reward.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(7,)),
        tf.keras.layers.Dense(units = 4, activation=tf.nn.relu),
        tf.keras.layers.Dense(units = 2)
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model


def predict(model: Sequential, data):
    input_array = np.fromstring(data, sep=',').reshape(1, 7)
    result = model.predict(input_array, verbose=False).flatten()
    return ','.join(map(str, result))


def train(model: Sequential, data):
    parsed = np.fromstring(data, sep=',')
    x = np.array(parsed[:7], dtype=float).reshape(1, 7)
    y = np.array(parsed[7:9], dtype=float).reshape(1, 2)
    model.fit(x, y, epochs=1)


async def handler(websocket):
    """
    Handles the WebSocket connection.

    Receives messages from the client, processes them, and sends a response.
    The connection is closed when the client sends an "exit" message.
    """
    print(f"Client connected from {websocket.remote_address}")

    model = create_and_compile_model()
    try:
        while True:
            # Wait for a message from the client
            message = await websocket.recv()
            print(f"< {message}")

            if message.lower() == "exit":
                print("Client sent 'exit' message. Closing connection.")
                await websocket.send("Goodbye!")
                break

            if PREDICT_TAG in message:
                response = f"predict:{predict(model, message[PREDICT_TAG_LEN:])}"
            elif TRAIN_TAG in message:
                train(model, message[TRAIN_TAG_LEN:])
                response = "Train done"
            else:
                response = f"Processed message: '{message}'"

            # Send the response back to the client
            await websocket.send(response)
            print(f"> {response}")

    except websockets.exceptions.ConnectionClosed as e:
        print(f"Connection closed unexpectedly: {e}")
    finally:
        print(f"Client from {websocket.remote_address} disconnected.")


async def main():
    """
    Starts the WebSocket server.
    """
    host = "localhost"
    port = 8765
    async with websockets.serve(handler, host, port):
        print(f"WebSocket server started at ws://{host}:{port}")
        await asyncio.Future()  # Run forever


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Server is shutting down.")
