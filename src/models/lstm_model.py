import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


def build_lstm_model(
    lookback: int,
    n_features: int,
    horizon: int,
    units: int = 64,
    dropout: float = 0.2,
) -> tf.keras.Model:
    """
    Many-to-many LSTM: input (lookback, n_features) → output (horizon,)
    """
    model = Sequential()
    model.add(
        LSTM(
            units,
            input_shape=(lookback, n_features),
            return_sequences=False,
        )
    )
    model.add(Dropout(dropout))
    model.add(Dense(units // 2, activation="relu"))
    model.add(Dense(horizon))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mae"],
    )
    return model
