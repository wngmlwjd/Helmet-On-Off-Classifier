import tensorflow as tf
from tensorflow.keras import layers, models
from data_prep.utils import TARGET_SIZE

# ===============================
# model_14 에서 변형
# 3번째 층 추가
# ===============================

def main_cnn(selected_color):
    w, h = TARGET_SIZE
    d = 3 if selected_color == 'color' else 1

    inputs = tf.keras.Input(shape=(h, w, d))

    # 1번째 Conv + AveragePooling
    x = layers.Conv2D(6, 5, padding="same", activation="relu")(inputs)
    x = layers.AveragePooling2D(pool_size=2)(x)

    # 2번째 Conv + AveragePooling
    x = layers.Conv2D(16, 5, activation="relu")(x)
    x = layers.AveragePooling2D(pool_size=2)(x)
    
    # 3번째 Conv 
    x = layers.Conv2D(16, 3, activation="relu")(x)

    # Flatten 후 Dense
    x = layers.Flatten()(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    return models.Model(inputs, outputs)
