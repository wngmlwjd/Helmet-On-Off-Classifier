import tensorflow as tf
from tensorflow.keras import layers, models
from data_prep.utils import TARGET_SIZE

# ===============================
# model_15 에서 변형
# AveragePooling -> MaxPooling
# BatchNormalization 추가, use_bias=False 설정
# 3번째 층 커널 크기 3->5
# ===============================

def main_cnn(selected_color):
    w, h = TARGET_SIZE
    d = 3 if selected_color == 'color' else 1

    inputs = tf.keras.Input(shape=(h, w, d))

    # 1번째 Conv + MaxPooling
    x = layers.Conv2D(6, 5, padding="same", activation="relu")(inputs)
    x = layers.MaxPooling2D(pool_size=2)(x)

    # 2번째 Conv + BN + MaxPooling
    x = layers.Conv2D(16, 5, activation="relu", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    
    # 3번째 Conv + BN
    x = layers.Conv2D(16, 5, activation="relu", use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    # Flatten 후 Dense
    x = layers.Flatten()(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    return models.Model(inputs, outputs)
