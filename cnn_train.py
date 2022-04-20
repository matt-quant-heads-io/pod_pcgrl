import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
import os
import pandas as pd
import numpy as np
from keras.utils import np_utils


obs_size = 5
data_size = 1
version = 3

pod_root_path = f'pod_exp_traj_obs_{obs_size}_ep_len_77_goal_size_{data_size}'

tile_map = {
    0: "empty",
    1: "solid",
    2: "player",
    3: "key",
    4: "door",
    5: "bat",
    6: "scorpion",
    7: "spider"
}

dfs = []
X = []
y = []

for file in os.listdir(pod_root_path):
    print(f"compiling df {file}")
    df = pd.read_csv(f"{pod_root_path}/{file}")
    dfs.append(df)

df = pd.concat(dfs)

df = df.sample(frac=1).reset_index(drop=True)
y_true = df[['target']]
y = np_utils.to_categorical(y_true)
df.drop('target', axis=1, inplace=True)
y = y.astype('int32')

for idx in range(len(df)):
    x = df.iloc[idx, :].values.astype('int32').reshape((obs_size, obs_size, 8))
    X.append(x)

X = np.array(X)


(state, action)
[([['x','b','c',' '],
  ['x','b','b',' ']], 'b')
([['x','b','c',' '],
  ['x','b','c',' ']], 'x')
([['x','b','c',' '],
  ['x','b','c',' ']], 'c')
([['x','b','c',' '],
  ['x','b','c',' ']], 'b')
([['x','b','c',' '],
  ['x','b','c',' ']], '')]


model_abs_path = f"model_obs_{obs_size}_goal_size_{data_size}_model_num_{version}.h5"

# Model for obs 5
if obs_size == 5:
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', input_shape=(obs_size, obs_size, 8), padding="SAME"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding="SAME"),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding="SAME"),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(8, activation='softmax')
    ])
elif obs_size == 9:
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', input_shape=(obs_size, obs_size, 8), padding="SAME"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding="SAME"),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(8, activation='softmax')
    ])
elif obs_size == 15:
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', input_shape=(obs_size, obs_size, 8)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(8, activation='softmax')
    ])

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=[tf.keras.metrics.CategoricalAccuracy()])
mcp_save = ModelCheckpoint(model_abs_path, save_best_only=True, monitor='categorical_accuracy', mode='max')
history = model.fit(X, y, epochs=500, steps_per_epoch=64, verbose=2, callbacks=[mcp_save])