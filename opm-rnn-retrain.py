import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Dropout, LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import argparse

import os

import utils


def encode(y):
    return np_utils.to_categorical(y)

def create_model(X, y, params):
    input = Input(shape=(X.shape[1], X.shape[2]))
    lstm1 = LSTM(params["LSTM-1"], return_sequences=True)(input)
    drop1 = Dropout(params["Dropout-1"])(lstm1)
    lstm2 = LSTM(params["LSTM-2"])(drop1)
    drop2 = Dropout(params["Dropout-2"])(lstm2)
    output = Dense(y.shape[1], activation=params["activation"])(drop2)
    return Model(inputs=input, outputs=output)


def train(dataset, output):
    raw_text = open(dataset).read()
    pre_text = utils.pre_process(raw_text)
    char_map = utils.map_chars_to_int(pre_text)

    params = {
        "seq_length":80,
        "n_chars" : len(pre_text),
        "n_vocab" : len(char_map)
        }

    X = []
    y = []
    X, y = utils.prepare_dset(pre_text, char_map, params)
    params["n_patterns"] = len(X)

    X = np.reshape(X, (params["n_patterns"], params["seq_length"], 1))
    y = encode(y)

    model_params = {
        "LSTM-1" : 512,
        "LSTM-2" : 256,
        "Dropout-1" : 0.3,
        "Dropout-2" : 0.2,
        "activation": "softmax",
        "loss" : "categorical_crossentropy",
        "optimizer" : "adam",
        "epochs" : 100,
        "batch_size" : 32
    }
    model = create_model(X, y, model_params)
    filepath = os.path.join(output, "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5")
    checkpoint = ModelCheckpoint(filepath, monitor='loss',
                                 verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    model.compile(loss=model_params["loss"],
             optimizer=model_params["optimizer"])

    model.fit(X, y, epochs=model_params["epochs"],
              batch_size=model_params["batch_size"],
              callbacks=callbacks_list)

    with open("model-opm.json" , "w") as json_file:
        json_file.write(model.to_json())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",
                        help="Path of the training dataset", default="opm-lyrics.txt")
    parser.add_argument("--output",
                        help="Path where to save the weights", default="weights/")

    args = parser.parse_args()
    dset = args.dataset
    out = args.output
    train(dset, out)
