import numpy as np
from keras.engine.saving import model_from_json
import argparse

import sys

import utils

def generate(dataset, weights, json_path):
    raw_text = open(dataset).read()
    pre_text = utils.pre_process(raw_text)
    char_map = utils.map_chars_to_int(pre_text)
    int_map  = utils.map_int_to_char(pre_text)


    params = {
        "seq_length":80,
        "n_chars" : len(pre_text),
        "n_vocab" : len(char_map)
    }

    testX, _ = utils.prepare_dset(pre_text, char_map, params)
    params["n_patterns"] = len(testX)


    with open(json_path, 'r') as json_file:
        json_model = json_file.read()

    model = model_from_json(json_model)
    model.load_weights(weights)
    model.compile(loss="categorical_crossentropy", optimizer="adam")

    start = np.random.randint(0 , len(testX) - 1)
    sentence = testX[start]
    output = []
    for i in range(1500):
        x = np.reshape(sentence, (1, len(sentence), 1))
        x = x / float(params["n_vocab"])
        prediction = model.predict(x, verbose=0)
        index = np.argmax(prediction)
        result = int_map[index]
        output.append(result)
        sentence.append(index)
        sentence = sentence[1:len(sentence)]

    print("".join(output))





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",
                        help="Path of the training dataset", default="opm-lyrics.txt")
    parser.add_argument("--weights",
                        help="Path of the saved weights to load",
                        default="weights/model-opm-weights.hdf5")
    parser.add_argument("--json",
                        help="Path of the json model to load",
                        default="model-opm.json")

    args = parser.parse_args()
    dataset = args.dataset
    weights = args.weights
    json_path = args.json
    generate(dataset, weights, json_path)