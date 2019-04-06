# ---------- IMPORTS

import os
import sys
import json
import random
import torch.optim as optim
from config import *
from potential_models import *
from sklearn.metrics import *


# ---------- I/O UTILS

# Generator for lines in json file
def read_perline_json(json_file_path):
    """
    Read per line JSON and yield.
    :param json_file: Just a open file. file-like with a next method.

    :return:          yield one json object.
    """
    json_file = open(json_file_path, "r")
    for json_line in json_file:

        # Try to manually skip bad chars.
        # https://stackoverflow.com/a/9295597/3262406
        try:
            f_dict = json.loads(json_line.replace('\r\n', '\\r\\n'),
                                encoding='utf-8')
            yield f_dict

        # Skip case which crazy escape characters.
        except ValueError:
            yield {}

def print_collection(bag, dictionary=False, delimeter=":"):
    """
    Print the collection one element per line
    :param bag:         collection to be printed
    :param dictionary:  whether the collection is a dictionary
    :param delimeter:   delimeter to be printed b/w key and value
    """
    for element in bag:
        if dictionary:
            print(element,delimeter, bag[element])
        else:
            print(element)

def write_predictions(output_file, predictions, idx2role):
    """
    Write predictions to a file in the form of "y_pred   y_gold" in each line
    :param output_file:     File to write the predictions to
    :param predictions:     list containing tuples of the form (y_pred, y_gold)
    :param idx2role:        Mapping role -> integer
    """
    f = open(output_file,"w+")
    for prediction, gold_label in predictions:
        f.write(idx2role[prediction] + "\t" + idx2role[gold_label] + "\n")
    f.close()

def read_prediction_file(input_file):
    """
    Read predictions from a file in the form of "y_pred   y_gold" in each line
    :param input_file:     File to read the predictions from

    :return:               Lists of predictions and corresponding true labels
    """
    with open(input_file,"r") as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    y_pred = []
    y_gold = []
    for element in content:
        line = element.split("\t")
        y_pred += [line[0]]
        y_gold += [line[1]]
    assert(len(y_pred) == len(y_gold))

    return y_pred, y_gold


# ---------- PRE-PROCESSING UTILS

def accumulate_mapping(line_gen, pred_map, arg_map, role_map):
    """
    Update respective mappings + return formatted data-set
    :param line_gen:    generator object for the respective file
    :param pred_map:    dict that maps predicates to integers
    :param arg_map:     dict that maps arguments to integers
    :param role_map:    dict that maps roles to integers

    :return: updated pred_map, arg_map, role_map, formatted data-set
    """
    formatted_examples = []
    for sample in line_gen:
        # Only one predicate per sample
        predicate = sample[PREDICATE_KEY][0]
        arguments = sample[ARGUMENT_KEY]
        for index in range(len(arguments)):
            # Check if predicate in map
            if predicate not in pred_map:
                pred_map[predicate] = len(pred_map)

            argument_at_index = sample[ARGUMENT_KEY][index]
            role_at_index = sample[ROLE_KEY][index]

            # check if argument in map
            if argument_at_index not in arg_map:
                arg_map[argument_at_index] = len(arg_map)

            # check if role in map
            if role_at_index not in role_map:
                role_map[role_at_index] = len(role_map)

            # Add example in the format -> (p, a, r)
            formatted_examples += [(pred_map[predicate],
                                       arg_map[argument_at_index],
                                       role_map[role_at_index])]

    return pred_map, arg_map, role_map, formatted_examples


def form_reverse_mapping(pred_map, arg_map, role_map):
    """
    Reverse respective mappings
    :param pred_map:    dict that maps predicates to integers
    :param arg_map:     dict that maps arguments to integers
    :param role_map:    dict that maps roles to integers

    :return: reversed pred_map, arg_map, role_map [eg. rev_role = idx -> role]
    """
    pred_rev_map, arg_rev_map, role_rev_map = {}, {}, {}

    # Reverse predicate map
    for k in pred_map:
        pred_rev_map[pred_map[k]] = k

    # Reverse argument map
    for k in arg_map:
        arg_rev_map[arg_map[k]] = k

    # Reverse role map
    for k in role_map:
        role_rev_map[role_map[k]] = k

    return pred_rev_map, arg_rev_map, role_rev_map


# ---------- TRAINING MECHANISM

def batcher(samples, batch_size):
    for i in range(0, len(samples), batch_size):
        yield samples[i : i + batch_size]

def combine_shuffle(list_a, list_b):
    combined = list(zip(list_a, list_b))
    random.shuffle(combined)
    ret_a, ret_b = zip(*combined)

    return ret_a, ret_b


def train(num_epochs, training_data, model, loss_fn, optimizer):
    """
    Reverse respective mappings
    :param num_epochs:      Number of epochs to go over the training data
    :param training_data:   The data to train on
    :param model:           The model to be trained
    :param loss_fn:         Loss Function to help in training
    :param optimizer:       Optimizer function eg. SGD

    :return: a trained model
    """
    for epoch in range(num_epochs):
        # shuffle training data here
        # get a batch
        for pred, arg, role in training_data:

            # REMEMBER to clear out gradients for each instance
            model.zero_grad()

            # Obtain the probabilities
            probs = model((pred, arg))

            # Computing Loss (LEARNING)
            target = torch.LongTensor([role])
            loss = loss_fn(probs, target)
            loss.backward()
            optimizer.step()

    return model


# ---------- PREDICTIONS + EVAL

def test_performance(data, model):
    """
    Every now and then evaluate the model on some data
    :param data:    data set to evaluate the model on
    :param model:   the model to make use while making predictions on ``data''

    :return: list of tuples of the form (y_pred, y_true)
    """
    preds_labels = []
    with torch.no_grad():
        for pred, arg, role in data:
            probs = model((pred, arg))
            preds_labels += [(demistify_predictions(probs[0]), role)]

    return preds_labels

def demistify_predictions(probs_tensor):
    """
    Make a prediction based on the scores/probabilities
    :param probs_tensor:    Tensor containing the probabilities of each label

    :return:                Index (label) having MAX score/probability
    """
    max_val = MIN_INT
    index = 0
    m_id = 0
    for val in probs_tensor:
        if val > max_val:
            max_val = val
            m_id = index
        index += 1

    return m_id

def evaluate_performance(metric_fn, **kwargs):
    """
    Print the requested metric on the arguments given
    :param metric_fn:   The sklearn metric fucntion to be used
    :param **kwargs:    The arguments of the metric function
    """
    print(metric_fn(**kwargs))


# ---------- MAIN EXECUTION

# Driver main
if __name__ == "__main__":

    # Playground

    # m = [i for i in range(100)]
    # for batch in batcher(m, 10):
    #     print(len(batch))
    # a = ["one", "two", "three", "four"]
    # b = [1,2,3,4]
    # print(a)
    # print(b)
    # print("////////////////")
    # a, b = combine_shuffle(a, b)
    # print(a)
    # print(b)
    # exit()



    # Pre-Processing
    assert(os.path.exists(TRAINING_FILE_PATH))
    assert(os.path.exists(DEV_FILE_PATH))
    assert(os.path.exists(TEST_FILE_PATH))

    samples_generator_train = read_perline_json(TRAINING_FILE_PATH)
    samples_generator_dev = read_perline_json(DEV_FILE_PATH)
    samples_generator_test = read_perline_json(TEST_FILE_PATH)

    pred2idx, arg2idx, role2idx, train_set = accumulate_mapping(
                                                samples_generator_train,
                                                {}, {}, {})
    pred2idx, arg2idx, role2idx, dev_set = accumulate_mapping(
                                                samples_generator_dev,
                                                pred2idx, arg2idx, role2idx)
    pred2idx, arg2idx, role2idx, test_set = accumulate_mapping(
                                                samples_generator_test,
                                                pred2idx, arg2idx, role2idx)

    idx2pred, idx2arg, idx2role = form_reverse_mapping(pred2idx, arg2idx,
                                                                    role2idx)

    assert(len(pred2idx) == len(idx2pred))
    assert(len(arg2idx) == len(idx2arg))
    assert(len(role2idx) == len(idx2role))


    # Learning
    model = RolePredictor(len(pred2idx), len(arg2idx), len(role2idx))


    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    trained_model = train(NUM_EPOCHS, train_set, model, loss_function, optimizer)


    preds_labels = test_performance(dev_set, trained_model)
    write_predictions(TEMP_PREDICTION_FILE, preds_labels, idx2role)

    y_pred, y_gold = read_prediction_file(TEMP_PREDICTION_FILE)

    if DESTROY:
        os.remove(TEMP_PREDICTION_FILE)

    # metric_args = {"y_pred": y_pred, "y_true": y_gold, "average": None}
    metric_args = {"y_pred": y_pred, "y_true": y_gold, "normalize": True}
    evaluate_performance(accuracy_score, **metric_args)
