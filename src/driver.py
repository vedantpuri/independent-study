# ---------- IMPORTS

import os
import sys
import json
import copy
import random
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt

from constants import *
from ConfigManager import *
from sklearn.metrics import *
from potential_models import *


# ---------- I/O UTILS

# Generator for lines in json file
def read_perline_json(json_file_path):
    """
    Read per line JSON and yield.
    :param json_file: Just an open file. file-like with a next method.

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

def plot_graph(x_axis, y_axis, x_label, y_label, fig_name):
    """
    Plots the graph and saves it as ../Figures/$fig_name$
    :param x_axis:      Horizontal axis values
    :param y_axis:      Vertical axis values
    :param x_label:     Horizontal axis label
    :param y_label:     Vertical axis label
    :param fig_name:    Name of figure to save it as
    """
    graph_dir = '../Figures/'
    if not os.path.isdir(graph_dir):
        os.makedirs(graph_dir)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.plot(x_axis, y_axis)
    plt.savefig(graph_dir + fig_name)
    plt.close()

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
    formatted_labels = []
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

            # Add example in the format -> (p, a)
            formatted_examples += [(pred_map[predicate],
                                     arg_map[argument_at_index])]
            formatted_labels +=  [role_map[role_at_index]]

    return pred_map, arg_map, role_map, formatted_examples, formatted_labels


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

def dict_to_list(d):
    """
    Convert dictionary to list. Keys(Assumed integers)
    :param d:   The dictionary to be converted

    :return:    A list where the k'th index has d[k]
    """
    indexed_vals = []
    for k in d:
        indexed_vals += [d[k]]

    return indexed_vals

# ---------- TRAINING MECHANISM

def batcher(samples, labels, batch_size):
    """
    Provide batch of samples of required size
    :param samples:         The samples to be batched
    :param labels:          Corresponding labels of the samples (also batched)
    :param batch_size:      Size of each batch

    :return:                Yield a batch of samples, labels
    """
    for i in range(0, len(samples), batch_size):
        yield samples[i : i + batch_size], labels[i : i + batch_size]

def combine_shuffle(list_a, list_b):
    """
    Shuffle two lists while maintaining correspondence
    :param list_a:  One of the lists
    :param list_b:  The other list

    :return:        Shuffled versions of the lists (correspondence maintained)
    """
    combined = list(zip(list_a, list_b))
    random.shuffle(combined)
    ret_a, ret_b = zip(*combined)

    return ret_a, ret_b


def train(config, train_data, model, loss_fn, optimizer, dev_data):
    """
    Training function
    :param config:          Config data containing epochs,batch_size,check_every
    :param train_data:      The data to train on
    :param model:           The model to be trained
    :param loss_fn:         Loss Function to help in training [data + labels]
    :param optimizer:       Optimizer function eg. SGD
    :param dev_data:        Dev data for checking performance [data + labels]

    :return: a trained model
    """
    train_samples, train_labels = zip(*train_data)
    dev_samples, dev_labels = zip(*dev_data)

    dev_f1_scores = []
    best_dev_f1 = - np.inf
    iteration = 0
    epoch_losses = []
    iter_losses = []
    all_losses = []

    for epoch in range(config.epochs):
        # Shuffle training data here
        train_samples, train_labels = combine_shuffle(train_samples,
                                                                   train_labels)
        # Get a batch
        for samples, labels in batcher(train_samples, train_labels,
                                                             config.batch_size):

            # Remember to clear out gradients for each instance
            model.zero_grad()

            # Obtain the probabilities
            probs = model(samples)

            # Computing Loss (LEARNING)
            target = torch.LongTensor(labels)
            loss = loss_fn(probs, target)
            iter_losses += [loss.item()]
            loss.backward()
            optimizer.step()

            # Check every few iterations on the dev dev_set
            if iteration % config.check_every == 0:
                predictions = model_predict(dev_samples, model)
                res = evaluate_performance(f1_score, predictions, dev_labels,
                                                            {"average":'micro'})
                dev_f1_scores += [res]
                if res > best_dev_f1:
                    best_dev_f1 = res
                    best_params = copy.deepcopy(model.state_dict())

            iteration += 1
        epoch_losses += [np.mean(iter_losses)]
        all_losses += iter_losses
        iter_losses = []

    return best_params, epoch_losses, all_losses, dev_f1_scores


# ---------- PREDICTIONS + EVAL

def model_predict(data, model):
    """
    Evaluate the model on some data
    :param data:    data set to evaluate the model on
    :param model:   the model to make use while making predictions on ``data''

    :return:        list of predictions
    """
    predictions = []
    with torch.no_grad():
        probs = model(data, True)
        for idx in range(len(data)):
            predictions += [demistify_predictions(probs[idx].flatten())]

    return predictions

def demistify_predictions(probs_tensor, maximum=True):
    """
    Make a prediction based on the scores/probabilities
    :param probs_tensor:    Tensor containing the probabilities of each label
    :param maximum:         Whether to chose maximum score or minimum

    :return:                Index (label) having MAX score/probability
    """
    if maximum:
        return torch.max(probs_tensor, 0)[1].item()
    else:
        return torch.min(probs_tensor, 0)[1].item()

def evaluate_performance(metric_fn, predictions, labels, extra_args={}):
    """
    Provide the requested metric on the arguments given
    :param metric_fn:       The sklearn metric fucntion to be used
    :param predictions:     Predictions from the model
    :param labels:          Gold labels
    :param extra_args:      Additional args for the metric fn

    :return:                The value returned from the metric function
    """
    # standard args for every sklearn metric fn
    sklearn_std_args = {"y_pred": predictions, "y_true": labels}
    final_args = {**sklearn_std_args, **extra_args}

    return metric_fn(**final_args)


def random_predictor(data, label_size):
    """
    :param data:        Data to be predicted on
    :param label_size:  Number of labels available for classification

    :return:            Random predictions for the data
    """
    preds = []
    for elem in data:
        preds += [np.random.randint(0, label_size)]

    return preds

def majority_predictor(data, majority_label):
    """
    :param data:            Data to be predicted on
    :param majority_label:  The label present in dominance in the ENTIRE dataset

    :return:                Majority label predictions for every sample in data
    """
    preds = []
    for elem in data:
        preds += [majority_label]

    return preds

# calculate precision, recall, f1
def perform_benchmarking(predictions, labels):
    """
    Evaluate predictions and labels on these metrics: precision, recall, f1
    :param predictions:     Predictions made by some model on some data
    :param labels:          Labels of that data

    :return:                A triple: precision, recall, f1 (In that order)
    """
    p = evaluate_performance(precision_score, predictions, labels,
                                                            {"average":'micro'})
    r = evaluate_performance(recall_score, predictions, labels,
                                                            {"average":'micro'})
    f1 = evaluate_performance(f1_score, predictions, labels,
                                                            {"average":'micro'})

    return p, r, f1

# ---------- MAIN EXECUTION

# https://stackoverflow.com/a/1518632/9319690
def most_common(lst):
    """
    Return the majority element in a list
    :param lst:     The collection of items [list assumed]

    :return:        The element that occurs most frequently
    """
    return max(set(lst), key=lst.count)

# Driver main
if __name__ == "__main__":
    # Load configuration
    assert(len(sys.argv) == 2)
    assert(os.path.exists(sys.argv[1]))
    configuration = ConfigManager(sys.argv[1])
    print(CONFIG_PARSE_BEGIN)
    configuration.parse_config()
    print(CONFIG_PARSE_SUCC)

    # Reading in tools
    print(ENCODING_BEGIN)
    samples_generator_train = read_perline_json(configuration.train_file_path)
    samples_generator_dev = read_perline_json(configuration.dev_file_path)
    samples_generator_test = read_perline_json(configuration.test_file_path)

    # Form Mappings
    pred2idx, arg2idx, role2idx, train_set, train_labels = accumulate_mapping(
                                                        samples_generator_train,
                                                        {}, {}, {})
    pred2idx, arg2idx, role2idx, dev_set, dev_labels = accumulate_mapping(
                                                        samples_generator_dev,
                                                    pred2idx, arg2idx, role2idx)
    pred2idx, arg2idx, role2idx, test_set, test_labels = accumulate_mapping(
                                                        samples_generator_test,
                                                    pred2idx, arg2idx, role2idx)

    idx2pred, idx2arg, idx2role = form_reverse_mapping(pred2idx, arg2idx,
                                                                   role2idx)
    assert(len(pred2idx) == len(idx2pred))
    assert(len(arg2idx) == len(idx2arg))
    assert(len(role2idx) == len(idx2role))
    print(ENCODING_SUCC)

    # Learning
    if configuration.train_switch:
        print(DIVIDER)
        print(TRAIN_INV)
        model = RolePredictor(len(pred2idx), len(arg2idx), len(role2idx),
                                configuration.drop_p, configuration.embed_size,
                                                   configuration.linearity_size)
        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=configuration.learn_rate)
        training_data = zip(train_set, train_labels)
        dev_data = zip(dev_set, dev_labels)
        print(TRAIN_INV_SUCC)

        print(TRAIN_BEGIN)
        results = train(configuration, training_data, model, loss_function,
                                                            optimizer, dev_data)
        print(TRAIN_SUCC)

        best_params, epoch_losses, all_loss, dev_f1_scores = results
        plot_graph(range(len(dev_f1_scores)), dev_f1_scores, "Iteration",
                                                "F1 score on dev set", "dev_f1")
        plot_graph(range(len(epoch_losses)), epoch_losses, "Epoch",
                                        "Training Loss", "epoch_train_loss")
        plot_graph(range(len(all_loss)), all_loss, "Iteration",
                                        "Training Loss", "iter_train_loss")
        print(GRAPH_PLOT_MSG)
        torch.save({
                'p2i': len(pred2idx),
                'a2i': len(arg2idx),
                'r2i': len(role2idx),
                'drop': configuration.drop_p,
                'embed': configuration.embed_size,
                'linearity': configuration.linearity_size,
                'model_state_dict': best_params,
                }, configuration.model_file)
        print(MODEL_DUMP_MSG, configuration.model_file)

    # Predicting + Benchmarking
    if configuration.predict_switch:
        print(DIVIDER)
        print(TEST_INV)
        loader = torch.load(configuration.model_file)
        trained_model = RolePredictor(loader['p2i'], loader['a2i'],
                                      loader['r2i'], loader['drop'],
                                      loader['embed'], loader['linearity'])
        trained_model.load_state_dict(loader['model_state_dict'])

        print(TEST_PREDS_BEGIN)
        predictions = model_predict(test_set, trained_model)
        print(PREDS_END)

        if configuration.dump_test_preds:
            print(WRITE_PREDS_BEGIN)
            write_predictions(TEMP_PREDICTION_FILE,
                                zip(predictions,test_labels), idx2role)
            print(WRITE_PREDS_SUCC)

        index_labels = dict_to_list(idx2role)
        print(TEST_REPORT_MSG)
        print(evaluate_performance(classification_report, predictions,
                                   test_labels, {"target_names": index_labels}))

        print(TRAIN_PREDS_BEGIN)
        predictions = model_predict(train_set, trained_model)
        print(PREDS_END)
        p, r, f1 = perform_benchmarking(predictions, train_labels)
        print(TRAIN_METRICS)
        print(PR, p)
        print(RE, r)
        print(F1, f1)
        print()

        print(DEV_PREDS_BEGIN)
        predictions = model_predict(dev_set, trained_model)
        print(PREDS_END)
        p, r, f1 = perform_benchmarking(predictions, dev_labels)
        print(DEV_METRICS)
        print(PR, p)
        print(RE, r)
        print(F1, f1)
        print()

        if configuration.run_baselines:
            print(DIVIDER)
            print(BASE_BEGIN)
            # Random Prediction Baseline [TEST SET]
            print(RAND_PRED)
            predictions = random_predictor(test_set, len(role2idx))
            print(PREDS_END)
            p, r, f1 = perform_benchmarking(predictions, test_labels)
            print(RAND_METRIC_MSG)
            print(PR, p)
            print(RE, r)
            print(F1, f1)
            print()

            # Majority Label Baseline [TEST SET]
            majority_label = most_common(train_labels + dev_labels +
                                                                    test_labels)
            print(MAJ_PRED)
            predictions = random_predictor(test_set, majority_label)
            print(PREDS_END)
            p, r, f1 = perform_benchmarking(predictions, test_labels)
            print(MAJ_METRIC_MSG)
            print(PR, p)
            print(RE, r)
            print(F1, f1)
            print()

            print(BASE_END)
    print(TERMINATED)
