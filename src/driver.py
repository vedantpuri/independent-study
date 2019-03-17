import json
from config import *
from potential_models import *


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

            # ------- CANNOT USE DICT DUE TO COLLISIOINS (same (p, a) => l1, l2)

            # tup = (sample[PREDICATE_KEY][0], sample[ARGUMENT_KEY][index])
            # if tup in training_examples and
            #               training_examples[tup] != sample[ROLE_KEY][index]:
            #     # print("true")
            #     # print(tup,":", training_examples[tup])
            #     print(predicate, argument_at_index, training_examples[tup],
            #           sample[ROLE_KEY][index])
            #     # exit()
            #
            # if tup not in training_examples:
            #     training_examples[tup] = sample[ROLE_KEY][index]

            # ------- CANNOT USE DICT DUE TO COLLISIOINS (same (p, a) => l1, l2)

            # Add example in the format -> (p, a, r)
            formatted_examples += [(pred_map[predicate],
                                       arg_map[argument_at_index],
                                       role_map[role_at_index])]

    return pred_map, arg_map, role_map, formatted_examples






# Driver main
if __name__ == "__main__":
    # Pre-Processing
    samples_generator_train = read_perline_json(TRAINING_FILE_PATH)
    samples_generator_dev = read_perline_json(DEV_FILE_PATH)
    samples_generator_test = read_perline_json(TEST_FILE_PATH)

    pred2idx, arg2idx, role2idx, train_set = accumulate_mapping(
                                                samples_generator_train,
                                                {}, {}, {})
    # print(len(pred2idx), len(arg2idx), len(role2idx))

    pred2idx, arg2idx, role2idx, dev_set = accumulate_mapping(
                                                samples_generator_dev,
                                                pred2idx, arg2idx, role2idx)
    # print(len(pred2idx), len(arg2idx), len(role2idx))

    pred2idx, arg2idx, role2idx, test_set = accumulate_mapping(
                                                samples_generator_test,
                                                pred2idx, arg2idx, role2idx)
    # print(len(pred2idx), len(arg2idx), len(role2idx))

    # Learning
    model = RolePredictor(len(pred2idx), len(arg2idx), len(role2idx))
    print(model)
