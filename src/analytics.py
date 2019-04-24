import json
import pprint

training_file = "../data/supfex-train-gold.json"
dev_file = "../data/supfex-dev-gold.json"
test_file = "../data/supfex-test-gold.json"

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


def num_sentences(f_name):
    sentence_set = set()
    gen = read_perline_json(f_name)
    for annotated_dict in gen:
        print(annotated_dict)
        print()
        sentence_set.add(annotated_dict['segment'])
    exit()

    return len(sentence_set)

def num_predicates(f_name):
    predicate_set = set()
    gen = read_perline_json(f_name)
    for annotated_dict in gen:
        for x in annotated_dict['row']:
            predicate_set.add(x)

    return len(predicate_set)


def labels_data(f_name, label_count):
    gen = read_perline_json(f_name)
    for annotated_dict in gen:
        for x in annotated_dict['col_roles']:
            if x in label_count:
                label_count[x] += 1
            else:
                label_count[x] = 1

    return label_count



ret = num_sentences(training_file)
print("Number of sentences in Training Data: ", ret)
ret = num_sentences(dev_file)
print("Number of sentences in Dev Data: ", ret)
ret = num_sentences(test_file)
print("Number of sentences in Test Data: ", ret, "\n")


ret = num_predicates(training_file)
print("Number of predicates in Training Data: ", ret)
ret = num_predicates(dev_file)
print("Number of predicates in Dev Data: ", ret)
ret = num_predicates(test_file)
print("Number of predicates in Test Data: ", ret, "\n")



a = labels_data(training_file, {})
a = labels_data(dev_file, a)
a = labels_data(test_file, a)
print("Label Distribution is:")
pprint.pprint(a)
