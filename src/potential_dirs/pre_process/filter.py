from data_io import *

samples_generator_train = read_perline_json(TRAINING_FILE_PATH)
samples_generator_DEV = read_perline_json(DEV_FILE_PATH)
samples_generator_train = read_perline_json(TEST_FILE_PATH)

count = 0
training_examples = []

predicate_to_idx = {}
argument_to_idx = {}
role_to_idx = {}
for sample in samples_generator_train:
    # Only one predicate per sample
    print(sample)
    predicate = sample[PREDICATE_KEY][0]
    arguments = sample[ARGUMENT_KEY]
    for index in range(len(arguments)):
        # Check if predicate in map
        if predicate not in predicate_to_idx:
            predicate_to_idx[predicate] = len(predicate_to_idx)

        argument_at_index = sample[ARGUMENT_KEY][index]
        role_at_index = sample[ROLE_KEY][index]

        # check if argument in map
        if argument_at_index not in argument_to_idx:
            argument_to_idx[argument_at_index] = len(argument_to_idx)

        # check if role in map
        if role_at_index not in role_to_idx:
            role_to_idx[role_at_index] = len(role_to_idx)


        # ------- CANNOT USE DICT DUE TO COLLISIOINS

        # tup = (sample[PREDICATE_KEY][0], sample[ARGUMENT_KEY][index])
        # if tup in training_examples and training_examples[tup] != sample[ROLE_KEY][index]:
        #     # print("true")
        #     # print(tup,":", training_examples[tup])
        #     print(predicate, argument_at_index, training_examples[tup], sample[ROLE_KEY][index])
        #     # exit()
        #
        # if tup not in training_examples:
        #     training_examples[tup] = sample[ROLE_KEY][index]

        # ------- CANNOT USE DICT DUE TO COLLISIOINS

        training_examples += [(predicate_to_idx[predicate],
                              argument_to_idx[argument_at_index],
                              role_to_idx[role_at_index])]

        count += 1


for i in training_examples:
    print(i)
print(len(training_examples))
print(count)



# class DataFromatter:
#     def __init__(self, train_file_path, dev_file_path, test_file_path):
#         self.training_sample_generator = read_perline_json(train_file_path)
#         self.dev_sample_generator = read_perline_json(dev_file_path)
#         self.test_sample_generator = read_perline_json(test_file_path)
#
#         self.predicate_to_idx = {}
#         self.argument_to_idx = {}
#         self.role_to_idx = {}
#
#
#     def obtain_mappings():
#
#         pass
