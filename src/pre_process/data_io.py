# Import necessary modules
import json
from constants import *

# Utility functions
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


# Misc playground to test upper functions
# x = read_perline_json(TRAINING_FILE_PATH)
# # print(data)
# for i in x:
#     print(i)
#     # exit()
