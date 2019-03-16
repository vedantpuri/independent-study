from data_io import *

x = read_perline_json(TRAINING_FILE_PATH)
# print(data)
count = 0
d = {}
for i in x:
    # print(i['col_roles'], i['col'], i['row'])
    for index in range(len(i['col'])):
        tup = (i['row'][0], i['col'][index])
        if tup not in d:
            d[tup] = i['col_roles'][index]
    # print(i)

    count += 1
    # exit()
# print(count)
print(d)
