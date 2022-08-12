def save_dict_to_csv(dict, filename):
    with open(filename + '.csv', 'w') as f:
        for key in dict.keys():
            f.write("%s,%s\n" % (key, dict[key]))
