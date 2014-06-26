import os
import numpy as np
from omnihack import enumerator
from sklearn.datasets.base import Bunch


def get_files(dirname, extension=".source"):
    for root, dirs, files in os.walk(dirname):
        for fname in files:
            if fname.endswith(extension):
                yield os.path.join(root, fname)


def get_category(fname):
    return os.path.basename(os.path.splitext(fname)[0])


def get_data_frame(dirname, get_data):
    category_codes = enumerator()
    x_nested = []
    y_nested = []
    fnames = [fn for fn in get_files(dirname)]
    num_files = len(fnames)
    max_num_lines = 0
    for fname in fnames:
        file_data = []
        num_lines = 0
        with open(fname) as f:
            for line in f:
                file_data.append(get_data(line))
                num_lines += 1

        if num_lines > max_num_lines:
            max_num_lines = num_lines

        category_code = category_codes[get_category(fname)]
        categories = [category_code] * len(file_data)
        x_nested.append(file_data)
        y_nested.append(categories)

    # intersperse
    x_final = []
    y_final = []
    for j in range(0, max_num_lines):
        for i in range(0, num_files):
            try:
                x_final.append(x_nested[i][j])
                y_final.append(y_nested[i][j])
            except:
                continue

    return Bunch(
        DESCR="complete set",
        data=x_final,
        target=np.array(y_final),
        target_names=category_codes.keys(),
        filenames=fnames
    )


def split_list(l, ratio):
    pivot = int(len(l) * ratio)
    list_1st_half = l[0:pivot]
    list_2nd_half = l[pivot:]
    return list_1st_half, list_2nd_half


def get_data_frames(dirname, get_data, train_test_ratio=0.75):
    category_codes = enumerator()
    x_training = []
    x_testing = []
    y_training = []
    y_testing = []
    fnames = get_files(dirname)
    for fname in fnames:
        print("Using file " + fname)
        file_data = []
        with open(fname) as f:
            for line in f:
                file_data.append(get_data(line))
        category_code = category_codes[get_category(fname)]
        categories = [category_code] * len(file_data)
        data_train, data_test = split_list(file_data, train_test_ratio)
        cat_train, cat_test = split_list(categories, train_test_ratio)
        x_training.extend(data_train)
        x_testing.extend(data_test)
        y_training.extend(cat_train)
        y_testing.extend(cat_test)

    return (
        Bunch(
            DESCR="training set (3/4)",
            data=x_training,
            target=np.array(y_training),
            target_names=category_codes.keys(),
            filenames=fnames
        ),
        Bunch(
            DESCR="testing set (1/4)",
            data=x_testing,
            target=np.array(y_testing),
            target_names=category_codes.keys(),
            filenames=fnames
        )
    )
