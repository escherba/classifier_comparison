import os
import numpy as np
from omnihack import enumerator
from sklearn.datasets.base import Bunch
from sklearn.cross_validation import train_test_split


def get_files(dirname, extension=".source"):
    for root, dirs, files in os.walk(dirname):
        for fname in files:
            if fname.endswith(extension):
                yield os.path.join(root, fname)


def get_category(fname):
    return os.path.basename(os.path.splitext(fname)[0])


def get_data_frame(dirname, get_data, cat_filter=None):
    category_codes = enumerator()
    x_nested = []
    y_nested = []
    fnames = [fn for fn in get_files(dirname)]
    num_files = len(fnames)
    max_num_lines = 0
    for fname in fnames:
        cat_name = get_category(fname)
        if (cat_filter is not None) and (cat_name not in cat_filter):
            continue
        file_data = []
        num_lines = 0
        with open(fname) as f:
            for line in f:
                file_data.append(get_data(line))
                num_lines += 1

        if num_lines > max_num_lines:
            max_num_lines = num_lines

        category_code = category_codes[cat_name]
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


def get_data_frames(dirname, get_data, test_size=0.25, cat_filter=None):
    category_codes = enumerator()
    x_training = []
    x_testing = []
    y_training = []
    y_testing = []
    fnames = get_files(dirname)
    for fname in fnames:
        cat_name = get_category(fname)
        if (cat_filter is not None) and (cat_name not in cat_filter):
            continue
        file_data = []
        with open(fname) as f:
            for line in f:
                file_data.append(get_data(line))
                category_code = category_codes[cat_name]
        categories = [category_code] * len(file_data)

        data_train, data_test, cat_train, cat_test = \
            train_test_split(file_data, categories, test_size=test_size)

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
