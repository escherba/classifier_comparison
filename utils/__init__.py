import csv


def to_csv(fname, data):
    with open(fname, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='"',
                            quoting=csv.QUOTE_NONNUMERIC)
        writer.writerows(data)
