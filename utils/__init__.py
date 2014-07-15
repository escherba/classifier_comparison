import csv


def to_csv(fname, data):
    writer = csv.writer(fname, delimiter=',', quotechar='"',
                        quoting=csv.QUOTE_NONNUMERIC)
    writer.writerows(data)
