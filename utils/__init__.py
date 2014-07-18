import csv


def to_csv(fname, data):
    with open(fname, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='"',
                            quoting=csv.QUOTE_NONNUMERIC)
        writer.writerows(data)


def split_num(m, n):
    """
    Split a large number m into n approximately sized buckets
    returns a list of bucket sizes
    """
    avg_sz = m / n
    rem = m - avg_sz * (n - 1)
    result = [avg_sz] * (n - 1)
    remrem = rem - avg_sz
    for i in range(0, remrem):
        result[i] += 1
        remrem -= 1
    return result + [avg_sz + remrem]
