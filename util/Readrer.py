import csv

reader = []


def read_file(path):
    global reader
    table = []
    with open(path, "r") as file:
        reader = csv.reader(file)
        for row in reader:
            table.append(row)
    return table
