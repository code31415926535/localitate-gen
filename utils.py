import csv
import json

def read_csv():
    rows = []
    with open('orase.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            rows.append(row)
    return list(map(lambda x: x[2].lower(), rows[1:]))

def read_magyarorszag():
    rows = []
    with open('telepules.txt', 'r') as file:
        for row in file:
            rows.append(row.split('\t')[1].lower())
    return list(set(rows))

def read_stats():
    with open('stats.json', 'r') as file:
        return json.load(file)

def get_chars(rows):
    return sorted(list(set(''.join(rows))))

def find_first(rows, chars):
    for row in rows:
        for char in chars:
            if char in row:
                return row
    return None