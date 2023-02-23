import csv

def read_csv():
    rows = []
    with open('orase.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            rows.append(row)
    return list(map(lambda x: x[2].lower(), rows[1:]))

def get_chars(rows):
    return sorted(list(set(''.join(rows))))

def find_first(rows, chars):
    for row in rows:
        for char in chars:
            if char in row:
                return row
    return None