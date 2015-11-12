# -*- coding: utf-8 -*-

import csv

MY_FILE = "..."

def parse(raw_file, delimiter):

    opened_file = open(raw_file)
    
    csv_data = csv.reader(opened_file, delimiter=delimiter, )

    parsed_data = []

    fields = csv_data.next()

    for row in csv_data:
        parsed_data.append(dict(zip(fields, row)))

    opened_file.close()

    return parsed_data

def main():
    #visualize_data()
    parse(MY_FILE,"~")

if __name__ == "__main__":
    main()