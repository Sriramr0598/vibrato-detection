import numpy as np
from numpy import *
from derivative_peak_percentage import derivative_peak_percentage
import json

with open('input_data.json', 'r') as data_file:
    data = json.load(data_file)

percentages = []
for i in range(len(data)):
    file = data[i]['File_Name']
    percentage = derivative_peak_percentage(file)
    print(file, percentage)
    percentages.append(percentage)
pass