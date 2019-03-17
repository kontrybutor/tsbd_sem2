import numpy as np
import pandas as pd

filename = 'fertility_diagnosis.data'

names = ['Season', 'Age', 'Childish diseases', 'Accident', 'Surgical intervention',
         'High fevers', 'Alcohol consumption', 'Smoking habit', 'Sitting hours', 'Diagnosis']
fertility_df = pd.read_csv(filename, sep=",", header=None, names=names)


def print_statistics(attribute):
    attr = fertility_df[attribute]
    print("Median is:", attr.mean())
    print("Max value is:", attr.max())
    print("Min value is", attr.min())


for name in names[:-1]:
    print("Statistics for:", name)
    print_statistics(name)
    print()



