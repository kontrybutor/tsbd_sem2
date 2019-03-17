import numpy as np

filename = 'ecoli.data'
data = np.genfromtxt(filename, delimiter=',', names=True, dtype=None, encoding='utf-8')

print (data)