from copy import deepcopy
from src.emomarl.sample import Sample

'''
Define a MOPG task, which is a pair of a policy and a scalarization weight.
'''
class Task:
    def __init__(self, sample, weight):
        self.sample = Sample.copy_from(sample)
        self.weight = deepcopy(weight)
