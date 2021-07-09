import numpy as np
import matplotlib.pyplot as plt

classes = ['No Data', 'Cultivated Land', 'Forest', 'Grassland', 'Shrubland', 'Water', 'Wetlands', 'Tundra', 'Artificial Surface', 'Bareland']
def single_hist(a: np.array):
  plt.hist(a, bins=list(range(len(classes) + 1)))
  plt.xticks(ticks=list(map(lambda x: x+0.5, list(range(len(classes))))), labels=classes, rotation=90)
  plt.tight_layout()
  plt.show()

def eval_hist(true: np.array, pred: np.array):
  plt.figure(figsize=[15, 10])
  plt.hist([true, pred], width = 0.25, histtype='bar', weights=[(np.zeros_like(true) + 1. / true.size), (np.zeros_like(pred) + 1. / pred.size)], align='mid')
  plt.legend(['y True', 'y Prediction'])
  plt.xticks([i + 0.25 for i in range(len(classes))], classes)
  plt.show()