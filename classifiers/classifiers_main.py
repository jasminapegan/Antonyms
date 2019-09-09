import numpy as np
from classifiers.models import *


Cs = [np.logspace(-3, 3, 7),
          np.logspace(0, 2, 8),
          np.logspace(0, 2, 16),
          np.logspace(0, 1.5, 20),
          np.linspace(10, 50, 10),
          np.linspace(10, 50, 10)
      ]
gammas = [np.logspace(-3, 3, 7),
          np.logspace(-3, -2, 8),
          np.logspace(-3, -1, 16),
          np.logspace(-3, -2, 20),
          np.linspace(0.0, 0.5, 10),
          np.linspace(0.005, 0.2, 10)
          ]

#find_params_svm("../data/train.txt", "rbf", [1, 10], [0.1, 0.01], "params_svm.txt")

k_range = [128]
k2_scale = [16]
k3_scale = [4]
k4_scale = [2]
#find_params_neural(k_range, k2_scale, k3_scale, k4_scale, "../data/train.txt", "dnn_new.h5", "params_neural.txt")

#make_models_svm("../data/train.txt", [[10, 0.1], [100, 0.01]], "rbf", "svm_rbf_out.txt")
#make_model_neural("../data/train.txt", "models/dnn_new.h5", "neural_out.txt")

filenames = [
    "dnn_new.h5",
    "svm2/svm_18_0.joblib"
]

kfold("../data/train.txt", filenames, "out_kfold.txt")