from evaluation.evaluate import *
from evaluation.evaluate_corrected import *

# first test set
words_file = "../data/datasets/test_words.txt"
antonyms = '../data/datasets/test_antonyms.txt'
synonyms = '../data/datasets/test_synonyms.txt'

# balanced test set
words_file_balanced = "../data/datasets/balanced/test_words.txt"
antonyms_balanced = '../data/datasets/balanced/test_antonyms.txt'
synonyms_balanced = '../data/datasets/balanced/test_synonyms.txt'

# embedding data
data = "../data/sources/data_sskj.tsv"
data_more = "../data/sources/data_alpha_slo.tsv"
data_all = "../data/sources/data_all.tsv"

# Originally used methods
eval_svm(words_file, antonyms, synonyms, data, data_more,
         "../classifiers/models/final/svm_final_14_0.joblib", "out_svm_2.txt")

eval_dnn(words_file, antonyms, synonyms, data, data_more,
         "../classifiers/models/final/dnn_5_layer_2.h5", "out_neural_2.txt", delta=0.3)

# Corrected methods
evaluate(words_file, antonyms_balanced, synonyms_balanced, data, data_more,
         "../classifiers/models/final/svm_final_14_0.joblib", "out_svm_2.txt")

evaluate(words_file, antonyms, synonyms, data, data_more,
         "../classifiers/models/final/dnn_5_layer_2.h5", "out_neural.txt", delta=0.3)