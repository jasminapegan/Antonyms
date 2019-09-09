from numpy import dot, array
from numpy.linalg import norm
from joblib import load
import keras.models


def find_vec(word, data_file):
    """ Finds word embedding of 'word' in 'data_file'. """
    with open(data_file, 'r', encoding="utf8") as data:

        for line in data:
            line_list = line.split("\t")
            if line_list[0] == word:
                line_list[-1] = line_list[-1][:-1]
                return array(line_list[1:], dtype=float)

        return []


def find_neighbors(data, chosen_vec, norm_chosen, num_neighbors):
    """ Returns list of 'num_neighbors' nearest neighbors in 'data' file. """
    labels_vec_distances = []

    with open(data, 'r', encoding="utf8") as data_file:
        for j, line2 in enumerate(data_file):

            line_list = line2.split("\t")
            line_list[-1] = line_list[-1][:-1]
            word_vec = array(line_list[1:], dtype=float)

            cos_sim = dot(chosen_vec, word_vec) / (norm_chosen * norm(word_vec))

            if j <= num_neighbors:
                labels_vec_distances.append([line_list[0], word_vec, cos_sim])
                labels_vec_distances.sort(key=lambda x: x[2], reverse=True)

            elif cos_sim > labels_vec_distances[num_neighbors - 1][2]:
                labels_vec_distances[num_neighbors - 1] = [line_list[0], word_vec, cos_sim]
            labels_vec_distances.sort(key=lambda x: x[2], reverse=True)

    return labels_vec_distances[1:]  # first element equals chosen_vec


def compare_test_set(results, neighbors, chosen_word, ant_file, syn_file, neural, delta):
    """ Returns results for all of classification outcomings. """
    positive, negative, unknown, undefined = 0, 0, 0, 0
    TP, TN, FP, FN = 0, 0, 0, 0
    undef_pos0, undef_pos1, undef_neg0, undef_neg1 = 0, 0, 0, 0

    for res, wordVec2 in zip(results, neighbors):

        if not neural:  # svm returns differently shaped result
            res = [int(res)]

        pair1 = chosen_word + "\t" + wordVec2[0] + "\n"
        pair2 = wordVec2[0] + "\t" + chosen_word + "\n"

        pair_found = False

        with open(ant_file, 'r', encoding="utf8") as antonymsFile:
            for antonyms in antonymsFile:
                if pair1 == antonyms or pair2 == antonyms:
                    pair_found = True

                    if res[0] > 0.5 + delta:
                        positive += 1
                        TP += 1
                        break
                    elif res[0] < 0.5 - delta:
                        negative += 1
                        FN += 1
                        break
                    else:
                        if res[0] >= 0.5:
                            undef_pos1 += 1
                            break
                        else:
                            undef_neg1 += 1
                            break

        if not pair_found:
            with open(syn_file, 'r', encoding="utf8") as synonymsFile:
                for synonyms in synonymsFile:

                    if pair1 == synonyms or pair2 == synonyms:
                        pair_found = True

                        if res[0] < 0.5 - delta:
                            positive += 1
                            TN += 1
                            break
                        elif res[0] > 0.5 + delta:
                            negative += 1
                            FP += 1
                            break
                        else:
                            if res[0] >= 0.5:
                                undef_neg0 += 1
                                break
                            else:
                                undef_pos0 += 1
                                break

        if not pair_found:
            if 0.5 - delta <= res[0] <= 0.5 + delta:
                undefined += 1
            unknown += 1

    return {"positive": positive, "negative": negative, "unknown": unknown, "undefined": undefined,
            "TP": TP, "FP": FP, "TN": TN, "FN": FN,
            "undef_pos0": undef_pos0, "undef_pos1": undef_pos1, "undef_neg0": undef_neg0, "undef_neg1": undef_neg1}


def write_result(positive, negative, TP, TN, FP, FN, undefined, undef_pos1, undef_pos0, undef_neg1, undef_neg0,
                 unknown, num, num_neighbors, out_file, delta=0):
    """
        output shape:
                    defined     undefined   sum
        correct     TP  TN      uTP uTN     #correctly classified
        incorrect   FP  FN      uFP uFN     #missclassified
        sum         1   0       u1  u0      #known correctness
        unknown     defUnknown  undefUnkn   #unknown
                                            #all
    """
    with open(out_file, "a") as out:
        acc, acc2 = 0, 0
        if positive + negative != 0:
            acc = positive / (positive + negative)  # acc without undefined
        correct_sum = TP + TN + undef_pos1 + undef_pos0
        incorr_sum = FP + FN + undef_neg1 + undef_neg0
        if correct_sum + incorr_sum != 0:
            acc2 = correct_sum / (correct_sum + incorr_sum)  # acc with undefined

        out.write("delta: %d\n" % delta)
        out.write("unknown: %d\n" % unknown)
        out.write("undefined/TN, TP: %d, %d\n" % (undef_pos0, undef_pos1))
        out.write("undefined/FN, FP: %d, %d \n" % (undef_neg0, undef_neg1))
        out.write("undefined/unknown: %d\n" % undefined)
        out.write("TP, TN, FP, FN: %d, %d, %d, %d\n" % (TP, TN, FP, FN))
        out.write("acc: %d\n" % acc)
        out.write("acc with undefined: %d\n" % acc2)
        out.write("num groups: %d\n" % num)

        out.write("\t\t\tdefined\t\tundefined\t\tsum\n")
        out.write("correct\t\t%d\t%d\t\t%d\t%d\t\t%d\n" %
                  (TP, TN, undef_pos1, undef_pos0, correct_sum))
        out.write("incorrect\t%d\t%d\t\t%d\t%d\t\t%d\n" %
                  (FP, FN, undef_neg1, undef_neg0, incorr_sum))
        out.write("sum\t\t\t%d\t%d\t\t%d\t%d\t\t%d\n" %
                  (TP + FP, TN + FN, undef_pos1 + undef_neg1, undef_pos0 + undef_neg0, correct_sum + incorr_sum))
        out.write("unknown\t\t%d\t\t\t%d\t\t\t%d\n" %
                  (unknown - undefined, undefined, unknown))
        out.write("\t\t\t\t\t\t\t\t\t%d\n" % (num_neighbors * num))  # number of words per group * num groups


def evaluate(words_file, ant_file, syn_file, data_path, larger_data_path,
             model_file, out_file, num_neighbors=20, delta=0):

    TP, TN, FP, FN = 0, 0, 0, 0
    positive, negative, unknown, undefined = 0, 0, 0, 0
    undef_pos0, undef_pos1, undef_neg0, undef_neg1 = 0, 0, 0, 0
    num = 0
    visited = []

    is_neural = model_file[-3:] == ".h5"

    # make model
    if is_neural:
        model = keras.models.load_model(model_file)
        print("model loaded!")
        model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])
        print("model compiled!")

    else:
        model = load(model_file)
        print("model loaded!")

    i = 0
    with open(words_file, "r", encoding="utf8") as in_file:
        for line in in_file:
            chosen_word = line[:-1]

            if chosen_word not in visited:
                visited.append(chosen_word)

                chosen_vec = find_vec(chosen_word, data_path)

                if len(chosen_vec) == 0:  # word not found
                    chosen_vec = find_vec(chosen_word, larger_data_path)

                if len(chosen_vec) == 0:
                    print("Chosen word %s does not exist in file %s."
                          % (chosen_word, larger_data_path))

                else:
                    norm_chosen = norm(chosen_vec)
                    num += 1

                    # find and classify candidates
                    neighbors = find_neighbors(data_path, chosen_vec, norm_chosen, num_neighbors)
                    X = [chosen_vec.tolist() + candidate[1].tolist() for candidate in neighbors]
                    X = array(X)

                    results = model.predict(X)

                    # compare result with test set
                    compare = compare_test_set(results, neighbors, chosen_word, ant_file, syn_file,
                                               is_neural, delta=delta)

                    positive += compare["positive"]
                    negative += compare["negative"]
                    unknown += compare["unknown"]
                    undefined += compare["undefined"]
                    TP += compare["TP"]
                    FP += compare["FP"]
                    TN += compare["TN"]
                    FN += compare["FN"]
                    undef_pos0 += compare["undef_pos0"]
                    undef_neg0 += compare["undef_neg0"]
                    undef_pos1 += compare["undef_pos1"]
                    undef_neg1 += compare["undef_neg1"]
            i += 1

    write_result(positive, negative, TP, TN, FP, FN, undefined, undef_pos1, undef_pos0, undef_neg1, undef_neg0,
                 unknown, num, num_neighbors, out_file)
