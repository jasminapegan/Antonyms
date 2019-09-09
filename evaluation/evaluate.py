from numpy import dot, array
from numpy.linalg import norm
from joblib import load
from keras.models import load_model


def eval_svm(test_file, antonyms_file, synonyms_file, data_file, larger_data_file, model_file, out_file):
    with open(out_file, "a", encoding="utf8") as out:

        TP, TN, FP, FN = 0, 0, 0, 0
        positive, negative, unknown, undefined = 0, 0, 0, 0
        undef_pos0, undef_pos1, undef_neg0, undef_neg1 = 0, 0, 0, 0
        num = 0
        visited = []

        # load model
        model = load(model_file)
        print("model loaded!")
        i = 0
        with open(test_file, "r", encoding="utf8") as in_file:
            for line in in_file:
                chosen_word = line[:-1]

                if chosen_word not in visited:
                    visited.append(chosen_word)

                    # find vec of a given word
                    chosen_vec = []
                    labels_vec_distances = []

                    word_found = False
                    with open(data_file, 'r', encoding="utf8") as data:

                        for line in data:
                            line_list = line.split("\t")

                            if line_list[0] == chosen_word:
                                word_found = True
                                line_list[-1] = line_list[-1][:-1]
                                chosen_vec = array(line_list[1:], dtype=float)
                                break
                    if not word_found:
                        with open(larger_data_file, 'r', encoding="utf8") as data2:

                            for line in data2:
                                line_list = line.split("\t")

                                if line_list[0] == chosen_word:
                                    word_found = True
                                    line_list[-1] = line_list[-1][:-1]
                                    chosen_vec = array(line_list[1:], dtype=float)
                                    break
                    if not word_found:
                        print("Chosen word %s does not exist in database." % chosen_word)
                    else:
                        norm_chosen = norm(chosen_vec)
                        num += 1
                        # find nearest neighbors
                        with open(data_file, 'r', encoding="utf8") as data:
                            for j, line2 in enumerate(data):
                                line_list = line2.split("\t")
                                line_list[-1] = line_list[-1][:-1]
                                wordVec = array(line_list[1:], dtype=float)
                                cos_sim = dot(chosen_vec, wordVec) / (norm_chosen * norm(wordVec))

                                if j <= 20:
                                    labels_vec_distances.append([line_list[0], wordVec, cos_sim])
                                    labels_vec_distances.sort(key=lambda x: x[2], reverse=True)
                                elif cos_sim > labels_vec_distances[19][2]:
                                    labels_vec_distances[19] = [line_list[0], wordVec, cos_sim]
                                labels_vec_distances.sort(key=lambda x: x[2], reverse=True)

                        # #############################################################################
                        # classify candidates
                        X = [wordVec.tolist() + candidate[1].tolist() for candidate in labels_vec_distances[1:]]
                        X = array(X)
                        results = model.predict(X)

                        # compare results with test set
                        for res, wordVec2 in zip(results, labels_vec_distances[1:]):

                            pair1 = chosen_word + "\t" + wordVec2[0] + "\n"
                            pair2 = wordVec2[0] + "\t" + chosen_word + "\n"

                            pair_found = False

                            with open(antonyms_file, 'r', encoding="utf8") as antonymsFile:
                                for antonyms in antonymsFile:
                                    if pair1 == antonyms or pair2 == antonyms:
                                        pair_found = True

                                        if int(res) == 1:
                                            # print(pair1, res, "positive")
                                            positive += 1
                                            TP += 1
                                            break
                                        elif int(res) == 0:
                                            # print(pair1, res, "negative")
                                            negative += 1
                                            FP += 1
                                            break

                            if not pair_found:
                                with open(synonyms_file, 'r') as synonymsFile:
                                    for synonyms in synonymsFile:

                                        if pair1 == synonyms or pair2 == synonyms:
                                            pair_found = True

                                            if int(res) == 0:
                                                # print(pair1, res, "positive")
                                                positive += 1
                                                TN += 1
                                                break
                                            elif int(res) == 1:
                                                # print(pair1, res, "negative")
                                                negative += 1
                                                FN += 1
                                                break

                            if not pair_found:
                                # print(pair1, res, "unknown")
                                unknown += 1

                i += 1

        """
            output shape:
                        defined     sum
            correct     TP  TN      #correctly classified
            incorrect   FP  FN      #missclassified
            unknown     1   0       #unknown
        """
        acc = 0
        acc2 = 0
        if positive + negative != 0:
            acc = positive / (positive + negative)
        correctSum = TP + TN + undef_pos1 + undef_pos0
        incorrSum = FP + FN + undef_neg1 + undef_neg0
        if correctSum + incorrSum != 0:
            acc2 = correctSum / (correctSum + incorrSum)

        out.write("#unknown: " + str(unknown))
        out.write("undefined/TN, TP " + str(undef_pos0) + " " + str(undef_pos1) + "\n")
        out.write("undefined/FN, FP " + str(undef_neg0) + " " + str(undef_neg1) + "\n")
        out.write("undefined/unknown: " + str(undefined) + "\n")
        out.write("TP, TN, FP, FN: " + str(TP) + " " + str(TN) + " " + str(FP) + " " + str(FN) + "\n")
        out.write("acc: " + str(acc) + "\n")
        out.write("acc with undefined: " + str(acc2) + "\n")
        out.write("numClusters: " + str(num) + "\n")

        out.write("\t\t\tdefined\t\tsum\n")
        out.write("correct\t\t%d\t%d\t\t%d\n" %
                  (TP, TN, correctSum))
        out.write("incorrect\t%d\t%d\t\t%d\n" %
                  (FP, FN, incorrSum))
        out.write("unknown\t\t\t%d\n" % unknown)
        out.write("\t\t\t\t\t\t\t\t\t%d\n" % (20 * num))


def eval_dnn(test_file, antonyms_file, synonyms_file, data_file, larger_data_file, model_file, out_file, delta=0):
    with open(out_file, "a", encoding="utf8") as out:

        allTP, allTN, allFP, allFN = 0, 0, 0, 0
        positive, negative, unknown, undefined = 0, 0, 0, 0
        undef_pos0, undef_pos1, undef_neg0, undef_neg1 = 0, 0, 0, 0
        num = 0
        visited = []

        # load and compile model
        model = load_model(model_file)
        print("model loaded!")
        model.compile(loss='binary_crossentropy',
                      optimizer='adadelta',
                      metrics=['accuracy'])
        print("model compiled!")

        i = 0
        with open(test_file, "r", encoding="utf8") as in_file:
            for line in in_file:
                print("line %d" % i)

                chosen_word = line[:-1]

                if chosen_word not in visited:
                    visited.append(chosen_word)

                    # find vec of a given word
                    chosen_vec = []
                    labels_vec_distances = []

                    word_found = False
                    with open(data_file, 'r', encoding="utf8") as data:

                        for line in data:
                            line_list = line.split("\t")

                            if line_list[0] == chosen_word:
                                word_found = True
                                line_list[-1] = line_list[-1][:-1]
                                chosen_vec = array(line_list[1:], dtype=float)
                                break

                    if not word_found:
                        with open(larger_data_file, 'r', encoding="utf8") as data2:

                            for line in data2:
                                line_list = line.split("\t")

                                if line_list[0] == chosen_word:
                                    word_found = True
                                    line_list[-1] = line_list[-1][:-1]
                                    chosen_vec = array(line_list[1:], dtype=float)
                                    break
                    if not word_found:
                        print("Chosen word %s not found." % chosen_word)
                    else:
                        norm_chosen = norm(chosen_vec)
                        num += 1
                        # find nearest neighbors
                        with open(data_file, 'r', encoding="utf8") as data:
                            for j, line2 in enumerate(data):
                                line_list = line2.split("\t")
                                line_list[-1] = line_list[-1][:-1]
                                wordVec = array(line_list[1:], dtype=float)

                                cos_sim = dot(chosen_vec, wordVec) / (norm_chosen * norm(wordVec))

                                if j <= 20:
                                    labels_vec_distances.append([line_list[0], wordVec, cos_sim])
                                    labels_vec_distances.sort(key=lambda x: x[2], reverse=True)
                                elif cos_sim > labels_vec_distances[19][2]:
                                    labels_vec_distances[19] = [line_list[0], wordVec, cos_sim]
                                labels_vec_distances.sort(key=lambda x: x[2], reverse=True)

                        # #############################################################################
                        # classify candidates
                        X = [wordVec.tolist() + candidate[1].tolist() for candidate in labels_vec_distances[1:]]
                        X = array(X)

                        results = model.predict(X)

                        # compare results with test set
                        for res, wordVec2 in zip(results, labels_vec_distances[1:]):

                            pair1 = chosen_word + "\t" + wordVec2[0] + "\n"
                            pair2 = wordVec2[0] + "\t" + chosen_word + "\n"

                            pair_found = False

                            with open(antonyms_file, 'r', encoding="utf8") as antonymsFile:
                                for antonyms in antonymsFile:
                                    if pair1 == antonyms or pair2 == antonyms:
                                        pair_found = True

                                        if res[0] > 0.5 + delta:
                                            positive += 1
                                            allTP += 1
                                            break
                                        elif res[0] < 0.5 - delta:
                                            negative += 1
                                            allFP += 1
                                            break
                                        else:
                                            if res[0] >= 0.5:
                                                undef_pos1 += 1
                                                break
                                            else:
                                                undef_neg1 += 1
                                                break

                            if not pair_found:
                                with open(synonyms_file, 'r') as synonymsFile:
                                    for synonyms in synonymsFile:

                                        if pair1 == synonyms or pair2 == synonyms:
                                            pair_found = True

                                            if res[0] < 0.5 - delta:
                                                positive += 1
                                                allTN += 1
                                                break
                                            elif res[0] > 0.5 + delta:
                                                negative += 1
                                                allFN += 1
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
                i += 1


        """
            output shape:
                        defined     undefined   sum
            correct     TP  TN      uTP uTN     #correctly classified
            incorrect   FP  FN      uFP uFN     #missclassified
            sum         1   0       u1  u0      #known correctness
            unknown     defUnknown  undefUnkn   #unknown
                                                #all
        """
        acc = 0
        acc2 = 0
        if positive + negative != 0:
            acc = positive/(positive + negative)
        correctSum = allTP + allTN + undef_pos1 + undef_pos0
        incorrSum = allFP + allFN + undef_neg1 + undef_neg0
        if correctSum + incorrSum != 0:
            acc2 = correctSum / (correctSum + incorrSum)

        out.write("#unknown: " + str(unknown))
        out.write("undefined/TN, TP " + str(undef_pos0) + " " + str(undef_pos1) + "\n")
        out.write("undefined/FN, FP " + str(undef_neg0) + " " + str(undef_neg1) + "\n")
        out.write("undefined/unknown: "+str(undefined) + "\n")
        out.write("TP, TN, FP, FN: " + str(allTP) + " " + str(allTN) + " " + str(allFP) + " " + str(allFN) + "\n")
        out.write("acc: " + str(acc) + "\n")
        out.write("acc with undefined: " + str(acc2) + "\n")
        out.write("numClusters: " + str(num) + "\n")

        out.write("\t\t\tdefined\t\tundefined\t\tsum\n")
        out.write("correct\t\t%d\t%d\t\t%d\t%d\t\t%d\n" %
                  (allTP, allTN, undef_pos1, undef_pos0, correctSum))
        out.write("incorrect\t%d\t%d\t\t%d\t%d\t\t%d\n" %
                  (allFP, allFN, undef_neg1, undef_neg0, incorrSum))
        out.write("sum\t\t\t%d\t%d\t\t%d\t%d\t\t%d\n" %
                  (allTP + allFP, allTN + allFN, undef_pos1 + undef_neg1, undef_pos0 + undef_neg0, correctSum + incorrSum))
        out.write("unknown\t\t%d\t\t\t%d\t\t\t%d\n" %
                  (unknown - undefined, undefined, unknown))
        out.write("\t\t\t\t\t\t\t\t\t%d\n" % (20 * num))
