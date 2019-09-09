import csv
import re


def clean_tsv_file(out_file, data, num_lines, use_regex=False, regexpression=r'.*'):
    """
        Makes file named 'out_file' consisting only of readable lines from 'num_lines' of 'data' file.
        If optional parameter 'use_regex'=True, writes only lines that match 'regexpression'.
    """
    regex = re.compile(regexpression)

    with open(out_file, 'wt', encoding="utf8") as data_file:
        with open(data, 'r', encoding="utf8") as in_file:

            tsv_writer_data = csv.writer(data_file, delimiter='\t', lineterminator='\n')

            # first line tells num of lines and num of fields in line
            line = in_file.readline()

            err = 0
            i = 0
            while i < num_lines:
                try:
                    line = in_file.readline()
                except Exception as e:
                    print(e)
                    print("error: line", i)
                    # ignore 2 lines
                    err = 2
                    i += 1
                    continue
                finally:
                    # check if end of file
                    if len(line) == 0:
                        return

                    # if error occurred, jump line
                    if err:
                        print(line)
                        err -= 1
                    else:
                        line_list = line.split("\t")
                        line_list[-1] = line_list[-1][:-1]  # last char is newline

                        # first field is label
                        if not use_regex:
                            tsv_writer_data.writerow(line_list[0:100])
                        if use_regex and regex.match(line_list[0]):
                            tsv_writer_data.writerow(line_list[0:100])
                        i += 1

def make_dataset(antonyms, synonyms, embeddings, out_dataset, encoding_data="windows-1250"):
    """
        Makes dataset file named 'out_dataset' consisting of pairs of synonym and antonym embeddings,
        including info about class in the last field: 0 --> synonyms, 1 --> antonyms.
        Synonyms and antonyms in files must be words separated by '\t' (tab), ex. "dober  slab".
        Limits number of synonyms to number of antonyms (usually we have more than antonyms).
        Optional parameters: 'encoding_antsyn' is encoding of antonyms and synonyms files,
        'encoding_data' is encoding of data file.
    """

    with open(antonyms, 'r', encoding="utf8") as antonymsFile:
        antonyms = antonymsFile.readlines()

    with open(out_dataset, 'wt') as out_file:
        num_ant = 0
        for line in antonyms:
            num_ant += 1
            w1, w2 = line.split("\t")
            w2 = w2[:-1]
            new_line = "\t"
            with open(embeddings, 'r', encoding=encoding_data) as data:

                w1_found = False
                w2_found = False
                line2 = data.readline()

                while len(line2) != 0 and not (w1_found and w2_found):
                    word = line2.split("\t")
                    word[-1] = word[-1][:-1]

                    if word[0] == w1:
                        new_line = " ".join(word[1:]) + new_line
                        w1_found = True
                    elif word[0] == w2:
                        new_line += " ".join(word[1:])
                        w2_found = True

                    line2 = data.readline()

                new_line += "\t1\n"
                if w1_found and w2_found:
                    out_file.write(new_line)

        with open(synonyms, 'r', encoding="utf8") as synonyms:
            for i, line in enumerate(synonyms.readlines()):
                if i < num_ant:
                    w1, w2 = line.split(" ")
                    w2 = w2[:-1]
                    new_line = "\t"
                    with open(embeddings, 'r', encoding=encoding_data) as data:

                        w1_found = False
                        w2_found = False
                        line2 = data.readline()

                        while len(line2) != 0 and not (w1_found and w2_found):
                            word = line2.split("\t")
                            word[-1] = word[-1][:-1]  # last char is newline

                            if word[0] == w1:
                                new_line = " ".join(word[1:]) + new_line
                                w1_found = True
                            elif word[0] == w2:
                                new_line += " ".join(word[1:])
                                w2_found = True

                            line2 = data.readline()

                        new_line += "\t0\n"
                        if w1_found and w2_found:
                            out_file.write(new_line)


def remove_duplicate_lines(in_file, out_file):
    """ Writes to 'out_file' lines from 'in_file' jumping duplicate lines. """
    visited_lines = []

    with open(in_file, 'r') as input:
       with open(out_file, 'wt') as output:

            for line in input.readlines():
                if line not in visited_lines:

                    output.write(line)
                    visited_lines.append(line)


def remove_nonexistent(labels, in_file, out_file):
    """ Writes to 'out_file' only lines from 'in_file' that have label (first field) which also exists in 'labels'. """
    with open(labels, 'r', encoding="utf8") as labels:
        all_words = labels.readlines()

        with open(in_file, 'r') as antonyms:
            with open(out_file, 'wt', encoding="utf8") as output:

                for line in antonyms.readlines():
                    w1, w2 = line.split(" ")
                    w1 += "\n"

                    if w1 in all_words and w2 in all_words:
                        output.write(line)


def get_more_antonyms(existing_antonyms, out_file):
    """
        Reads antonyms from 'existing_antonyms' and writes to 'out_file' all combinations that can be induced.
        Example:
                w1 == w3, w2 == w5 --> w4, w6
                w1, w2: izključno,	delno
                w3, w4: izključno,	deloma
                w5, w6: delno,	popolnoma
        New pair: deloma, popolnoma
    """
    with open(existing_antonyms, 'r') as antonyms:
        with open(out_file, 'wt') as out:

            all_pairs = antonyms.readlines()

            for line in all_pairs:
                w1, w2 = line.split("\t")
                w2 = w2[:-1]

                for line2 in all_pairs:
                    w3, w4 = line2.split("\t")
                    w4 = w4[:-1]

                    for line3 in all_pairs:
                        w5, w6 = line3.split("\t")
                        w6 = w6[:-1]

                        if w1 == w3 and w2 == w5:
                            out.write(w6 + "\t" + w4 + "\n")


def dataset_to_words(syn_file, ant_file, dataset_file, data_file):
    """
        Reads 'dataset_file' with word embeddings and class, finds original words using 'data_file'
        and writes to 'syn_file' if class == 0 or to 'ant_file' if class == 1.
    """
    with open(syn_file, "w") as syn:
        with open(ant_file, "w") as ant:
            with open(dataset_file, "r") as in_file:

                for line in in_file:
                    vec1, vec2, clas = line.split("\t")
                    clas = int(clas)
                    vec1, vec2 = vec1.split(" "), vec2.split(" ")
                    a, b = "", ""

                    with open(data_file, "r", encoding="utf8") as data:
                        for line2 in data:

                            line2 = line2.split("\t")
                            vec = line2[1:]
                            vec[-1] = vec[-1][:-1]

                            if vec == vec1:
                                a = line2[0]
                            elif vec == vec2:
                                b = line2[0]

                            # check if both words are found yet
                            if len(a) * len(b) > 0:
                                out = a + "\t" + b + "\n"
                                if clas == 0:
                                    syn.write(out)
                                else:
                                    ant.write(out)
                                break


def get_labels(in_file, encoding, out_file):
    """ Writes to 'out_file' only first column of 'in_file'. """
    with open(in_file, "r", encoding=encoding) as data:
        with open(out_file, "w") as out:
           for line in data:
               out.write(line.split("\t")[0] + "\n")

