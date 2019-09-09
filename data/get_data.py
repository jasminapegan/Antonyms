import xml.etree.ElementTree as ET
import re


def antonyms_slownet(slownet_file, out_file):
    """ Finds all occurences of antonyms in 'slownet_file' SloWNet database and writes them to 'out_file'. """

    tree = ET.parse(slownet_file)
    root = tree.getroot()

    with open(out_file, 'wt', encoding="utf8") as file:

        for synset in root:
            #id = synset[0].text

            for ilr in synset.iter('ILR'):
                if ilr.attrib["type"] == 'near_antonym':

                    for synonym in synset.iter('SYNONYM'):
                        if synonym.attrib['{http://www.w3.org/XML/1998/namespace}lang'] == 'sl':
                            literals = synonym.iter('LITERAL')

                            for synset2 in root:
                                # check if word id equals antonym id
                                if synset2[0].text == ilr.text:

                                    for syn in synset2.iter('SYNONYM'):
                                        if syn.attrib['{http://www.w3.org/XML/1998/namespace}lang'] == 'sl':
                                            lits = syn.iter('LITERAL')

                                            for x in literals:
                                                for y in lits:
                                                    file.write(x.text + "\t" + y.text + "\n")


def antonyms_sskj(sskj_file, out_file):
    """ Finds all antonym occurences in 'sskj_file' SSKJ database and writes them to 'out_file'. """

    regex = re.compile('>ant\.')
    regex2 = re.compile('<span title="Iztočnica">.*?</span>')
    regex3 = re.compile('<span title="Protipomenka"><a.*?>.*?</a>')

    with open(sskj_file, "r", encoding="utf8") as sskj:
        with open(out_file, 'wt', encoding="utf8") as file:

            for line in sskj.readlines():
                results = regex.search(line)

                if results != None:
                    word = regex2.search(line).group(0)
                    word = word[word.index(">") + 1:word.index("</")]

                    antonym = regex3.search(line).group(0)
                    antonym = antonym[antonym.index(">") + 4:antonym.index("</")]

                    file.write(word + "\t" + antonym + "\n")


def no_plus_word(labels_file, out_file, prefix_list):
    """
        Adds prefixes from 'prefix_list' to each word from 'labels_file'
        and writes to 'out_file' only if such word exists in 'labels_file'.
    """
    with open(labels_file, 'r') as in_file:
        with open(out_file, 'wt') as out:

            words = in_file.readlines()

            # try to add prefixes to each word
            for word in words:
                for prefix in prefix_list:
                    ne_word = prefix + word

                    # check if such word exists in labels
                    if ne_word in words:
                        out.write(word[:-1] + "\t" + ne_word)


def synonyms_cjvt(cjvt_file, out_file):
    """ Writes to 'out_file' only most reliable synonyms in 'cjvt_file' CJVT database. """

    with open(cjvt_file, "r") as in_file:
        tree = ET.parse(in_file)
    root = tree.getroot()

    with open(out_file, 'wt') as file:

        for i, entry in enumerate(root):
            headword = entry[0].text

            #  all of our data only has one word, so we do not use multi-word expressions
            if " " not in headword:
                for group in entry.iter('groups_core'):
                    for candidate in group[0].iter('candidate'):
                        synonym = candidate[0].text

                        if " " not in synonym:
                            try:
                                file.write(headword + " " + synonym + "\n")
                            except Exception as e:
                                print(e)
