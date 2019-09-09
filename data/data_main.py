from data.make_files import *
from data.get_data import *

# Examples using defined methods.

# matches words that contain only characters of slovene alphabet
regex = r'^[a-pr-vzčšž]+$'

clean_tsv_file('sources/dataAlphaSlo.tsv', 'sources/dataAll.tsv', 20, use_regex=True, regexpression=regex)

make_dataset('dataset/brezPonovitev/vsiAntonimi.txt',
             'dataset/brezPonovitev/vsiObstojeciSinonimi.txt',
             'sources/dataAlphaSlo.tsv', 'dataset/podatkovnaMnozica.txt')

get_labels('sources/dataAlphaSlo.tsv', "utf8", "sources/labelsAlphaSlo.txt")

remove_nonexistent('sources/labelsAlphaSlo.txt', 'dataset/brezPonovitev/sinonimi.txt',
                   'dataset/brezPonovitev/vsiObstojeciSinonimiNew.txt')

get_more_antonyms('dataset/brezPonovitev/vsiObstojeciAntonimi2.txt', 'dataset/vsiPariObstojecihAntonimov2.txt')


dataset_to_words("testSyn.txt", "testAnt.txt", "test.txt", "sources/data_sskj.tsv")


antonyms_slownet('sources/slownet-2015-05-07.xml', 'dataset/antonimi_slownet.txt')
antonyms_sskj('sources/sskj2_v1.txt', 'dataset/antonimi_sskj.txt')
no_plus_word('sources/labels_alpha_slo.txt', 'dataset/ne_plus_words.txt', ["ne", "proti", "brez"])
synonyms_cjvt('sources/CJVT_Thesaurus-v1.0.xml', 'dataset/sinonimi.txt')
