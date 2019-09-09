### REPOSITORY FOR 

## INSTALLATION
To use this project, following packages need to be installed:
* scikit-learn (using version 0.21.0)
* joblib (0.13.2)
* Theano (1.0.4) _or_ tensorflow (1.10.0)
* Keras (2.2.4)
* numpy (1.16.3)

Optional (for drawing plots):
* matplotlib (used version 3.0.3)

If recollecting data from sources and rebuilding datasets following libraries are needed:
* re
* csv
* xml
* random

Using different version for _joblib_ can render models unusable.

## STRUCTURE
Package _data_ contains datasets in folder _sources_ and source code 
used for extracting data from sources. Example usage is shown in script *classifiers_main.py*.
Package _classifiers_ contains trained models in folder _models_ and
training and testing sets in folder _datasets_. It also contains 
source code used when searching for best parameters and saving trained models.
Example usage is shown in script *classifiers_main.py*.
Package _evaluation_ contains source code for evaluation of models.
Originally used code is _evaluate.py_, corrected code is *evaluate_corrected.py*.
Example usage is shown in script *evaluation_main.py*.

## USAGE
To use modules, one can import them in terminal and use immediately
or write to one of scripts that end with _main.py_ and then execute script.

## SOURCES
Getting data again from sources is possible, but source files are not provided here.

Synonym sources:
* [SloWNet database](https://www.clarin.si/repository/xmlui/handle/11356/1026)

Antonym sources:
* [Thesaurus of Modern Slovene 1.0](https://www.clarin.si/repository/xmlui/handle/11356/1166)
* [SSKJ](https://fran.si/)

## Author
Jasmina Pegan

## LICENCE
