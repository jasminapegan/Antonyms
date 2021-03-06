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
Package _data_ contains source code used for extracting data from sources. Example usage is shown in script *classifiers_main.py*. One of datasets is in folder _sources_, others will be on Dropbox (link will be added in a day or two) because files were too big to upload on Github.

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
Extracting data again from sources is possible, but I do not provide source files.

Synonym sources:
* [SloWNet database](https://www.clarin.si/repository/xmlui/handle/11356/1026)

Antonym sources:
* [Thesaurus of Modern Slovene 1.0](https://www.clarin.si/repository/xmlui/handle/11356/1166)
* [SSKJ](https://fran.si/)

## Author
Jasmina Pegan

## LICENSE
<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Licenca Creative Commons" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />To delo je objavljeno pod <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">licenco Creative Commons Priznanje avtorstva-Deljenje pod enakimi pogoji 4.0 Mednarodna</a>.
