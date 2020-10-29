# efselab-wrapper

Wrapper for the swedish annotation pipeline https://github.com/robertostling/efselab. Easy to use setup and wrapper python code for processing a corpus as a list of strings into a list of list of tokens, annotated with POS tags, lemmas and NER-tags.

## Dependencies

- Python modules
  - Cython
  - 
- C compiler

Tested with python 3.8 and Apple clang version 11.0.0

## Setup


1. Clone the project
2. `$ cd setup/`
3. `$ bash _setup.sh`
4. `$ cd ../ && python test.py`

You can now run aribrary python scripts in the root directory, importing the pipeline functions