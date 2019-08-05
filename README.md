This repository together with https://github.com/HansBambel/gpt-2 has the work that I have done in the two months that I was at the University College Dublin in the Afflatus (http://afflatus.ucd.ie/) Creative Language System Group.

Main work was: Implement GPT-2 (done in Tensorflow for fine-tuning (other repository) and pytorch for further usage (this repository)).

## Requirements
- tqdm
- pytorch
- pytorch-transformers

# Usage
In `application.py` there are many functions that can be used to get insight to the generation of text from GPT-2. 
- Given an input and words (e.g. `data/emotions.txt`) calculate the probability of each of the words. This can be used to create wordvectors. This can be done with multiprocessing or singleprocessing.
- Given an input text (e.g. "Global warming is a") a language model (e.g. GPT-2 117M or 345M) is used to calculate the probability of the next word.
- Given one or more prompts it is possible to calculate the wordvectors of a list of words (we used `data/emotions.txt`). The code can be used to run on a normal machine or on a cluster. 
Somehow the cluster did not utilize multiprocessing correctly, therefore there is code for running it consequently and with multiprocessing.
- Given about 28.5k prompts we created for every single one the wordvector of emotions. This enables us to calculate a distance between them and a given input for further use.

In `HelperNotebook.ipynb` I did data cleaning and preparations for implementing functions in `application.py`.

### Models
GPT-2 117M is easily loadable through the pytorch_pretrained_bert module by using `from_pretrained('gpt2')` for the models. 

The 345M model can be loaded using `from_pretrained('gpt2-medium')`

##### GPT-2 pytorch implementation taken from https://github.com/allenai/lm-explorer