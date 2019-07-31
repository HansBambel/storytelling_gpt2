## Requirements
- tqdm
- pytorch
- pytorch-transformers

# Usage
- Given an input and words (e.g. `emotions.txt`) calculate the probability of each of the words. This can be used to create wordvectors. This can be done with multiprocessing or singleprocessing.
- Given an input text (e.g. "Global warming is a") a language model (e.g. GPT-2 117M or 345M) is used to calculate the probability of the next word.

### Models
GPT-2 117M is easily loadable through the pytorch_pretrained_bert module by using `from_pretrained('gpt2')` for the models. 

The 345M model can be loaded using `from_pretrained('gpt2-medium')`

##### GPT-2 pytorch implementation taken from https://github.com/allenai/lm-explorer