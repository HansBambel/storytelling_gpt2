import torch
from torch.nn.functional import softmax
from gpt2 import GPT2LanguageModel
import numpy as np
from tqdm import tqdm
import pickle
import os
import multiprocessing as mp
# import pytorch_pretrained_bert
import pytorch_transformers

class Pred():

    def __init__(self, model_name, filename):
        self.load_model(model_name)
        self.setWords(filename)
        self.set_encoded_words()

    def load_model(self, model_name):
        # loads the required model
        if model_name == "117M":
            self.model = GPT2LanguageModel(model_name='117M')
        else:
            self.model = GPT2LanguageModel(model_name='345M')

    def setWords(self, filename):
        with open(filename, "r", encoding='utf-8') as f:
            words = f.readlines()
        self.words = [e.strip() for e in words]

    def set_encoded_words(self):
        # encode words to tokens and make a vector from them
        encoded_words = []
        longest_encoding = 0
        for i, word in enumerate(self.words):
            # Add an "is " so that the word gets encoded as if in a sentence and not single
            encoding = np.array(self.model.tokenizer.encode("is " + word))
            # Only use the actual word (this workaround is needed since whitespaces get removed in tokenizer of pytorch_transformers)
            encoding = encoding[1:]
            encoded_words.append(encoding)
            if len(encoding) > longest_encoding:
                longest_encoding = len(encoding)
        # create a padding on the right
        token_vector = np.ones((len(self.words), longest_encoding), dtype=int) * -1
        # write the tokens in the token_vector
        for i, encoded_word in enumerate(encoded_words):
            token_vector[i, :len(encoded_word)] = encoded_word

        # calc the single tokens (they are simple to calculate)
        same_tokens = np.expand_dims(token_vector[:, 0], axis=1) == np.expand_dims(token_vector[:, 0], axis=0)
        single_token_mask = np.sum(same_tokens, axis=1) == 1

        self.encoded_words = encoded_words
        self.token_vector = token_vector
        self.single_token_mask = single_token_mask

    def saveSingleMultipleTokens(self):
        singleWordTokens = []
        twoWordTokens = []
        threeWordTokens = []
        otherWordTokens = []
        for i, word in enumerate(self.words):

            encoding = np.array(self.model.tokenizer.encode(" " + word))
            if len(encoding) == 1:
                singleWordTokens.append(word)
            elif len(encoding) == 2:
                twoWordTokens.append(word)
            elif len(encoding) == 3:
                threeWordTokens.append(word)
            else:
                otherWordTokens.append(word)
                print(len(encoding), word, encoding)
        print("Single word tokens: ", len(singleWordTokens), "out of ", len(self.words))
        print("Two  word tokens: ", len(twoWordTokens), "out of ", len(self.words))
        print("Three  word tokens: ", len(threeWordTokens), "out of ", len(self.words))

        with open("SingleTokenEmotions.txt", "w") as f:
            for word in singleWordTokens:
                f.write(word+"\n")
        with open("MultipleTokenEmotions.txt", "w") as f:
            for word in otherWordTokens:
                f.write(word+"\n")

    def get_next_words(self, context, given_words, depth):
        # get next "word" given context
        if depth == 0:
            return given_words
        new_words = []
        for i, word in enumerate(given_words):
            logits = self.model.predict(context, word)
            # take the one with the highest probability
            # next_word_logit, next_index = logits.topk(1)
            next_index = torch.argmax(logits)
            next_word = self.model[next_index.item()]

            new_words.append(word + next_word)
        return self.get_next_words(context, new_words, depth-1)


    def calc_wordvector(self, context):
        probs = np.ones(self.token_vector.shape)
        for i, encoded_word in enumerate(tqdm(self.encoded_words)):
            new_context = context
            # go through the (at most two) tokens of the word and calc the probability
            for j, token in enumerate(encoded_word):
                logits = self.model.predict(new_context, None)
                probabilities = softmax(logits, dim=-1)
                probs[i, j] = probabilities[token].item()
                # Speedup: if first token does not occur another time we can stop for the word
                if (j==0 and token not in self.token_vector[~self.single_token_mask, 0]):
                    break
                # feed in model with new context (oldContext+token) and get probability
                new_context += self.model.tokenizer.decode([token])
            # print(encoded_word, model.tokenizer.decode(encoded_word), probs)

        # some words have the same (start-)tokens --> get following tokens prob and scale
        multi_probs = probs[~self.single_token_mask]
        multi_tokens = self.token_vector[~self.single_token_mask]
        start_tokens = {token: 0.0 for token in np.unique(multi_tokens)}
        # sum up probability of second token with same first token
        for i, token in enumerate(multi_tokens):
            start_tokens[token[0]] += multi_probs[i, 1]
        # divide by summed probability to scale
        for start_token, summed_prob in start_tokens.items():
            multi_probs[multi_tokens[:, 0]==start_token, 1] /= summed_prob
        probs[~self.single_token_mask] = multi_probs
        # Note: for now only until the second token is taken into account (this assumes that after the second token the
        #       word is unique already from the others (e.g. dis-similar, dis-like))
        # this multiplies the first two tokens and writes it in the first column
        probs[~self.single_token_mask, 0] = np.prod(probs[~self.single_token_mask, :2], axis=1)
        # scale the first column
        return probs[:, 0]/np.sum(probs[:, 0])

    def get_wordvector(self, context, useFile=True):
        # get rid of linebreaks
        context = context.strip()
        save = False
        vectors = {}
        filename = f"wordvectors/{context}.pkl"
        if os.path.isfile(filename) and useFile:
            with open(filename, "rb") as f:
                vectors = pickle.load(f)
            # If the context was also calculated --> reuse it
            if context in vectors:
                wordvector = vectors[context]
            # else calculate it and write it in the file
            else:
                wordvector = self.calc_wordvector(context)
                save = True
        else:
            wordvector = self.calc_wordvector(context)
            save = True

        if save and useFile:
            vectors[context] = wordvector
            with open(filename, "wb") as f:
                pickle.dump(vectors, f, protocol=pickle.HIGHEST_PROTOCOL)

        return wordvector

    def get_topk_words(self, context, topk, nTokens=0):
        logits = self.model.predict(context, None)

        probabilities = softmax(logits, dim=-1)
        best_logits, best_indices = logits.topk(topk)
        best_words = [self.model[idx.item()] for idx in best_indices]

        # the last number indicates how many tokens are calculated
        most_likely_words = self.get_next_words(context, best_words, nTokens)
        best_probabilities = probabilities[best_indices].tolist()

        return most_likely_words, best_probabilities

    # def comparison(self, word):
    #     trans_tokenizer = pytorch_transformers.tokenization_gpt2.GPT2Tokenizer.from_pretrained('gpt2')
    #     pretrainedBert_tokenizer = pytorch_pretrained_bert.tokenization_gpt2.GPT2Tokenizer.from_pretrained('gpt2')
    #     trans_encoding = trans_tokenizer.encode("is " + word)
    #     trans_encoding = trans_encoding[1:]
    #     pretrained_encoding = pretrainedBert_tokenizer.encode("is " + word)
    #     pretrained_encoding = pretrained_encoding[1:]
    #     print(f"Transformer encoding: {trans_encoding}")
    #     print(f"{[trans_tokenizer.decode(enc) for enc in trans_encoding]}")
    #     print(f"pretrained encoding: {pretrained_encoding}")
    #     print(f"{[pretrainedBert_tokenizer.decode([enc]) for enc in pretrained_encoding]}")


if __name__ == '__main__':
    model_name = "345M"
    pred = Pred(model_name, "emotions.txt")

    # NOTE A trailing whitespace gives other output than without
    context = "I had a great time "


    ### Comparison of pytorch_pretrained_ber and pytorch_transformers in encoding a word
    # comparison("disgraceful")

    ### define inner function for multiprocessing
    def multiprocess_prompts(prompt):
        pred.get_wordvector(prompt, useFile=True)


    ### load prompts for multiprocessing
    # print(f"Available cores: {mp.cpu_count()}")
    # with open("GPT prompts.txt", "r", encoding='utf-8') as f:
    #     prompts = f.readlines()
    # prompts = prompts[8:16]
    # print(len(prompts))
    # print(f"Example prompt: {prompts[-1]}")
    # pool = mp.Pool(mp.cpu_count())
    # pool.map(pred.get_wordvector, prompts)

    ### filter words given list of words
    # print(f"Model: {model_name} Context = {context}")
    #
    # emotionVector = pred.get_wordvector(context, useFile=True)
    #
    # sorted_idx = np.argsort(emotionVector)[::-1]
    # for i, idx in enumerate(sorted_idx):
    #     if i > 10:
    #         break
    #     print(f"{emotionVector[idx]:.4f}, {pred.words[idx]}")

    # pred.saveSingleMultipleTokens(emotions)

    ################ Display top 10 words ######################
    most_likely_words, best_probabilities = pred.get_topk_words(context, 10, nTokens=4)

    print("Input: ", context)
    for i, prob in enumerate(best_probabilities):
        print(f"{prob*100:.3f}%: {most_likely_words[i].strip()}")
