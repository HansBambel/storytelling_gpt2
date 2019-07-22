import torch
from torch.nn.functional import softmax
from gpt2 import GPT2LanguageModel
import numpy as np
from tqdm import tqdm
import pickle
import os
import multiprocessing as mp
import pytorch_pretrained_bert
import pytorch_transformers


def saveSingleMultipleTokens(model_name, words):
    model = load_model(model_name)
    singleWordTokens = []
    twoWordTokens = []
    threeWordTokens = []
    otherWordTokens = []
    for i, word in enumerate(words):

        encoding = np.array(model.tokenizer.encode(" " + word))
        if len(encoding) == 1:
            singleWordTokens.append(word)
        elif len(encoding) == 2:
            twoWordTokens.append(word)
        elif len(encoding) == 3:
            threeWordTokens.append(word)
        else:
            otherWordTokens.append(word)
            print(len(encoding), word, encoding)
    print("Single word tokens: ", len(singleWordTokens), "out of ", len(words))
    print("Two  word tokens: ", len(twoWordTokens), "out of ", len(words))
    print("Three  word tokens: ", len(threeWordTokens), "out of ", len(words))

    with open("SingleTokenEmotions.txt", "w") as f:
        for word in singleWordTokens:
            f.write(word+"\n")
    with open("MultipleTokenEmotions.txt", "w") as f:
        for word in otherWordTokens:
            f.write(word+"\n")

def load_model(model_name):
    # loads the required model
    if model_name == "117M":
        model = GPT2LanguageModel(model_name='117M')
    else:
        model = GPT2LanguageModel(model_name='345M')
    return model

def get_next_words(model, context, words, depth):
    # get next "word" given context
    if depth == 0:
        return words
    new_words = []
    for i, word in enumerate(words):
        logits = model.predict(context, word)
        # take the one with the highest probability
        # next_word_logit, next_index = logits.topk(1)
        next_index = torch.argmax(logits)
        next_word = model[next_index.item()]

        new_words.append(word + next_word)
    return get_next_words(model, context, new_words, depth-1)

def encode_words(model, words):
    # encode words to tokens and make a vector from them
    encoded_words = []
    longest_encoding = 0
    for i, word in enumerate(words):
        # Add an "is " so that the word gets encoded as if in a sentence and not single
        encoding = np.array(model.tokenizer.encode("is " + word))
        # Only use the actual word (this workaround is needed since whitespaces get removed in tokenizer of pytorch_transformers)
        encoding = encoding[1:]
        encoded_words.append(encoding)
        if len(encoding) > longest_encoding:
            longest_encoding = len(encoding)
    # create a padding on the right
    token_vector = np.ones((len(words), longest_encoding), dtype=int) * -1
    # write the tokens in the token_vector
    for i, encoded_word in enumerate(encoded_words):
        token_vector[i, :len(encoded_word)] = encoded_word

    # calc the single tokens (they are simple to calculate)
    same_tokens = np.expand_dims(token_vector[:, 0], axis=1) == np.expand_dims(token_vector[:, 0], axis=0)
    single_token_mask = np.sum(same_tokens, axis=1) == 1

    return encoded_words, token_vector, single_token_mask


def calc_wordvector(model, context, words):
    encoded_words, token_vector, single_token_mask = encode_words(model, words)

    probs = np.ones(token_vector.shape)
    for i, encoded_word in enumerate(tqdm(encoded_words)):
        new_context = context
        # go through the (at most two) tokens of the word and calc the probability
        for j, token in enumerate(encoded_word):
            logits = model.predict(new_context, None)
            probabilities = softmax(logits, dim=-1)
            probs[i, j] = probabilities[token].item()
            # Speedup: if first token does not occur another time we can stop for the word
            if (j==0 and token not in token_vector[~single_token_mask, 0]):
                break
            # feed in model with new context (oldContext+token) and get probability
            new_context += model.tokenizer.decode([token])
        # print(encoded_word, model.tokenizer.decode(encoded_word), probs)

    # some words have the same (start-)tokens --> get following tokens prob and scale
    multi_probs = probs[~single_token_mask]
    multi_tokens = token_vector[~single_token_mask]
    start_tokens = {token: 0.0 for token in np.unique(multi_tokens)}
    # sum up probability of second token with same first token
    for i, token in enumerate(multi_tokens):
        start_tokens[token[0]] += multi_probs[i, 1]
    # divide by summed probability to scale
    for start_token, summed_prob in start_tokens.items():
        multi_probs[multi_tokens[:, 0]==start_token, 1] /= summed_prob
    probs[~single_token_mask] = multi_probs
    # Note: for now only until the second token is taken into account (this assumes that after the second token the
    #       word is unique already from the others (e.g. dis-similar, dis-like))
    # this multiplies the first two tokens and writes it in the first column
    probs[~single_token_mask, 0] = np.prod(probs[~single_token_mask, :2], axis=1)
    # scale the first column
    return probs[:, 0]/np.sum(probs[:, 0])

def get_wordvector(model, context, words, useFile=True):
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
            wordvector = calc_wordvector(model, context, words)
            save = True
    else:
        wordvector = calc_wordvector(model, context, words)
        save = True

    if save and useFile:
        vectors[context] = wordvector
        with open(filename, "wb") as f:
            pickle.dump(vectors, f, protocol=pickle.HIGHEST_PROTOCOL)

    return wordvector

def get_topk_words(model_name, context, topk):
    model = load_model(model_name)

    logits = model.predict(context, None)

    probabilities = softmax(logits, dim=-1)
    best_logits, best_indices = logits.topk(topk)
    best_words = [model[idx.item()] for idx in best_indices]

    # the last number indicates how many tokens are calculated
    most_likely_words = get_next_words(model, context, best_words, 0)
    best_probabilities = probabilities[best_indices].tolist()

    return most_likely_words, best_probabilities

def comparison(word):
    trans_tokenizer = pytorch_transformers.tokenization_gpt2.GPT2Tokenizer.from_pretrained('gpt2')
    pretrainedBert_tokenizer = pytorch_pretrained_bert.tokenization_gpt2.GPT2Tokenizer.from_pretrained('gpt2')
    trans_encoding = trans_tokenizer.encode("is " + word)
    trans_encoding = trans_encoding[1:]
    pretrained_encoding = pretrainedBert_tokenizer.encode("is " + word)
    pretrained_encoding = pretrained_encoding[1:]
    print(f"Transformer encoding: {trans_encoding}")
    print(f"{[trans_tokenizer.decode(enc) for enc in trans_encoding]}")
    print(f"pretrained encoding: {pretrained_encoding}")
    print(f"{[pretrainedBert_tokenizer.decode([enc]) for enc in pretrained_encoding]}")

def multiprocess_prompts(prompt, model_name="345M", word_file="emotions.txt", useFile=True):
    with open(word_file, "r") as f:
        words = f.readlines()
    emotions = [e.strip() for e in words]
    model = load_model(model_name)
    get_wordvector(model, prompt, emotions, useFile)

def main():
    model_name = "345M"
    model = load_model(model_name)

    with open("emotions.txt", "r") as f:
        words = f.readlines()
    words = [e.strip() for e in words]

    # NOTE A trailing whitespace gives other output than without
    context = "abusing power. Emperor Caligula thinks that powers are \n"

    ### Comparison of pytorch_pretrained_ber and pytorch_transformers in encoding a word
    # comparison("disgraceful")

    ### load prompts for multiprocessing
    with open("GPT prompts.txt", "r") as f:
        prompts = f.readlines()
    prompts = prompts[:8]
    print(len(prompts))
    print(prompts[-1])
    pool = mp.Pool(mp.cpu_count())
    pool.map(multiprocess_prompts, prompts)

    ### filter words given list of words
    # print(f"Model: {model_name} Context = {context}")
    #
    # emotionVector = get_wordvector(model, context, words, useFile=True)
    #
    # sorted_idx = np.argsort(emotionVector)[::-1]
    # for i, idx in enumerate(sorted_idx):
    #     if i > 10:
    #         break
    #     print(f"{emotionVector[idx]:.4f}, {words[idx]}")

    # saveSingleMultipleTokens(model_name, emotions)


    ################ Display top 10 words ######################
    # most_likely_words, best_probabilities = get_topk_words(model_name, context, 10)
    #
    # print("Input: ", context)
    # for i, prob in enumerate(best_probabilities):
    #     print(f"{prob*100:.3f}%: {most_likely_words[i].strip()}")


if __name__ == '__main__':
    main()
