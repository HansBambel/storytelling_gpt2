import torch
from torch.nn.functional import softmax
from gpt2 import GPT2LanguageModel
import numpy as np
from tqdm import tqdm
import time


def saveSingleMultipleTokens(probsWithWords):
    singleWordTokens = []
    otherWordTokens = []
    for entry in probsWithWords:
        if len(entry[0]) == 1:
            singleWordTokens.append(entry[1])
        else:
            otherWordTokens.append(entry[1])
    #         print(entry[1], len(entry[0]))
    print("Single word tokens: ", len(singleWordTokens), "out of ", len(probsWithWords))

    with open("SingleTokenEmotions.txt", "w") as f:
        for word in singleWordTokens:
            f.write(word+"\n")
    with open("MultipleTokenEmotions.txt", "w") as f:
        for word in otherWordTokens:
            f.write(word+"\n")

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

def get_wordvector(model, context, words):
    # encode words to tokens and make a vector from them
    encoded_words = []
    longest_encoding = 0
    for i, word in enumerate(words):
        # Add a whitespace to the comparisons if there is no trailing whitespace in context
        encoding = np.array(model.tokenizer.encode(" " + word if context[-1] != " " else word))
        encoded_words.append(encoding)
        if len(encoding) > longest_encoding:
            longest_encoding = len(encoding)
    token_vector = np.ones((len(words), longest_encoding), dtype=int) * -1
    for i, encoded_word in enumerate(encoded_words):
        token_vector[i, :len(encoded_word)] = encoded_word

    same_tokens = np.expand_dims(token_vector[:, 0], axis=1) == np.expand_dims(token_vector[:, 0], axis=0)
    single_token_mask = np.sum(same_tokens, axis=1) == 1

    probs = np.ones(token_vector.shape)
    # If comparison is composed of multiple words find them one after the other
    for i, encoded_word in enumerate(tqdm(encoded_words)):
        new_context = context
        for j, token in enumerate(encoded_word):
            logits = model.predict(new_context, None)
            probabilities = softmax(logits, dim=-1)
            probs[i, j] = probabilities[token].item()
            # feed in model with new context (oldContext+token) and get probability
            new_context += model.tokenizer.decode([token])
        # print(encoded_word, model.tokenizer.decode(encoded_word), probs)

    # some words have the same (start-)tokens --> get next tokens prob and scale
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
    # Note for now only until the second token is taken into account (this assumes that after the second token the
    #       word is different already from the others (e.g. dis-similar, dis-like)
    probs[~single_token_mask, 0] = np.prod(probs[~single_token_mask, :2], axis=1)

    return probs[:, 0]/np.sum(probs[:, 0])


def main():
    model_name = "117M"
    if model_name == "117M":
        model = GPT2LanguageModel(model_name='117M')
    else:
        model = GPT2LanguageModel(model_name='345M')

    with open("emotions.txt", "r") as f:
        words = f.readlines()

    emotions = [e.strip() for e in words]

    # NOTE A trailing whitespace gives other output than without
    context = "Global warming is"
    # comparisons = ["big myth", "myth", "fascinating", "hoax", "farce", "onomatopeia"]

    # filter words given comparison list
    print("Context = ", context)

    emotionVector = get_wordvector(model, context, emotions)

    sorted_idx = np.argsort(emotionVector)[::-1]
    for i, idx in enumerate(sorted_idx):
        if i > 10:
            break
        print(emotionVector[idx], emotions[idx])

    # saveSingleMultipleTokens(probsWithWords)


    ################ Display top 10 words ######################
    # topk = 10
    #
    # logits = model.predict(context, None)
    #
    # probabilities = softmax(logits, dim=-1)
    # best_logits, best_indices = logits.topk(topk)
    # best_words = [model[idx.item()] for idx in best_indices]
    #
    # most_likely_words = get_next_words(model, context, best_words, 0)
    #
    # best_probabilities = probabilities[best_indices].tolist()
    #
    # print("Input: ", context)
    # for i, prob in enumerate(best_probabilities):
    #     print(f"{prob*100:.3f}%: {most_likely_words[i].strip()}")


if __name__ == '__main__':
    main()
