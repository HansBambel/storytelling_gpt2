import torch
from torch.nn.functional import softmax
from gpt2 import GPT2LanguageModel
from util.sampling import top_p_sample
import numpy as np
from tqdm import tqdm
import pickle
import os
import sys
import multiprocessing as mp
import time
from scipy.spatial.distance import cdist
import pytorch_transformers

class Pred():

    def __init__(self, model_name, filename=None):
        self.model_name = model_name
        self.load_model(model_name)
        self.setWords(filename)

    def load_model(self, model_name):
        # loads the required model
        if model_name == "117M":
            self.model = GPT2LanguageModel(model_name='117M')
        elif model_name == "345M":
            self.model = GPT2LanguageModel(model_name='345M')
        else:
            self.model = GPT2LanguageModel(model_name=model_name)

    def setWords(self, filename):
        if filename==None:
            self.words = None
        else:
            with open(filename, "r", encoding='utf-8') as f:
                words = f.readlines()
            self.words = [e.strip() for e in words]
            self.set_encoded_words()

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

        with open("data/SingleTokenEmotions.txt", "w") as f:
            for word in singleWordTokens:
                f.write(word+"\n")
        with open("data/MultipleTokenEmotions.txt", "w") as f:
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
        orig_logits = self.model.predict(context, None)
        orig_probabilities = softmax(orig_logits, dim=-1)
        probs[:, 0] = orig_probabilities[self.token_vector[:, 0]].detach().numpy()

        mask = self.token_vector[:, 1] != -1
        second_word_probs = np.ones(np.sum(mask))
        # for visualization of progress tqdm can be used (remove when running on cluster)
        for i, encoded_word in enumerate(tqdm(self.token_vector[mask])):
            # look at second token probability after first token is fed in
            new_context = context + self.model.tokenizer.decode([encoded_word[0]])
            logits = self.model.predict(new_context, None)
            probabilities = softmax(logits, dim=-1)
            second_word_probs[i] = probabilities[encoded_word[1]].item()
        probs[mask, 1] = second_word_probs

        # some words have the same (start-)tokens --> get following tokens prob and scale
        multi_probs = probs[~self.single_token_mask]
        multi_tokens = self.token_vector[~self.single_token_mask]
        start_tokens = {token: 0.0 for token in np.unique(multi_tokens[:, 0])}
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
        filename = f"data/wordvectors/{context.replace('/', '-')}.pkl"
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

    def generate_story_with_connectives(self, prompt, log_con, max_sentences=8):
        output = prompt + "\n"
        # generate until max sentences are reached or endoftext
        pbar = tqdm(total=max_sentences)
        sentence_count = 0
        while sentence_count < max_sentences:
            # Create until end of sentence is found
            end_of_sentence = False
            while not end_of_sentence:
                logits = self.model.predict(output)
                random_sample = top_p_sample(logits, top_p=0.9)
                next_word = self.model.tokenizer.decode(random_sample.item())
                output += next_word
                # check if last things in output are end of sentence
                if ("." in output[-2:]) or ("?" in output[-2:]) or ("!" in output[-2:]):
                    end_of_sentence = True
            sentence_count +=1
            if (sentence_count < max_sentences) and (sentence_count%2 == 0):
                output += " " + np.random.choice(log_con)
            pbar.update(1)
        pbar.close()
        return output

    def generate_story(self, prompt, numTokens):
        output = prompt
        currentTokens = 0
        while(True):
            logits = self.model.predict(output)
            random_sample = top_p_sample(logits, top_p=0.9)
            next_token = self.model.tokenizer.decode(random_sample.item())
            output += next_token
            currentTokens += 1
            # Finish sentence with a colon, question mark or exclamation mark
            if currentTokens > numTokens:
                if ("." in output[-2:]) or ("?" in output[-2:]) or ("!" in output[-2:]):
                    break
        return output


def getDistanceToOthers(vector, vectorMatrix, metric="euclidean", pred=None):
    if isinstance(vector, str):
        if pred==None:
            raise ValueError("No model to convert string specified to be used!")
        # If vector is a string --> use the model to create the emotion vector
        correctedVec = np.expand_dims(pred.get_wordvector(vector, False), axis=0)
    else:
        correctedVec = np.expand_dims(vector, axis=0)

    distanceToOthers = cdist(correctedVec, vectorMatrix, metric=metric)[0]
    return distanceToOthers



if __name__ == '__main__':
    # model_name = "345M"
    word_file = "data/emotions.txt"
    model_name = "models/writingprompts_117M"
    pred = Pred(model_name)

    # NOTE A trailing whitespace gives other output than without
    # context = "Global warming is"


    ################ process prompts one after the other ################
    # start_time = time.time()
    # with open("data/GPT prompts stripped.txt", "r", encoding='utf-8') as f:
    #     prompts = f.readlines()
    # print("Total number of prompts: ", len(prompts))

    # numberPrompts = 1500
    # promptsFromLine = int(sys.argv[1])
    # print(f"Processing prompts from line {promptsFromLine} to {promptsFromLine + numberPrompts}")
    # prompts = prompts[promptsFromLine:promptsFromLine + numberPrompts]
    #
    # for p in tqdm(prompts):
    #     pred.get_wordvector(p, useFile=True)
    # print(f"Processing {len(prompts)} prompts took about {time.time()-start_time:2f} seconds")


    ################ load prompts for multiprocessing ################
    # start_time = time.time()
    # print(f"Available cores: {mp.cpu_count()}")
    # with open("data/GPT prompts stripped.txt", "r", encoding='utf-8') as f:
    #     prompts = f.readlines()
    # # prompts = prompts[0:1000 + 2 * mp.cpu_count()]
    # print(len(prompts))
    # # print(f"Example prompt: {prompts[-1]}")
    # pool = mp.Pool(mp.cpu_count())
    # pool.map(pred.get_wordvector, prompts)
    # print(f"Required for {len(prompts)} prompts about {time.time() - start_time:.2f} seconds")

    ################ filter words given list of words and print the best fitting ones ################
    # print(f"Model: {model_name} Context = {context}")
    #
    # emotionVector = pred.get_wordvector(context, useFile=False)
    #
    # sorted_idx = np.argsort(emotionVector)[::-1]
    # for i, idx in enumerate(sorted_idx):
    #     if i > 10:
    #         break
    #     print(f"{emotionVector[idx]:.4f}, {pred.words[idx]}")

    # pred.saveSingleMultipleTokens(emotions)

    ################ Display top 10 words ################
    # most_likely_words, best_probabilities = pred.get_topk_words(context, 10, nTokens=0)
    #
    # print("Input: ", context)
    # for i, prob in enumerate(best_probabilities):
    #     print(f"{prob*100:.3f}%: {most_likely_words[i].strip()}")

    ################ Get distance of context to other contexts ################
    # with open("data/Wordvectors.pkl", "rb") as f:
    #     wordvectors = pickle.load(f)
    # with open("data/Contexts.pkl", "rb") as f:
    #     contexts = pickle.load(f)
    # with open("data/emotions.txt", "r") as f:
    #     emotions = f.readlines()
    # emotions = [e.strip() for e in emotions]
    #
    # context = "harvesting crops. Jose think that potatoes are"
    # n = 10
    # # print(context)
    # distanceToOthers = getDistanceToOthers(context, wordvectors, metric="euclidean", pred=pred)
    #
    # order = np.argsort(distanceToOthers)
    # print("Closest")
    # for ind in order[:n]:
    #     print(f"Dist: {distanceToOthers[ind]:.5f} Context: {contexts[ind]}")
    # print("Farthest")
    # for ind in order[::-1][:n]:
    #     print(f"Dist: {distanceToOthers[ind]:.5f} Context: {contexts[ind]}")

    ################ Create stories with logical connectives ################
    # use model and after a sentence use a logical connective and feed it in with that
    prompt = '''[WP] You’ve been stuck in a time loop that repeats the same day over and over. You’ve perfected every skill, you speak every language ever spoken. One day you go crazy, by the end of the day the entire town is dead. You wake up the next morning still covered in blood, the loop finally broke.'''
    # NOTE sentences can end with: . ." ! !" ? ?"
    log_connectives = ["So", "But", "Then", "Yet", "Thereafter", "Meanwhile", "Suddenly", "Surprisingly", "Mysteriously", "Nonetheless", "Nevertheless", "Similarly"]
    start_time = time.time()
    story = pred.generate_story_with_connectives(prompt, log_connectives, max_sentences=8)
    print(f"{time.time() - start_time} seconds")
    print(story)

    ################ Generate story from a NOC story prompt ################
    # nocPrompt = '''You’ve been stuck in a time loop that repeats the same day over and over. You’ve perfected every skill, you speak every language ever spoken. One day you go crazy, by the end of the day the entire town is dead. You wake up the next morning still covered in blood, the loop finally broke.'''
    # # promptWithNudge = nocPrompt + "\n[Prompt]"
    # promptWithNudge = "[Prompt] " + nocPrompt + "\n"
    # start_time = time.time()
    # story = pred.generate_story(promptWithNudge, numTokens=300)
    # print(f"{time.time()-start_time} seconds")
    # print(story)