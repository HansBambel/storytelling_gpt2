import torch
from torch.nn.functional import softmax
from gpt2 import GPT2LanguageModel


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

def get_probabilities_words(model, context, words):
    # encode words to tokens
    # Add a whitespace to the comparisons if there is no trailing whitespace in context
    encoded_comp = model.tokenizer.encode(" " + words if context[-1] != " " else words)
    # If comparison is composed of multiple words find them one after the other
    probs = []
    new_context = context
    for token in encoded_comp:
        logits = model.predict(new_context, None)
        probabilities = softmax(logits, dim=-1)
        probs.append(probabilities[token].item())
        # feed in model with new context (oldContext+token) and get probability
        new_context += model.tokenizer.decode([token])
    print(encoded_comp, model.tokenizer.decode(encoded_comp), probs)
    # TODO calculate with bayes proper probability?
    return probs


def main():
    model_117M = GPT2LanguageModel(model_name='117M')
    model_345M = GPT2LanguageModel(model_name='345M')

    model_name = "117M"
    model = model_117M if model_name == "117M" else model_345M

    # NOTE A trailing whitespace gives other output than without
    context = "Global warming is a"
    comparisons = ["big myth", "hoax", "farce", "onomatopeia"]

    # sorted_probabilities, order = torch.sort(probabilities, descending=True)
    # words = [model[idx.item()].strip() for idx in order]
    # print(words)

    # filter words given comparison list
    print("Context = ", context)
    probs = []
    for comp in comparisons:
        probs.append(get_probabilities_words(model, context, comp))
        print(f'With probability of {probs[-1]}: "{comp}"')


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
