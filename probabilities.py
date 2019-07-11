
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
        next_word_logit, next_index = logits.topk(1)
        next_word = model[next_index.item()]

        new_words.append(word + next_word)
    return get_next_words(model, context, new_words, depth-1)


def main():
    model_117M = GPT2LanguageModel(model_name='117M')
    model_345M = GPT2LanguageModel(model_name='345M')

    model_name = "345M"
    model = model_117M if model_name == "117M" else model_345M

    context = "Global warming is a"
    next_str = None

    topk = 10

    logits = model.predict(context, next_str)

    probabilities = softmax(logits, dim=-1)

    best_logits, best_indices = logits.topk(topk)
    best_words = [model[idx.item()] for idx in best_indices]

    most_likely_words = get_next_words(model, context, best_words, 1)

    best_probabilities = probabilities[best_indices].tolist()

    print("Input: ", context)
    for i, prob in enumerate(best_probabilities):
        print(f"{prob*100:.3f}%: {most_likely_words[i].strip()}")


if __name__ == '__main__':
    main()
