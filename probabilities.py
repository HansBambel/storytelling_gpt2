
from torch.nn.functional import softmax
from gpt2 import GPT2LanguageModel


def main():
    model_117M = GPT2LanguageModel(model_name='117M')
    model_345M = GPT2LanguageModel(model_name='345M')

    previous_str = "Global warming is a"
    next_str = None

    topk = 10

    logits = None
    model_name = "345M"
    if model_name == "117M":
        logits = model_117M.predict(previous_str, next_str)
    elif model_name == "345M":
        logits = model_345M.predict(previous_str, next_str)

    probabilities = softmax(logits, dim=-1)

    best_logits, best_indices = logits.topk(topk)
    best_words = []
    if model_name == "117M":
        best_words = [model_117M[idx.item()] for idx in best_indices]
    elif model_name == "345M":
        best_words = [model_345M[idx.item()] for idx in best_indices]
    best_probabilities = probabilities[best_indices].tolist()

    print("Input: ", previous_str)
    for i, prob in enumerate(best_probabilities):
        print(f"{best_words[i]:12} {prob*100:.3f}%")


if __name__ == '__main__':
    main()
