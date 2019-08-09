import torch
import torch.nn.functional as F

def random_sample(logits: torch.Tensor, temperature: float = 1.0) -> int:
    d = torch.distributions.Categorical(logits=logits / temperature)
    return d.sample().item()

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits

def top_p_sample(logits, top_p=0.9):
    filtered_logits = top_k_top_p_filtering(logits, top_p=top_p)
    return torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)


#         # Sample from logits
#         d = torch.distributions.Categorical(logits=logits[0, -1])
#         next_id = d.sample().item()

#         if next_id == END_OF_TEXT:
#             break

#         token_ids.append(next_id)
#         inputs = torch.LongTensor([[next_id]])

#     # Decode
#     return tokenizer.decode(token_ids)
