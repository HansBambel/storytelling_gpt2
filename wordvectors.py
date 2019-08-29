#!/usr/bin/env python3
# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Conditional text generation with the auto-regressive models of the library (GPT/GPT-2/Transformer-XL/XLNet)
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from tqdm import trange, tqdm

import torch
import torch.nn.functional as F
import numpy as np
import pickle

from pytorch_transformers import GPT2LMHeadModel, GPT2Tokenizer


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

MODEL_CLASSES = {
    'gpt2': (GPT2LMHeadModel, GPT2Tokenizer)
}



def set_seed(seed, n_gpu):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


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

def calc_wordvector(model, words, context, device=torch.device('cpu')):
    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.unsqueeze(0)
    with torch.no_grad():
        inputs = {'input_ids': context}

        outputs = model(**inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet (cached hidden-states)
        next_token_logits = outputs[0][0, -1, :]
        # filter given word list
        next_token_probs = F.softmax(next_token_logits, dim=-1)
        wordvector = next_token_probs[words]
    return wordvector/torch.sum(wordvector)

def sample_sequence(model, length, context, num_samples=1, temperature=1, top_k=0, top_p=0.0, device='cpu'):
    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.unsqueeze(0).repeat(num_samples, 1)
    sentence_end_tokens = torch.tensor([[0], [13], [30], [526], [2474], [1701], [50256]], dtype=torch.long,
                                       device=device)
    sentence_count = torch.tensor(0, dtype=torch.int, device=device)
    generated = context
    with torch.no_grad():
        # Generate length number of sentences
        pbar = tqdm(total=length, desc="Sentences")
        while sentence_count < length:
            end_of_sentence = torch.tensor(0, dtype=torch.int8)
            while end_of_sentence == 0:

                inputs = {'input_ids': generated}

                outputs = model(**inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet (cached hidden-states)
                next_token_logits = outputs[0][0, -1, :] / temperature
                filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
                generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
                if torch.any(next_token == sentence_end_tokens):
                    end_of_sentence = 1
            sentence_count += 1
            pbar.update(1)
        pbar.close()
    return generated

def main():

    # Convert emotions to tokens
    with open("data/emotions.txt", "r") as f:
        emotions = f.readlines()
    emotions = [emo.strip() for emo in emotions]

    # My Configs
    # args.seed = np.random.randint(1000000)
    seed = 1337
    model_type = 'gpt2'
    model_name_or_path = 'gpt2'
    # model_name_or_path = "gpt2-medium"
    # model_name_or_path = "models/gpt-2-large"
    no_cuda = False


    device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()

    set_seed(seed, n_gpu)

    model_type = model_type.lower()
    model_class, tokenizer_class = MODEL_CLASSES[model_type]
    # tokenizer is not loaded converted model --> workaround: use vanilla tokenizer
    tokenizer = tokenizer_class.from_pretrained("gpt2-medium")
    model = model_class.from_pretrained(model_name_or_path)
    model.to(device)
    model.eval()


    tokenized_emotions = [tokenizer.encode(". " + con)[1:] for con in emotions]
    single_token_emotions = [tokenizer.decode(con) for con in tokenized_emotions if len(con) == 1]
    # multi_tokens = [tokenizer.decode(con) for con in tokenized_connectives if len(con) > 1]
    tokenized_single_emotions = torch.tensor([tokenizer.encode("."+con)[1:] for con in single_token_emotions], dtype=torch.long, device=device)

    with open("data/GPT prompts.txt", "r", encoding="utf-8") as f:
        prompts = f.readlines()

    wordvectors = np.zeros((len(prompts), len(single_token_emotions)))
    for i, prompt in enumerate(tqdm(prompts)):
        context_tokens = tokenizer.encode(prompt.strip())

        wordvector = calc_wordvector(
            model=model,
            words=tokenized_single_emotions,
            context=context_tokens,
            device=device,
        )
        wordvectors[i] = np.array(wordvector.cpu().squeeze())
    with open("data/wordvectors_117M.pkl", "wb") as wvf:
        # Save wordvectors
        pickle.dump(wordvectors, wvf)


if __name__ == '__main__':
    main()