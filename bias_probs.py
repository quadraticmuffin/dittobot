import torch
import torch.nn.functional as F
from huggingface.interact import top_filtering
from huggingface.train import build_input_from_segments
import warnings
from flags import FLAGS

def next_token_probs(personality, history, tokenizer, model, current_output):
    instance = build_input_from_segments(personality, history, current_output, tokenizer, with_eos=False)
    input_ids = torch.tensor(instance["input_ids"], device=FLAGS.device).unsqueeze(0)
    token_type_ids = torch.tensor(instance["token_type_ids"], device=FLAGS.device).unsqueeze(0)

    output = model(input_ids, token_type_ids=token_type_ids)
    logits = output.logits

    if isinstance(logits, tuple):  # for gpt2 and maybe others
        logits = logits[0]
    logits = logits[0, -1, :] / FLAGS.temperature
    logits = top_filtering(logits, top_k=FLAGS.top_k, top_p=FLAGS.top_p)
    return F.softmax(logits, dim=-1)

def bias_probs(probs, bias, method):
    # bias must have positive values only
    if method == 'naive':
        for i in range(len(probs)):
            if i in bias:
                probs[i] *= bias[i]

    if method == 'cap':
        for i in range(len(probs)):
            if i in bias:
                probs[i] *= max(0.5, min(2, bias[i]))
    
    return probs

def biased_next_token_probs(personality, history, tokenizer, model, current_output, bias, method, verbose=False):
    vanilla_probs = next_token_probs(personality, history, tokenizer, model, current_output)
    
    # output verbose info if needed
    def ids_to_token_list(ids):
        return [tokenizer.convert_tokens_to_string(x) for x in tokenizer.convert_ids_to_tokens(ids)]
    if verbose:
        k_print = 8
        print("Before: ", ids_to_token_list(torch.topk(vanilla_probs, k_print)[1]))
        print(list(round(x.item(), 3) for x in torch.topk(vanilla_probs, k_print)[0]))

    new_probs = bias_probs(vanilla_probs, bias, method)

    if verbose:
        print("After: ", ids_to_token_list(torch.topk(new_probs, k_print)[1]))
        print(list(round(x.item(), 3) for x in torch.topk(new_probs, k_print)[0]))

    return new_probs

def next_token_from_probs(new_probs, special_tokens_ids, i):
    prev = torch.topk(new_probs, 1)[1] if FLAGS.no_sample else torch.multinomial(new_probs, 1)

    if i < FLAGS.min_length and prev.item() in special_tokens_ids:
        while prev.item() in special_tokens_ids:
            if new_probs.max().item() == 1:
                warnings.warn("Warning: model generating special token with probability 1.")
                break  # avoid infinitely looping over special token
            prev = torch.multinomial(new_probs, num_samples=1)
    return prev