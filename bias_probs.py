import torch
import torch.nn.functional as F
from huggingface.interact import top_filtering
from huggingface.train import build_input_from_segments
import warnings
from flags import FLAGS
import transformers

def next_token_probs(personality, history, tokenizer, model, current_output):
    instance = build_input_from_segments(personality, history, current_output, tokenizer, with_eos=False)
    input_ids = torch.tensor(instance["input_ids"], device=FLAGS.device).unsqueeze(0)
    token_type_ids = torch.tensor(instance["token_type_ids"], device=FLAGS.device).unsqueeze(0)

    output = model(input_ids, token_type_ids=token_type_ids)
    if transformers.__version__ >= '4':
        logits = output.logits
    else:
        logits = output

    if isinstance(logits, tuple):  # for gpt2 and maybe others
        logits = logits[0]
    logits = logits[0, -1, :] / FLAGS.temperature
    logits = top_filtering(logits, top_k=FLAGS.top_k, top_p=FLAGS.top_p)
    return F.softmax(logits, dim=-1)

def bias_probs(probs, bias):
    # bias must have positive values only
    for i in range(len(probs)):
        if i in bias:
            if FLAGS.bias_method == 'naive':
                probs[i] *= bias[i]
            elif FLAGS.bias_method == 'cap':
                probs[i] *= max(1/FLAGS.bias_cap, min(FLAGS.bias_cap, bias[i]))
            elif FLAGS.bias_method == 'scale':
                probs[i] *= bias[i] * FLAGS.bias_scale
    
    return probs

def biased_next_token_probs(personality, history, tokenizer, model, current_output, bias):
    vanilla_probs = next_token_probs(personality, history, tokenizer, model, current_output)

    # output verbose info if needed
    def ids_to_token_list(ids):
        return [tokenizer.convert_tokens_to_string(x) for x in tokenizer.convert_ids_to_tokens(ids)]
    if FLAGS.verbose>1:
        print("Vanilla Probs: ", ids_to_token_list(torch.topk(vanilla_probs, 8)[1]))
        print('\t', list(round(x.item(), 3) for x in torch.topk(vanilla_probs, 8)[0]))

    new_probs = bias_probs(vanilla_probs, bias)

    if FLAGS.verbose>0:
        print("Biased Probs: ", ids_to_token_list(torch.topk(new_probs, 8)[1]))
        print('\t', list(round(x.item(), 3) for x in torch.topk(new_probs, 8)[0]))

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

# given some response and model parameters, this function evalutes the log probability of our model 
# outputting that response
def response_probability(response, personality, history, tokenizer, model, bias, use_bias=False, current_output=None):
    if current_output is None:
        current_output = []
    logprob = 0
    enc_txt = tokenizer.encode(response)
    for next_token in enc_txt: # iterate over enc_txt tokens
        probs = next_token_probs(personality, history, tokenizer, model, current_output)
        if use_bias:
            probs = bias_probs(probs, bias)

        # update total probability
        prob = probs[next_token]/torch.sum(probs)
        logprob += torch.log(prob).item() if prob > 0 else -100000
        print(f'\tprob: {round(prob.item(), 3)}, logprob: {round(logprob, 3)}, ' +
                f'token: {tokenizer.decode(next_token)}')

        # add this token to output
        current_output.append(next_token)

    return logprob