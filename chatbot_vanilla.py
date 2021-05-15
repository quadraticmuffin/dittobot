from bias_probs import biased_next_token_probs, next_token_from_probs
from huggingface.train import SPECIAL_TOKENS, add_special_tokens_
from huggingface.utils import download_pretrained_model
from transformers import OpenAIGPTTokenizer, OpenAIGPTLMHeadModel
from wordfreqs import freqs_from_tweets
from flags import FLAGS
from collections import defaultdict
import torch
import json

# full decoding function that samples a response based on a personality, conversation history, and word frequency biasers
def sample_seq(personality, freqs, history, tokenizer, model, current_output=None, verbose=False):
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    if current_output is None:
        current_output = []

    for i in range(FLAGS.max_length):
        # give verbose output
        if verbose:
          print(f"\nToken {i}")

        # Calculate the (biased) next token probabilities
        new_probs = biased_next_token_probs(
            personality, 
            history, 
            tokenizer, 
            model, 
            current_output, 
            freqs, 
            method='cap', 
            verbose=verbose) #sorry for long number of inputs! (this subroutine is useful for scoring)

        # Sample next token (which is confusingly named prev) according to the prob distribution
        prev = next_token_from_probs(new_probs, special_tokens_ids, i)

        #end loop if we are at the end of the response
        if prev.item() in special_tokens_ids:
            break

        #otherwise add to the output
        current_output.append(prev.item())

    return current_output

#function to talk with bot with modified decoding strategy
#text is just a string from the user to the bot and history is a tokenized history of the past conversation
def respond(text, history, personality, freqs, verbose=True):
    #make sure the text isn't empty
    if not text or text=='exit':
        print('Conversation terminated by user')
        return 0

    history.append(tokenizer.encode(text))
    with torch.no_grad():
        out_ids = sample_seq(personality, freqs, history, tokenizer, model, verbose=verbose)
    history.append(out_ids)
    history = history[-(2*FLAGS.max_history+1):]
    out_text = tokenizer.decode(out_ids, skip_special_tokens=False)
    print(out_text)
    return 1

def run(info, freq, verbose=False):
    history = []
    while True and respond(input('>>> '), history, info, freq, verbose=verbose):
        continue

if __name__=="__main__":
    tokenizer_class, model_class = OpenAIGPTTokenizer, OpenAIGPTLMHeadModel
    print('Getting model...')
    pretrained_model = download_pretrained_model() #downloads the pretrained model from S3
    model = model_class.from_pretrained(pretrained_model)
    tokenizer = tokenizer_class.from_pretrained(pretrained_model)

    add_special_tokens_(model, tokenizer)
    screen_name = 'BarackObama'
    with open(f'timelines/{screen_name}.json', 'r', encoding='utf-8') as f:
        user_tweets = json.load(f)
    user_freqs = freqs_from_tweets(user_tweets, tokenizer)
    with open(f'word-freqs/persona.json', 'r', encoding='utf-8') as f:
        persona_freqs = json.load(f)
    persona_freqs = defaultdict(int, {int(k):v for k, v in persona_freqs.items()})
    vocab = {**user_freqs, **persona_freqs}.keys()
    freq_diff = {k: user_freqs[k] / (persona_freqs[k] or 1) for k in vocab}
    
    obama_info = tokenizer.batch_encode_plus(["My name is Barack Obama.", "I was the 44th president of the United States.",
              "A member of the Democratic Party, I was the first African-American president of the United States.", 
              "I served as a U.S. senator from Illinois from 2005 to 2008 and as an Illinois state senator from 1997 to 2004"],
              "I was born in Honolulu, Hawaii.")['input_ids']

    run(obama_info, freq_diff)
    