from bias_probs import biased_next_token_probs, next_token_from_probs
from huggingface.train import SPECIAL_TOKENS, add_special_tokens_
from huggingface.utils import download_pretrained_model
from transformers import OpenAIGPTTokenizer, OpenAIGPTLMHeadModel, pipeline
from wordfreqs import freqs_from_tweets
from flags import FLAGS
from collections import defaultdict
import torch
import json

with torch.no_grad(): # voodoo to prevent automatic differentiation/ weird errors?
    nlp = pipeline("question-answering")

# full decoding function that samples a response based on a personality, conversation history, and word frequency biasers
def sample_seq(personality, freqs, context, history, tokenizer, model, current_output=None, verbose=False):
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    if current_output is None:
        current_output = []

    qa_output = nlp(question=tokenizer.decode(history[-1]), context=context)
    qa_answer = tokenizer.encode(qa_output['answer'])
    if verbose:
        print(f'qa_output: {qa_output}')
    prev_qa_prob = 0
    qa_inserted = False
    for i in range(FLAGS.max_length):
        if verbose:
            print(f"\nToken {i}")
        # Calculate the (biased) next token probabilities
        probs = biased_next_token_probs(
            personality, 
            history, 
            tokenizer, 
            model, 
            current_output, 
            freqs, 
            method='cap', 
            verbose=verbose)

        cur_qa_prob = probs[qa_answer[0]]
        # give verbose output
        if verbose:
            print(f'cur_qa_prob: {cur_qa_prob}')

        if (not qa_inserted) and (qa_output['score'] > FLAGS.qa_conf_thresh) \
        and cur_qa_prob == torch.max(probs):
            qa_inserted = True
            print("Inserted qa_answer!")
            current_output.extend(qa_answer)
        else:
            cur = next_token_from_probs(probs, special_tokens_ids, i)
            if verbose:
                print('output:', tokenizer.decode(cur.item()))
                
            if cur.item() in special_tokens_ids:
                break

            current_output.append(cur.item())
            prev_qa_prob = cur_qa_prob
            # end loop if we are at the end of the response

    return current_output

# function to talk with bot with modified decoding strategy
# text is just a string from user to bot
# history is a tokenized history of the past conversation
def respond(text, history, personality, freqs, context, verbose=True):
    # make sure the text isn't empty
    if not text or text=='exit':
        print('Conversation terminated by user')
        return 0

    history.append(tokenizer.encode(text))
    with torch.no_grad():
        out_ids = sample_seq(personality, freqs, context, history, tokenizer, model, verbose=verbose)
    history.append(out_ids)
    history = history[-(2*FLAGS.max_history+1):]
    out_text = tokenizer.decode(out_ids, skip_special_tokens=False)
    print(out_text)
    return 1

def run(info, freq, context, verbose=False):
    history = []
    while True and respond(input('>>> '), history, info, freq, context, verbose=verbose):
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
    
    obama_context = """Barack Obama is an American politician and attorney 
    who served as the 44th president of the United States from 2009 to 2017. 
    A member of the Democratic Party, Obama was the first African-American president of the United States. 
    He previously served as a U.S. senator from Illinois from 2005 to 2008 and as an Illinois state senator 
    from 1997 to 2004. Obama was born in Honolulu, Hawaii. After graduating from Columbia University in 1983, 
    he worked as a community organizer in Chicago. In 1988, he enrolled in Harvard Law School, 
    where he was the first black president of the Harvard Law Review. After graduating, 
    he became a civil rights attorney and an academic, teaching constitutional law at the 
    University of Chicago Law School from 1992 to 2004. Turning to elective politics, h
    e represented the 13th district in the Illinois Senate from 1997 until 2004, when he ran for the U.S. Senate. 
    Obama received national attention in 2004 with his March Senate primary win, 
    his well-received July Democratic National Convention keynote address, and his landslide November election 
    to the Senate. In 2008, he was nominated by the Democratic Party for president a year after beginning his 
    campaign, and after a close primary campaign against Hillary Clinton. Obama was elected over Republican 
    nominee John McCain in the general election and was inaugurated alongside his running mate, Joe Biden, on 
    January 20, 2009. Nine months later, he was named the 2009 Nobel Peace Prize laureate."""

    run(obama_info, freq_diff, obama_context, verbose=True)
    