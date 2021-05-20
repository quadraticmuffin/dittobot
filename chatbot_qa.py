from bias_probs import biased_next_token_probs, next_token_from_probs, response_probability
from huggingface.train import SPECIAL_TOKENS
from wiki_proc import split_into_sentences
from flags import FLAGS
import torch

# full decoding function that samples a response based on a personality, conversation history, and word frequency biasers
def sample_seq(history, personality, freqs, tokenizer, model, qa_answer, current_output=None):
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    if current_output is None:
        current_output = []
    start_len = len(current_output)
    qa_probs = torch.zeros((FLAGS.max_length,))
    for i in range(start_len, FLAGS.max_length):
        if FLAGS.verbose:
            print(f"\nToken {i}")
        probs = biased_next_token_probs(personality, history, 
            tokenizer, model, current_output, freqs)
        cur_qa_prob = probs[qa_answer[0]] if qa_answer else 0
        if FLAGS.verbose:
            print(f'cur_qa_prob: {cur_qa_prob}')
        qa_probs[i] = cur_qa_prob
        cur_token = next_token_from_probs(probs, special_tokens_ids, i)
        if FLAGS.verbose:
            print('output:', tokenizer.decode(cur_token.item()))
            
        if cur_token.item() in special_tokens_ids:
            break

        current_output.append(cur_token.item())
        # end loop if we are at the end of the response

    return current_output, qa_probs

# function to talk with bot with modified decoding strategy
# text is just a string from user to bot
# history is a tokenized history of the past conversation
def respond(text, history, personality, freqs, context, tokenizer, model, nlp):
    # make sure the text isn't empty
    if not text or text == 'exit':
        print('Conversation terminated by user')
        return 0

    history.append(tokenizer.encode(text))
    current_output = None

    use_qa = FLAGS.insert_qa_cond in ['at_start', 'if_most_likely', 'retro']
    if use_qa:
        question = split_into_sentences(text)[-1]
        qa_output = nlp(question=question, context=context)
        if FLAGS.verbose:
            print(f'qa_output: {qa_output}')
        use_qa = use_qa and qa_output['score'] > FLAGS.qa_conf_thresh
        qa_answer = tokenizer.encode(qa_output['answer'])
    else:
        qa_answer=None

    if use_qa and FLAGS.insert_qa_cond == "at_start":
        current_output = qa_answer
        
    with torch.no_grad():
        out_ids, qa_probs = sample_seq(history, personality, freqs, tokenizer, model, qa_answer, current_output=current_output)

    if use_qa and FLAGS.insert_qa_cond == "retro":
        insert_idx = torch.argmax(qa_probs)
        output_to_ans = out_ids[:insert_idx] + qa_answer
        if FLAGS.verbose:
            print(f'Answer inserted: {tokenizer.decode(output_to_ans)}')
            print('Resampling from answer...')
        with torch.no_grad():
            out_ids, _ = sample_seq(history, personality, freqs, tokenizer, model, None, current_output=output_to_ans)
    
    history.append(out_ids)
    history = history[-(2*FLAGS.max_history+1):]
    out_text = tokenizer.decode(out_ids, skip_special_tokens=False)
    print(out_text)
    if FLAGS.verbose > 2:
        print("\nno bias probabilities")
        response_probability(out_text, personality, history, tokenizer, model, freqs, use_bias=False)
        print("\nbias probabilities")
        response_probability(out_text, personality, history, tokenizer, model, freqs, use_bias=True)
    return 1


def run(personality, freq, context, tokenizer, model, nlp):
    history = []
    while True and respond(input('>>> '), history, personality, freq, context, tokenizer, model, nlp):
        continue