from bias_probs import biased_next_token_probs, next_token_from_probs, response_probability
from huggingface.train import SPECIAL_TOKENS
from wiki_proc import split_into_sentences
from flags import FLAGS
import torch

# full decoding function that samples a response based on a personality, conversation history, and word frequency biasers
def sample_seq(personality, freqs, context, history, tokenizer, model, nlp, current_output=None):
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    if current_output is None:
        current_output = []

    if FLAGS.insert_qa != "none":
        input_text = tokenizer.decode(history[-1])
        question = split_into_sentences(input_text)[-1]
        qa_output = nlp(question=question, context=context)
        use_qa = qa_output['score'] > FLAGS.qa_conf_thresh
        qa_answer = tokenizer.encode(qa_output['answer'])

        if FLAGS.insert_qa == "at_start" and use_qa:
            qa_inserted = True
            current_output = qa_answer
        else:
            qa_inserted = False
    else:
        use_qa = False
        qa_inserted=False

    if FLAGS.verbose:
        print(f'qa_output: {qa_output}')
    for i in range(FLAGS.max_length):
        if FLAGS.verbose:
            print(f"\nToken {len(current_output)}")
        # Calculate the (biased) next token probabilities
        probs = biased_next_token_probs(
            personality, 
            history, 
            tokenizer, 
            model, 
            current_output, 
            freqs, 
            method='cap', 
            )

        cur_qa_prob = probs[qa_answer[0]]
        # give verbose output
        if FLAGS.verbose:
            print(f'cur_qa_prob: {cur_qa_prob}')

        if (not qa_inserted) and (use_qa) and (cur_qa_prob == torch.max(probs)):
            qa_inserted = True
            if FLAGS.verbose:
                print("Inserted qa_answer!")
            current_output.extend(qa_answer)
        else:
            cur = next_token_from_probs(probs, special_tokens_ids, i)
            if FLAGS.verbose:
                print('output:', tokenizer.decode(cur.item()))
                
            if cur.item() in special_tokens_ids:
                break

            current_output.append(cur.item())
            # end loop if we are at the end of the response

    return current_output

#MODIFIED full decoding function that samples a response based on a personality, conversation history, and word frequency biasers
#makes a first pass to get response and then if qa_score is high enough, it places the qa response in the place with the highest probability of appearing
def mod_sample_seq(personality, freqs, context, history, tokenizer, model, nlp, current_output=None):
    def argmax(L):
        largest = max(L)
        for i in range(len(L)):
            if L[i] == largest:
                return i
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    if current_output is None:
        current_output = []

    input_text = tokenizer.decode(history[-1])
    question = split_into_sentences(input_text)[-1]
    qa_output = nlp(question=question, context=context)
    qa_answer = tokenizer.encode(qa_output['answer'])
    if FLAGS.verbose:
        print(f'qa_output: {qa_output}')

    #make first pass
    qa_probs = []

    for i in range(FLAGS.max_length):
        #give verbose output
        if FLAGS.verbose:
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
            ) #sorry for long number of inputs! (this subroutine is useful for scoring)
        
        qa_prob = new_probs[qa_answer[0]]
        qa_probs.append(qa_prob)

        if FLAGS.verbose:
            print(f'qa_prob: {qa_prob}')
            
        prev = next_token_from_probs(new_probs, special_tokens_ids, i)
        if FLAGS.verbose:
            print(f'output: {tokenizer.decode(prev.item())}')
        #end loop if we are at the end of the response
        if prev.item() in special_tokens_ids:
            break

        #otherwise add to the output
        current_output.append(prev.item())
    
    if qa_output['score'] > FLAGS.qa_conf_thresh:
        print(f"Inserted qa_answer! Non-QA response: {tokenizer.decode(current_output)}")
        qa_insertion_index = argmax(qa_probs) #gets index where qa_prob was maximized
        current_output = current_output[:qa_insertion_index] #get initial segment of current output up to i
        current_output.extend(qa_answer) #add in the qa answer

        #complete the remaining string
        for i in range(FLAGS.max_length - len(current_output)):
            #give verbose output
            if FLAGS.verbose:
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
                ) #sorry for long number of inputs! (this subroutine is useful for scoring)

            prev = next_token_from_probs(new_probs, special_tokens_ids,i)
            if FLAGS.verbose:
                print(f'output: {tokenizer.decode(prev.item())}')
            #end loop if we are at the end of the response
            if prev.item() in special_tokens_ids:
                break

            #otherwise add to the output
            current_output.append(prev.item())


    return current_output

# function to talk with bot with modified decoding strategy
# text is just a string from user to bot
# history is a tokenized history of the past conversation
def respond(text, history, personality, freqs, context, tokenizer, model, nlp):
    # make sure the text isn't empty
    if not text or text=='exit':
        print('Conversation terminated by user')
        return 0

    history.append(tokenizer.encode(text))
    with torch.no_grad():
        out_ids = sample_seq(personality, freqs, context, history, tokenizer, model, nlp)
    history.append(out_ids)
    history = history[-(2*FLAGS.max_history+1):]
    out_text = tokenizer.decode(out_ids, skip_special_tokens=False)
    print(out_text)
    return 1

#similar function to respond but it also calculates the response probabilities with and without bias
def respond_compare_bias(text, history, personality, freqs, context, tokenizer, model, nlp):
    #make sure the text isn't empty
    if not text or text=='exit':
        print('Conversation terminated by user')
        return 0

    history.append(tokenizer.encode(text))
    with torch.no_grad():
        out_ids = sample_seq(personality, freqs, context, history, tokenizer, model, nlp)

    out_text = tokenizer.decode(out_ids, skip_special_tokens=False)
    print(out_text)

    print("\nno bias probabilities")
    response_probability(out_text, personality, history, tokenizer, model, freqs, method='none')

    print("\nbias probabilities")
    response_probability(out_text, personality, history, tokenizer, model, freqs, method='cap')

    history.append(out_ids)
    history = history[-(2*FLAGS.max_history+1):]
    
    return 1

def run(personality, freq, context, tokenizer, model, nlp):
    history = []
    if FLAGS.verbose>2:
        while True and respond_compare_bias(input('>>> '), history, personality, freq, context, tokenizer, model, nlp):
            continue
    else:
        while True and respond(input('>>> '), history, personality, freq, context, tokenizer, model, nlp):
            continue