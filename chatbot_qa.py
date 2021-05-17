from torch.functional import split
from bias_probs import biased_next_token_probs, next_token_from_probs
from huggingface.train import SPECIAL_TOKENS, add_special_tokens_
from huggingface.utils import download_pretrained_model
from transformers import OpenAIGPTTokenizer, OpenAIGPTLMHeadModel, pipeline
from twitter_proc import freq_diffs
from wiki_proc import split_into_sentences
from flags import FLAGS
import torch

# full decoding function that samples a response based on a personality, conversation history, and word frequency biasers
def sample_seq(personality, freqs, context, history, tokenizer, model, nlp, current_output=None, verbose=False):
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    if current_output is None:
        current_output = []

    input_text = tokenizer.decode(history[-1])
    question = split_into_sentences(input_text)[-1]
    qa_output = nlp(question=question, context=context)
    qa_answer = tokenizer.encode(qa_output['answer'])
    if verbose:
        print(f'qa_output: {qa_output}')
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
            # end loop if we are at the end of the response

    return current_output

def argmax(L):
  largest = max(L)
  for i in range(len(L)):
    if L[i] == largest:
      return i

#MODIFIED full decoding function that samples a response based on a personality, conversation history, and word frequency biasers
#makes a first pass to get response and then if qa_score is high enough, it places the qa response in the place with the highest probability of appearing
def mod_sample_seq(personality, freqs, context, history, tokenizer, model, nlp, current_output=None, verbose=False):
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    if current_output is None:
        current_output = []

    input_text = tokenizer.decode(history[-1])
    question = split_into_sentences(input_text)[-1]
    qa_output = nlp(question=question, context=context)
    qa_answer = tokenizer.encode(qa_output['answer'])
    if verbose and FLAGS.use_qa:
        print(f'qa_output: {qa_output}')

    #make first pass
    qa_probs = []

    for i in range(FLAGS.max_length):
        #give verbose output
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
        
        qa_prob = new_probs[qa_answer[0]]
        qa_probs.append(qa_prob)

        if verbose and FLAGS.use_qa:
            print(f'qa_prob: {qa_prob}')
            
        prev = next_token_from_probs(new_probs, special_tokens_ids, i)
        if verbose:
            print(f'output: {tokenizer.decode(prev.item())}')
        #end loop if we are at the end of the response
        if prev.item() in special_tokens_ids:
            break

        #otherwise add to the output
        current_output.append(prev.item())
    
    if FLAGS.use_qa and qa_output['score'] > FLAGS.qa_conf_thresh:
        print(f"Inserted qa_answer! Non-QA response: {tokenizer.decode(current_output)}")
        qa_insertion_index = argmax(qa_probs) #gets index where qa_prob was maximized
        current_output = current_output[:qa_insertion_index] #get initial segment of current output up to i
        current_output.extend(qa_answer) #add in the qa answer

        #complete the remaining string
        for i in range(FLAGS.max_length - len(current_output)):
            #give verbose output
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

            prev = next_token_from_probs(new_probs, special_tokens_ids,i)
            if verbose:
                print(f'output: {tokenizer.decode(prev.item())}')
            #end loop if we are at the end of the response
            if prev.item() in special_tokens_ids:
                break

            #otherwise add to the output
            current_output.append(prev.item())


    return current_output

def respond_qa_start(text, history, personality, freqs, context, tokenizer, model, nlp, verbose=True):
    # make sure the text isn't empty
    if not text or text=='exit':
        print('Conversation terminated by user')
        return 0

    history.append(tokenizer.encode(text))

    question = split_into_sentences(text)[-1]
    qa_output = nlp(question=question, context=context)
    qa_answer = tokenizer.encode(qa_output['answer'])
    current_output = qa_answer if (qa_output['score'] > FLAGS.qa_conf_thresh) else None
    with torch.no_grad():
        out_ids = sample_seq(personality, freqs, context, history, tokenizer, model, nlp, 
                                current_output = current_output, verbose=verbose)
    history.append(out_ids)
    history = history[-(2*FLAGS.max_history+1):]
    out_text = tokenizer.decode(out_ids, skip_special_tokens=False)
    print(out_text)
    return 1
# function to talk with bot with modified decoding strategy
# text is just a string from user to bot
# history is a tokenized history of the past conversation
def respond(text, history, personality, freqs, context, tokenizer, model, nlp, verbose=True):
    # make sure the text isn't empty
    if not text or text=='exit':
        print('Conversation terminated by user')
        return 0

    history.append(tokenizer.encode(text))
    with torch.no_grad():
        out_ids = sample_seq(personality, freqs, context, history, tokenizer, model, nlp, verbose=verbose)
    history.append(out_ids)
    history = history[-(2*FLAGS.max_history+1):]
    out_text = tokenizer.decode(out_ids, skip_special_tokens=False)
    print(out_text)
    return 1

def run(info, freq, context, tokenizer, model, nlp, verbose=False):
    history = []
    while True and respond_qa_start(input('>>> '), history, info, freq, context, tokenizer, model, nlp, verbose=verbose):
        continue

if __name__=="__main__":
    with torch.no_grad():
        nlp = pipeline('question-answering')
    tokenizer_class, model_class = OpenAIGPTTokenizer, OpenAIGPTLMHeadModel
    print('Getting model...')
    pretrained_model = download_pretrained_model() #downloads the pretrained model from S3
    model = model_class.from_pretrained(pretrained_model)
    tokenizer = tokenizer_class.from_pretrained(pretrained_model)
    add_special_tokens_(model, tokenizer)

    screen_name = FLAGS.username
    freq_diffs = freq_diffs(screen_name, 'persona')
    
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

    run(obama_info, freq_diffs, obama_context, tokenizer, model, nlp, verbose=True)
    