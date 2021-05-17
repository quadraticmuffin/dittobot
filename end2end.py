from torch import no_grad
from transformers import OpenAIGPTTokenizer, OpenAIGPTLMHeadModel, pipeline
from huggingface.train import add_special_tokens_
from huggingface.utils import download_pretrained_model
from twitter_proc import freq_diffs, get_twitter_screen_name
from flags import FLAGS
from wiki_proc import info_and_context
from chatbot_qa import run

def end2end():
    with no_grad(): # voodoo to prevent automatic differentiation/ weird errors?
        nlp = pipeline("question-answering")

    tokenizer_class, model_class = OpenAIGPTTokenizer, OpenAIGPTLMHeadModel 
    print('Getting model...')
    pretrained_model = download_pretrained_model() #downloads the pretrained model from S3
    model = model_class.from_pretrained(pretrained_model)
    tokenizer = tokenizer_class.from_pretrained(pretrained_model)
    add_special_tokens_(model, tokenizer)
    
    screen_name = get_twitter_screen_name(FLAGS.name)
    freq_diffs = freq_diffs(screen_name, 'persona', tokenizer)

    wiki = info_and_context(FLAGS.name)
    info, qa_context = wiki.info, wiki.context
    personality = tokenizer.batch_encode_plus([f'My name is {FLAGS.name}.'] + info)['input_ids']

    run(personality, freq_diffs, qa_context, tokenizer, model, nlp, verbose=True)

if __name__ == '__main__':
    end2end()