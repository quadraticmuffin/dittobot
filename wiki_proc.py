import wikipedia
import re

class FromWiki:
    def __init__(self, info, context):
        self.info = info
        self.context = context

def split_into_sentences(text):
    '''Copied from https://stackoverflow.com/questions/4576077/how-can-i-split-a-text-into-sentences
    with minor adjustment for non-punctuated (chat-like) input'''
    alphabets= "([A-Za-z])"
    prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
    suffixes = "(Inc|Ltd|Jr|Sr|Co)"
    starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
    acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
    websites = "[.](com|net|org|io|gov)"
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = [s.strip() for s in sentences]
    if sentences[-1] == '':
        sentences = sentences[:-1]
    return sentences

def third_to_first(text):
    conversions = {
        'he': 'i', 'she': 'i', 'they': 'i',
        'his': 'my', 'her': 'my', 'their': 'my', 
    }
    return ' '.join(conversions[word] if word in conversions else word for word in text.split() )

def remove_blurb(text):
    '''Removes the blurb containing, e.g. pronunciations and birthdates,
    typically contained in parentheses in Wikipedia articles'''
    open_paren = text.index('(')
    pardepth = 0
    for i, char in enumerate(text[open_paren:]):
        if char == '(':
            pardepth += 1
        if char == ')':
            pardepth -= 1
        if pardepth == 0:
            close_paren = i + open_paren
            break
    return text[:open_paren-1] + text[close_paren+1:]

def info_and_context(name):
    summary = wikipedia.summary(name, auto_suggest=False)
    summary = third_to_first(summary)
    summary = remove_blurb(summary)
    info = split_into_sentences(summary)[:5]
    print(f'Wikipedia page begins:\n{" ".join(info[:2])}')
    context = summary
    return FromWiki(info, context)
