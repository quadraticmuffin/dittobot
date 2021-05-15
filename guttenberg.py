# Imports
import os, json, sys, time
from collections import defaultdict
from types import SimpleNamespace
from tqdm import tqdm
from gutenberg.acquire import load_etext
from gutenberg.cleanup import strip_headers
from gutenberg.query import get_etexts
from gutenberg.query import get_metadata

# Pre-trained Tokenizer
from tokenizers import BertWordPieceTokenizer



TIMELINES_PATH = "timelines"
FREQS_PATH = "word-freqs"
PLAY_PATH = "plays"

"""       
def get_timeline(screen_name, count=3200, save_json=True, sleep_every=-1):
    timeline = []
    statuses = tweepy.Cursor(api.user_timeline, 
        screen_name=screen_name, count=count, tweet_mode='extended').items()
    for i, status in enumerate(statuses):
        timeline.append(status.full_text)
        if sleep_every>0 and (i+1) % sleep_every == 0: # Wait out the API rate limit
            print("Got {i} statuses. Sleeping 15 mins...")
            for _ in tqdm(range(15)):
                time.sleep(60)
    
    if save_json:
        json_path = os.path.join(TIMELINES_PATH, '{screen_name}.json')
        with open(json_path, mode='w', encoding='utf-8') as f:
            json.dump(timeline, f, indent=2, ensure_ascii=False)
    
    return timeline
"""  

def get_play(text_number, title, save_json = True):
    text = strip_headers(load_etext(text_number)).strip()  #Hamlet

    if save_json:
        json_path = os.path.join(PLAY_PATH, title + '.json')
        with open(json_path, mode='w', encoding='utf-8') as f:
            json.dump(text, f, indent=2, ensure_ascii=False)

    return text

def proc_timeline(timeline):
    ret = []
    for twt in timeline:
        # remove retweets
        if twt[:3] == "RT ":
            continue
        # remove @s, links, ellipses for truncated tweets
        twt = ' '.join(word for word in twt.split(' ') if (word and
            word[0] != '@' and word[:4] != 'http' and word[-1] != '…'))
        
        if twt:
            ret.append(twt)
    return ret

def proc_play(text):
    text = text.replace('\n', ' ')
    words = text.split(' ')
    #for word in words: 
     #   if (len(word) != 0 and not(word[0] == '_' and word[-1] == '_')):
      #      next.append(word)
    
    words = [word for word in words if (len(word)!= 0 and not(word[0] == '_' and word[-1] == '_'))]
    ret = []
    i = 0
    while i <len(words):
        if '[' in words[i] and ']' in words[i]:
            index = words[i].find('[')
            ret.append(words[i][:index])
            i+=1
        elif "[" in words[i]:
            for j in range(i, len(words)):
                if ']' in words[j]:
                    i = j+1
                    break
        else:
            ret.append(words[i])
            i+=1
    

    ret = [word for word in ret if len(word) > 0 and not(word.isupper()) and not(word[-1] == '.' and word[:-1].isupper())]

    
    #for word in words:
     #   ' '.join(word for word in text.split(' ') if (word and
      #      not(word[0] == '_' and word[-1] == '_') and not(word[0] == '(' and word[-1] == ')') ))
    # gets rids of line assingments to characters 
    i = len(words) -1
    #print(i)
    """ while i >=0:
        if ']' in words[i]:
            if '[' in words[i]:
                index = words[i].find('[')
                words[i] = words[:index]
                i-=1
            else:
                not_found = True
                for j in range(i-1, -1, -1):
                    if '[' in words[j]:
                        words = words[0:j] + words[i+1:]
                        i = j-1
                        not_found = False
                if not_found:
                    words = words[i+1:]
                    i = -1
                    
        else:
            i-=1
    #print(len(words)) """
    return ret

# gets word frequencies given screen name or Twitter timeline
def word_freq(title, save_json=True):
    play_path = os.path.join(PLAY_PATH, title + '.json')
    if os.path.exists(play_path):
        with open(play_path, mode='r', encoding='utf-8') as f:
            play = json.load(f)

    timeline = proc_play(play)

    freqs = defaultdict(int)
    for tweet in timeline:
        tokens = tokenizer.encode(tweet).tokens
        for token in tokens:
            freqs[token] += 1

    if save_json:
        json_path = os.path.join(FREQS_PATH, title+'.json')
        with open(json_path, mode='w', encoding='utf-8') as f:
            json.dump(freqs, f, indent=2, ensure_ascii=False)

    return freqs

if __name__ == "__main__":
    #screen_name = sys.argv[1]
    tokenizer = BertWordPieceTokenizer("bert-base-uncased-vocab.txt", lowercase=True)
    #text = get_play(1524, "Hamlet")  #27761
    #list = proc_play(text) 

    #print(get_etexts('title', 'Moby Dick; Or, The Whale'))
    #timeline = get_timeline(screen_name, count=10000, save_json=True, sleep_every=500)

    freqs = word_freq("Hamlet")
    # print(sorted(list(freqs.keys()), key=lambda i: freqs[i]))

    