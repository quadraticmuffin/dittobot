# Imports
import os, json, sys, time
from collections import defaultdict
import tweepy
import tqdm

# API keys as environment variables
from dotenv import load_dotenv
load_dotenv()

# Pre-trained Tokenizer
from tokenizers import BertWordPieceTokenizer

CONSUMER_KEY = os.getenv('CONSUMER_KEY')
CONSUMER_SECRET = os.getenv('CONSUMER_SECRET')
ACCESS_TOKEN = os.getenv('ACCESS_TOKEN')
ACCESS_TOKEN_SECRET = os.getenv('ACCESS_TOKEN_SECRET')
BEARER_TOKEN = os.getenv('BEARER_TOKEN')

TIMELINES_PATH = "timelines"
FREQS_PATH = "word-freqs"

auth = tweepy.AppAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
# auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
api = tweepy.API(auth)

# tweepy.Client for Twitter API v2 apparently doesn't exist... throws the error:
#   AttributeError: module 'tweepy' has no attribute 'Client'
# client = tweepy.Client(
    #     bearer_token = os.getenv('BEARER_TOKEN'),
    #     consumer_key = os.getenv('CONSUMER_KEY'),
    #     consumer_secret = os.getenv('CONSUMER_SECRET'),
    #     access_token = os.getenv('ACCESS_TOKEN'),
    #     access_token_secret = os.getenv('ACCESS_TOKEN_SECRET'),
    #     wait_on_rate_limit = True
    #     )
        
def get_timeline(screen_name, count=3200, save_json=True, sleep_on_rate_limit=False):
    timeline = []
    statuses = tweepy.Cursor(api.user_timeline, 
        screen_name=screen_name, count=count, tweet_mode='extended').items()
    for i, status in enumerate(statuses):
        timeline.append(status.full_text)
        if sleep_on_rate_limit and i % 1000 == 0: # Wait out the API rate limit
            print(f"Got {i} statuses. Sleeping 15 mins...")
            for _ in tqdm(range(15)):
                time.sleep(60)
    
    if save_json:
        json_path = os.path.join(TIMELINES_PATH, f'{screen_name}2.json')
        with open(json_path, mode='w', encoding='utf-8') as f:
            json.dump(timeline, f, indent=2, ensure_ascii=False)
    
    return timeline

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

# gets word frequencies given screen name or Twitter timeline
def word_freq(screen_name, save_json=True):
    timeline_path = os.path.join(TIMELINES_PATH, f'{screen_name}.json')
    if os.path.exists(timeline_path):
        with open(timeline_path, mode='r', encoding='utf-8') as f:
            timeline = json.load(f)
    else:
        timeline = get_timeline(screen_name, count=3200, save_json=save_json)

    timeline = proc_timeline(timeline)

    freqs = defaultdict(int)
    for tweet in timeline:
        tokens = tokenizer.encode(tweet).tokens
        for token in tokens:
            freqs[token] += 1

    if save_json:
        json_path = os.path.join(FREQS_PATH, f'{screen_name}.json')
        with open(json_path, mode='w', encoding='utf-8') as f:
            json.dump(freqs, f, indent=2, ensure_ascii=False)

    return freqs

if __name__ == "__main__":
    screen_name = sys.argv[1]
    tokenizer = BertWordPieceTokenizer("bert-base-uncased-vocab.txt", lowercase=True)
    timeline = get_timeline(screen_name, count=3200, save_json=False)

    freqs = word_freq(screen_name)
    # print(sorted(list(freqs.keys()), key=lambda i: freqs[i]))

    