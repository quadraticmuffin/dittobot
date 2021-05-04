# Imports
import os, json, sys
from collections import defaultdict
import tweepy

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
        
def get_timeline(screen_name, count=3200, save_json=True):
    timeline = []
    statuses = tweepy.Cursor(api.user_timeline, screen_name=screen_name, count=count).items()
    for status in statuses:
        timeline.append(status.text)
    
    if save_json:
        json_path = os.path.join(TIMELINES_PATH, f'{screen_name}.json')
        with open(json_path, mode='w', encoding='utf-8') as f:
            json.dump(timeline, f, indent=2, ensure_ascii=False)
    
    return timeline

# gets word frequencies given screen name or Twitter timeline
def word_freq(screen_name, save_json=True):
    timeline_path = os.path.join(TIMELINES_PATH, f'{screen_name}.json')
    if os.path.exists(timeline_path):
        with open(timeline_path, mode='r', encoding='utf-8') as f:
            timeline = json.load(f)
    else:
        timeline = get_timeline(screen_name, count=3200)

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
    freqs = word_freq(screen_name)
    print(sorted(list(freqs.keys()), key=lambda i: freqs[i]))

    