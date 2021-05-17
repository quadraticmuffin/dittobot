from argparse import ArgumentParser
import torch

parser = ArgumentParser()

parser.add_argument("--dataset_path", type=str, default="", help="Path or url of the dataset. If empty download from S3.")
parser.add_argument("--dataset_cache", type=str, default='./dataset_cache', help="Path or url of the dataset cache")
parser.add_argument("--model", type=str, default="openai-gpt", help="Model type (openai-gpt or gpt2)", choices=['openai-gpt', 'gpt2'])  # anything besides gpt2 will load openai-gpt
parser.add_argument("--model_checkpoint", type=str, default="", help="Path, url or short name of the model")
parser.add_argument("--max_history", type=int, default=8, help="Number of previous utterances to keep in history")
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")

parser.add_argument("--no_sample", action='store_true', help="Set to use greedy decoding instead of sampling")
parser.add_argument("--max_length", type=int, default=20, help="Maximum length of the output utterances")
parser.add_argument("--min_length", type=int, default=1, help="Minimum length of the output utterances")
parser.add_argument("--seed", type=int, default=0, help="Seed")
parser.add_argument("--temperature", type=float, default=0.7, help="Sampling softmax temperature")
parser.add_argument("--top_k", type=int, default=0, help="Filter top-k tokens before sampling (<=0: no filtering)")
parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")

parser.add_argument("--bias_cap", type=float, default=2.0, help="Caps the amount of word frequency bias")

parser.add_argument("--use_qa", type=bool, default=True, help="Whether to use QA model to help answer questions")
parser.add_argument("--qa_conf_thresh", type=float, default=0.5, help="Minimum score needed for qa output to be used")

parser.add_argument("--username", type=str, default="BarackObama", help="(case-sensitive) Twitter username of interest.")
parser.add_argument("--name", type=str, default="Barack Obama", help="Name of interest, e.g. `lebron james` or `rihanna`")
FLAGS = parser.parse_args()