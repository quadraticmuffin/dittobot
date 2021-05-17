# dittobot
Adds individual personality to HuggingFace chatbot using celebrity Tweets and Wikipedia data.

## End-to-end Chatbot
This takes any sufficiently well-known name (e.g. "obama" or "Justin Bieber") and creates a chatbot based on the Wikipedia summary and recent Tweets of that person.
Note: For best results, 'sufficiently well-known' means that the person of interest will be the first result when searching their name in Twitter or Wikipedia.

### Usage
Clone this repo, then run the following from the cloned directory:
```
pip install -r requirements.txt
python end2end.py --name="name"
```
where "name" should be in quotes if there are spaces in the name.

To see some of the inner workings of the chatbot, add the option ```--verbose={1-3}```. Default is 1.

## Models Used
We used HuggingFace's pretrained OpenAIGPT models for the chatbot, and their question-answering pipeline pre-trained on the SQuAD dataset.
The chatbot randomly samples its response, token by token. The purpose of the question-answering model is to extract answers from Wikipedia, in an effort to make the chatbot more consistent in its answers to questions which are biographical in nature. For example, when asked "What is your name?", a chatbot trained to talk like President Barack Obama shouldn't answer "Betty" (which actually happened to us!)
