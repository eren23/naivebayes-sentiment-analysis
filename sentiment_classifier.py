# from nltk.corpus import twitter_samples
# from nltk import FreqDist
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
import re, string
# from nltk.corpus import stopwords
# import random
# from nltk import classify
# from nltk import NaiveBayesClassifier
# from nltk.tokenize import word_tokenize
# import pickle




# text = twitter_samples.strings('tweets.20150430-223406.json')
# tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
# positive_tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
# negative_tweet_tokens = twitter_samples.tokenized('negative_tweets.json')

# positive_cleaned_tokens_list = []
# negative_cleaned_tokens_list = []



def remove_noise(tweet_tokens, stop_words = ()):

    cleaned_tokens = []

    for token, tag in pos_tag(tweet_tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
        token = re.sub("(@[A-Za-z0-9_]+)","", token)

        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens


# stop_words = stopwords.words('english')



# for tokens in positive_tweet_tokens:
#     positive_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

# for tokens in negative_tweet_tokens:
#     negative_cleaned_tokens_list.append(remove_noise(tokens, stop_words))



# def get_tweets_for_model(cleaned_tokens_list):
#     for tweet_tokens in cleaned_tokens_list:
#         yield dict([token, True] for token in tweet_tokens)

# positive_tokens_for_model = get_tweets_for_model(positive_cleaned_tokens_list)
# negative_tokens_for_model = get_tweets_for_model(negative_cleaned_tokens_list)

# # print(negative_tokens_for_model)

# positive_dataset = [(tweet_dict, "Positive")
#                      for tweet_dict in positive_tokens_for_model]

# negative_dataset = [(tweet_dict, "Negative")
#                      for tweet_dict in negative_tokens_for_model]

# dataset = positive_dataset + negative_dataset

# random.shuffle(dataset)

# train_data = dataset[:7000]
# test_data = dataset[7000:]


# classifier = NaiveBayesClassifier.train(train_data)



# custom_tweet = "This model barely works"

# custom_tokens = remove_noise(word_tokenize(custom_tweet))


# varssss = classifier.classify(dict([token, True] for token in custom_tokens))

# print(varssss)


# with open('model_pickle','wb') as file:
#     pickle.dump(classifier,file)