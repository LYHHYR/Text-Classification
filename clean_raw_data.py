from nltk import word_tokenize, corpus
from re import sub
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from pandas import DataFrame

# 0. transfer data form series to list
def transfer_form(raw_data, info):
    '''
    :param raw_data: raw data including phrases and sentiments
            info: 'Phrase' or 'Sentiment'
    :return: all phrases
    '''
    info_from_data = raw_data[info]
    info_from_data = list(info_from_data)
#    print ("step 0 successed")
    return info_from_data

# 1. tokenize phrases
def tokenize(phrases):
    '''
    :param phrases: phrases, several words are included in each phrase
    :return: a set of words in each phrase
    '''
    words_set = []
    # go through every row in series (phrases is a series)
    for phrase in phrases:
        words_set.append(word_tokenize(phrase))
#    print ("step 1 successed")
    return words_set

# 2. normalize words
# 2.1 remove punctuation
def remove_punctuation(words_set):
    new_words_set = []
    for words in words_set:
        new_words = []
        for word in words:
            new_word = sub(r'[^\w\s]', '', word)
            if new_word != '':
                new_words.append(new_word)
        new_words_set.append(new_words)
    return new_words_set

# 2.2 convert all characters to lowercase
def to_lowercase(words_set):
    new_words_set = []
    for words in words_set:
        new_words = []
        for word in words:
            new_word = word.lower()
            new_words.append(new_word)
        new_words_set.append(new_words)
    return new_words_set

# 2.3 remove numbers
def remove_numbers(words_set):
    new_words_set = []
    for words in words_set:
        new_words = []
        for word in words:
            new_word = sub("\d+", "", word)
            new_words.append(new_word)
        new_words_set.append(new_words)
    return new_words_set

# 2.4 remove stop words ("almost meaningless words")
def remove_stopwords(words_set):
    new_words_set = []
    for words in words_set:
        new_words = []
        for word in words:
            if word not in corpus.stopwords.words('english'):
                new_words.append(word)
        new_words_set.append(new_words)
    return new_words_set

# 2.5 stem words with stemmer
def stem_words(words_set):
    stemmer = PorterStemmer()
    new_words_set = []
    for words in words_set:
        new_words = []
        for word in words:
            new_words.append(stemmer.stem(word))
        new_words_set.append(new_words)
    return new_words_set

# 2.6 lemmatize, e.g. women -> woman
def lemmatize_words(words_set):
    lemmatizer = WordNetLemmatizer()
    new_words_set = []
    for words in words_set:
        new_words = []
        for word in words:
            new_words.append(lemmatizer.lemmatize(word))
        new_words_set.append(new_words)
    return new_words_set

# normalize the words
def normalize(words_set):
    '''
    :param words_set:
    :return: words sets in which each word has been selected and transformed
    '''
    new_words_set = remove_punctuation(words_set)
    new_words_set = to_lowercase(words_set)
    new_words_set = remove_numbers(words_set)
    new_words_set = remove_stopwords(words_set)
    new_words_set = stem_words(words_set)
    new_words_set = lemmatize_words(words_set)
#   print ("step 2 successed")
    return new_words_set

# cleaning as a whole
def cleaner(raw_data):
    '''
    :param raw_data: from load_raw_data
    :return: new_words_set (phrases), sentiments
    '''
    phrases = transfer_form(raw_data, 'Phrase')
    words_set = tokenize(phrases)
    new_words_set = normalize(words_set)

    if 'Sentiment' in raw_data:
        sentiments = transfer_form(raw_data, 'Sentiment')
        return new_words_set, sentiments
    elif 'PhraseId'in raw_data:
        return new_words_set, transfer_form(raw_data, 'PhraseId')
    else:
        return new_words_set

######################################
#      Testing functions below       #
######################################
def test_cleaner():
    test_raw_data = DataFrame({"Phrase":["Hello nature language processing!",
                     "Today is 4 Jun. 2019",
                     "is 4 Jun weird sentence I know",
                     "yesterday is 3 Jun. 2019",
                     "weird sentence life is much",
                     "Tomorrow is 5 Jun. 2019",
                     "weird sentence I know on purpose",
                     "So nice to know you",
                     "weird sentence So nice",
                     "I suffered a lot before",
                     "weird sentence Don't forget",
                     "But now, life is much easier",
                     "Don't forget to always be brave"],
                    "Sentiment":[5, 4, 5, 4, 5, 3, 2, 4, 3, 5, 2, 3, 4]})
    test_new_words_set, sentiments = cleaner(test_raw_data)
    print(test_new_words_set, sentiments)

if __name__ == "__main__":
    test_cleaner()