from load_raw_data import loader
from clean_raw_data import cleaner
from bag_of_words import to_vectors
# from bigrams import to_vectors
# from trigrams import to vectors
from representing_sentence import in_sentence
import softmax_class as sc
from pandas import DataFrame, merge

# step 1: load raw data
path = '~/Desktop/NLP-Beginner/task1'
raw_train = loader('train.tsv', path)
raw_test = loader('test.tsv', path)
answer = loader('sampleSubmission.csv', path)

# step 2: clean data
train_phrases, train_sentiments = cleaner(raw_train)
test_phrases, test_phraseid = cleaner(raw_test)

# step 3: transfer words/grams to vectors
train_feature_set = to_vectors(train_phrases)
test_feature_set = to_vectors(test_phrases)

# step 4: represent each sentence by summing the vector-form words/grams
train_feature = in_sentence(train_feature_set)
test_feature = in_sentence(test_feature_set)

# step 5: here use softmax function to deal with muti-classify problem
classifier = sc.Classifier(train_feature, train_sentiments)
loss, parameter_w, parameter_b = classifier.train()

# step 6: check the efficiency of our model
test_sentiment = classifier.predict(parameter_w, parameter_b)

my_result = DataFrame({'PhraseId': test_phraseid, 'Peredicted_Sentiment': test_sentiment})
checking = merge(my_result, answer)

print(checking)