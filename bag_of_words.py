import numpy

# 1. creat a bag of words
def bag_of_word(phrases):
    dictionary = set()
    for words in phrases:
        for word in words:
            dictionary.add(word)
    bag_size = len(dictionary)
    word_bag = {word: key for key, word in enumerate(dictionary)}
    return bag_size, word_bag

# 2. transfer words to vectors
def to_vectors(phrases):
    '''
    :param phrases:
    :return: one-hot vector for each unique word
    '''
    bag_size, word_bag = bag_of_word(phrases)
    feature_set = []
    for words in phrases:
        sentence_set = []
        for word in words:
            x = numpy.zeros(bag_size)
            word_position = word_bag[word]
            x[word_position] = 1
            sentence_set.append(x)
        feature_set.append(sentence_set)
    return feature_set

######################################
#      Testing functions below       #
######################################
def test_bag_of_word():
    test_phrases = [['Hello', 'nature', 'language', 'processing', '!'],
                    ['Today', 'is', '4', 'Jun', '.', '2019'],
                    ['is', '4', 'Jun', 'weird', 'sentence', 'I', 'know'],
                    ['yesterday', 'is', '3', 'Jun', '.', '2019'],
                    ['weird', 'sentence', 'life', 'is', 'much'],
                    ['Tomorrow', 'is', '5', 'Jun', '.', '2019'],
                    ['weird', 'sentence', 'I', 'know', 'on', 'purpose'],
                    ['So', 'nice', 'to', 'know', 'you'],
                    ['weird', 'sentence', 'So', 'nice'],
                    ['I', 'suffered', 'a', 'lot', 'before'],
                    ['weird', 'sentence', 'Do', "n't", 'forget'],
                    ['But', 'now', ',', 'life', 'is', 'much', 'easier'],
                    ['Do', "n't", 'forget', 'to', 'always', 'be', 'brave']]

    print(to_vectors(test_phrases))

if __name__ == "__main__":
    test_bag_of_word()