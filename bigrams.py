import numpy

# 1. creat a bag of bigrams
def bag_of_bigrams(phrases):
    dictionary = set()
    for words in phrases:
        p1 = '*'
        for word in words:
            p2 = word
            bigram = p1 + ' ' + p2
            dictionary.add(bigram)
            p1 = p2
    bag_size = len(dictionary)
    bigram_bag = {bigram: key for key, bigram in enumerate(dictionary)}
    return bag_size, bigram_bag

# 2. transfer bigrams to vectors
def to_vectors(phrases):
    bag_size, bigram_bag = bag_of_bigrams(phrases)
    feature_set = []
    for words in phrases:
        sentence_set = []
        p1 = '*'
        for word in words:
            p2 = word
            bigram = p1 + ' ' + p2
            x = numpy.zeros(bag_size)
            bigram_position = bigram_bag[bigram]
            x[bigram_position] = 1
            sentence_set.append(x)
            p1 = p2
        feature_set.append(sentence_set)
    return feature_set

######################################
#      Testing functions below       #
######################################
def test_to_vectors():
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
    test_to_vectors()

