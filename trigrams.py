import numpy

# 1. creat a bag of trigrams
def bag_of_trigrams(phrases):
    dictionary = set()
    for words in phrases:
        p1 = '*'
        p2 = '*'
        for word in words:
            p3 = word
            trigram = p1 + ' ' + p2 + ' ' + p3
            dictionary.add(trigram)
            p1 = p2
            p2 = p3
    bag_size = len(dictionary)
    trigram_bag = {trigram: key for key, trigram in enumerate(dictionary)}
    return bag_size, trigram_bag

# 2. transfer trigrams to vectors
def to_vectors(phrases):
    bag_size, trigram_bag = bag_of_trigrams(phrases)
    feature_set = []
    for words in phrases:
        sentence_set = []
        p1 = '*'
        p2 = '*'
        for word in words:
            p3 = word
            trigram = p1 + ' ' + p2 + ' ' + p3
            x = numpy.zeros(bag_size)
            trigram_position = trigram_bag[trigram]
            x[trigram_position] = 1
            sentence_set.append(x)
            p1 = p2
            p2 = p3
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

