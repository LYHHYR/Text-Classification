import numpy as np

def in_sentence(feature_set):
    '''
    simply add all the vectors of words in a sentence to represent the sentence
    '''
    features = []
    for feature in feature_set:
        features.append(np.sum(feature, axis=0))
    return features

######################################
#      Testing functions below       #
######################################
def test_in_sentence():
    test_feature_set = [[[0., 0., 0., 0., 1.], [1., 0., 0., 0., 0.], [0., 0., 1., 0., 0.], [0., 0., 0., 1., 0.]],
                        [[0., 0., 0., 1., 0.], [1., 0., 0., 0., 0.]],
                        [[0., 0., 0., 0., 1.], [0., 0., 1., 0., 0.], [0., 0., 1., 0., 0.], [0., 0., 1., 0., 0.], [1., 0., 0., 0., 0.], [0., 0., 0., 0., 1.]]
                        ]
    a_test_feature_set = np.array(test_feature_set)
    print(in_sentence(a_test_feature_set))

if __name__ == "__main__":
    test_in_sentence()