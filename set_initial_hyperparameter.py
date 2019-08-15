import numpy as np

def initial_parameters(features, labels):
    n_sample, n_feature = np.shape(features)
    n_class = np.shape(labels)[0]

    np.random.seed(20190813)
    w = np.random.rand(n_feature, n_class)
    b = np.zeros((n_sample, n_class))

    rlambda = 0.02
    learning_rate = 0.005
    n_epoch = 20
    batch_size = 100
    
    return w, b, learning_rate, rlambda, n_epoch, batch_size

######################################
#      Testing functions below       #
######################################
def test_initial_parameters():
    test_features = np.array([[1, 0, 2, 1], [2, 0, 1, 1], [3, 0, 0, 1]])
    test_labels = np.array([2, 3, 1])
    print(initial_parameters(test_features, test_labels))

if __name__ == "__main__":
    test_initial_parameters()



