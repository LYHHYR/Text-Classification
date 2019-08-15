import numpy as np

def one_hot(X, y):
    """
    While y belongs to {1, 2, ..., n} (n>2), tranform y to one-hot encoded matrix
    """
    n_samples, n_features = X.shape
    y_classes = set(y)
    n_classes = len(y_classes)
    one_hot = np.zeros((n_samples, n_classes))
    # y should be transfer into a series which begin with 0 in order to get the position information
    y_position = y - np.min(y)
    one_hot[np.arange(n_samples), y_position.T] = 1
    # np.arange(n_samples): generate an array([0, 1, 2, 3,..., (n_sample - 1)])
    return one_hot

######################################
#      Testing functions below       #
######################################
def test_one_hot():
    np.random.seed(20190606)
    X = np.random.rand(100, 10)
    y = np.random.randint(1, 5, 100)
    # the range of y is {1,2,3,4}
    y_one_hot = one_hot(X, y)
    print ("\n === TEST RESULT === ")
    print("\n the result of one hot function is")
    print (y, y_one_hot)

if __name__ == "__main__":
    test_one_hot()