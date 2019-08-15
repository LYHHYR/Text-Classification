import numpy as np
import matplotlib.pyplot as plt

def softmax_func(x):
    '''
    x: n_samples * n_classes
    caculate
    '''
    orig_shape = x.shape
    if len(x.shape) > 1:
        # Matrix
        row_max = np.max(x, axis=1).reshape(x.shape[0], 1)
        x -= row_max
        exp_x = np.exp(x)
        row_exp_sum = np.sum(exp_x, axis=1).reshape(exp_x.shape[0], 1)
        x = exp_x / row_exp_sum
    else:
        # Vector
        max = np.max(x)
        x -= max
        exp_x = np.exp(x)
        x = exp_x / np.sum(exp_x)
    assert x.shape == orig_shape
    return x

######################################
#      Testing functions below       #
######################################
def test_softmax():
    np.random.seed(20190606)
    softmax_inputs = np.random.rand(100, 10)
    softmax_outputs = softmax_func(softmax_inputs)
    print("\n === TEST RESULT === ")
    print("\n the result of softmax function is")
    print(softmax_outputs)
    return softmax_inputs, softmax_outputs

def picture_softmax():
    softmax_inputs, softmax_outputs = test_softmax()
    print("Softmax Function Input :: {}".format(softmax_inputs))
    print("Softmax Function Output :: {}".format(softmax_outputs))
    plt.plot(softmax_inputs, softmax_outputs)
    plt.xlabel("Softmax Inputs")
    plt.ylabel("Softmax Outputs")
    plt.show()

if __name__ == "__main__":
    test_softmax()
    picture_softmax()