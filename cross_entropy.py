import numpy as np
from math import log

def loss_function(real_y, predict_y):
    """
    caculate the difference between the real labels and the predicted labels
    :param real_y: n_sample * n_class
    :param predict_y: n_sample * n_class
    :return: loss
    """
    log_predict_y = []
    for predict_y_sample in predict_y:
        log_predict_y_sample = []
        for predict_y_sample_class in predict_y_sample:
            log_predict_y_sample.append(log(predict_y_sample_class))
        log_predict_y.append(log_predict_y_sample)
    loss_for_each_sample = - np.dot(real_y, np.mat(log_predict_y).T)
    n_sample = real_y.shape[0]
    loss = (1/n_sample) * np.sum(loss_for_each_sample)
    return loss

######################################
#      Testing functions below       #
######################################
def test_loss_function():
    test_real_y = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]])
    test_predict_y = np.array([[0.2, 0.5, 0.1, 0.1], [0.6, 0.1, 0.2, 0.1], [0.2, 0.2, 0.2, 0.4], [0.8, 0.05, 0.05, 0.05]])
    print(loss_function(test_real_y, test_predict_y))

if __name__ == "__main__":
    test_loss_function()
