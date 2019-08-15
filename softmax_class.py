import numpy as np
from set_initial_hyperparameter import initial_parameters
from softmax import softmax_func
from cross_entropy import loss_function
from one_hot_y import one_hot
from random import shuffle

class Classifier:
    def __init__(self, features, labels):
        self.feature = features
        self.label = labels
        self.parameter_w, self.parameter_b, self.learning_rate, self.rlambda, self.n_epoch, self.batch_size = initial_parameters(features, labels)
        self.n_sample, self.n_feature = features.shape
        self.n_class = labels.shape[0]
        self.n_iter = self.n_sample//self.batch_size

    def train(self):
        data = np.column_stack((self.feature, self.label))
        loss_record = []
        parameter_w = self.parameter_w
        parameter_b = self.parameter_b
        for i in range(self.n_epoch):
            shuffle(data)
            x = data[:, 0:len(data[0])-2]
            y = data[:, len(data[0])-1]
            for k in range(self.n_iter):
                x_batch = x[:, k*self.batch_size:(k+1)*self.batch_size]
                y_batch = y[:, k*self.batch_size:(k+1)*self.batch_size]
                loss, parameter_w, parameter_b = self.one_process(x_batch, y_batch, parameter_w, parameter_b)
                loss_record.append(loss)
        return loss_record, parameter_w, parameter_b

    def one_process(self, x, y, parameter_w, parameter_b):
        # transfer label to one_hot matrix form
        real_y = one_hot(x, y)
        # compute predicted label
        score = np.dot(x, parameter_w) + parameter_b
        possibility = softmax_func(score)
        # calculate loss
        loss = loss_function(real_y, possibility)
        # compute gradient
        d_w = (1/self.batch_size) * np.dot(x.T, (possibility - real_y))
        d_b = (1/self.batch_size) * np.sum(possibility - real_y, axis=0)
        # update parameters
        parameter_w -= self.learning_rate * d_w
        parameter_b -= self.learning_rate * d_b
        return loss, parameter_w, parameter_b

    def predict(self, test_x, parameter_w, parameter_b):
        score = np.dot(test_x, parameter_w) + parameter_b
        possibility = softmax_func(score)
        sentiment = np.argmax(possibility, axis=1)
        return sentiment

######################################
#      Testing functions below       #
######################################
def test_classifier():
    test_features = np.array([[1, 0, 2, 1], [2, 0, 1, 1], [3, 0, 0, 1]])
    test_labels = np.array([2, 3, 1])
    classifier = Classifier(test_features, test_labels)
    loss_record, parameter_w, parameter_b = classifier.train()
    print(loss_record, parameter_w, parameter_b)
    predict_label = classifier.predict(test_features, parameter_w, parameter_b)
    print(predict_label, test_labels - predict_label)

if __name__ == "__main__":
    test_classifier()
