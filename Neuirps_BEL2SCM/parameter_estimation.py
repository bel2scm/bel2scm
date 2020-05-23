import pyro
import torch
from torch.autograd import Variable
import torch.nn.functional as F

class RegressionNet(torch.nn.Module):
    """
    This class is used to train regression model for continuous nodes.
    """
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        x = self.predict(x)             # linear output
        return x

class LogisticNet(torch.nn.Module):
    """
    This class is used to train classification model for binary nodes.
    """
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        x = self.predict(F.sigmoid(x))             # linear output
        return x


class TrainNet():
    """
    This class initiates RegressionNet / LogisticNet, sets hyperparameters,
    and performs training.
    """
    # All hardcoded hyperparameter resides here.
    learning_rate = 0.01
    n_hidden = 10
    iterations = 10000
    train_error = 0

    def __init__(self, n_feature, n_output, isRegression):
        if isRegression:
            self.net = RegressionNet(n_feature, self.n_hidden, n_output)
            self.loss_func = torch.nn.MSELoss()
        else:
            self.net = LogisticNet(n_feature, self.n_hidden, n_output)
            self.loss_func = torch.nn.BCELoss()
        self.optimizer =  torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)

    def fit(x, y):
        for t in range(self.iterations):
            prediction = self.net(x)
            loss = self.loss_func(prediction, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


    def predict(x):
        return self.net(x)

class ParameterEstimation:
    """
    This class requires non-empty graph with available node_data.
    SCM class uses get_model_for_each_node function for each node after loading bel graph and data.
    """

    parameters = dict()
    def __init__(self, graph):

        if not graph.nodes or not graph.node_data:
            raise Exception("Empty Graph or data not loaded.")

        self.graph = graph

    def get_model_for_each_node(self):

        for node_str, features_and_target_data in self.graph.node_data:

            # if node is not a root node, and it has continuous parents
            if (not self.graph.nodes[node_str].root) and (features_and_target_data["features"] != None):
                # [TODO] Add train-test split and add test metrics
                if self.graph.nodes[node_str].label == "process":
                    trained_network = self._classification(features_and_target_data)
                else:
                    trained_network = self._regression(features_and_target_data)

                self.parameters[node_str] = trained_netwrok

    def _regression(self, features_and_target_data):
        feature_data = torch.tensor(features_and_target_data["features"])
        number_of_features = len(feature_data)

        target_data = torch.tensor(features_and_target_data["target"])

        train_network = TrainNet(n_feature=number_of_features, n_output=1, isRegression=True)
        train_network.train(feature_data, target_data)

        return train_network

    def _logistic_regression(self):
        feature_data = torch.tensor(features_and_target_data["features"])
        number_of_features = len(feature_data)

        target_data = torch.tensor(features_and_target_data["target"])

        train_network = TrainNet(n_feature=number_of_features, n_output=1, isRegression=False)
        train_network.train(feature_data, target_data)

        return train_network