import torch
import torch.nn.functional as F
import time
import pyro
from src.bel2scm.neurips_bel2scm.constants import VARIABLE_TYPE
import numpy as np


class SigmoidNet(torch.nn.Module):
    """
       This class is used to train model for SCM.
       """

    def __init__(self, n_feature, n_output, max_abundance):
        super(SigmoidNet, self).__init__()
        self.max_abundance = max_abundance
        self.predict = torch.nn.Linear(n_feature, n_output)  # hidden layer

    def forward(self, x):
        x = self.predict(x)
        return x


class TrainNet():
    """
	This class initiates SigmoidNet, sets hyperparameters,
	and performs training.
	"""
    # All hardcoded hyperparameter resides here.
    learning_rate = 0.005
    n_hidden = 10
    train_loss = 0
    test_loss = 0
    test_residual_std = 0
    train_test_split_index = 2000
    n_epochs = 1000
    batch_size = 500

    def __init__(self, n_feature, n_output, max_abundance, isRegression):
        self.isRegression = isRegression
        self.max_abundance = max_abundance

        self.net = SigmoidNet(n_feature, n_output, max_abundance)
        self.loss_func = torch.nn.SmoothL1Loss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)

    def fit(self, x, y):
        target_transformed_to_log = self.transform_target_to_log(y, max_abundance=self.max_abundance)

        # parent_transformed_to_log = self.transform_parent_to_log(x)

        train_x, train_y, test_x, test_y = self._get_train_test_data(x,
                                                                     target_transformed_to_log)
        train_x_untransformed, train_y_untransformed, test_x_untransformed, test_y_untransformed = \
            self._get_train_test_data(x, y)

        for epoch in range(self.n_epochs):

            # X is a torch Variable
            permutation = torch.randperm(train_x.size()[0])

            for i in range(0, train_x.size()[0], self.batch_size):
                self.optimizer.zero_grad()
                # get batch_x, batch_y
                indices = permutation[i:i + self.batch_size]
                batch_x, batch_y = train_x[indices], train_y[indices]
                prediction = self.net(batch_x)  # test_x = x[self.train_test_split_index:].flatten().view(-1, 1)
                loss = self.loss_func(prediction, batch_y)
                loss.backward()
                self.optimizer.step()

        # calculate train and test loss

        self.train_loss = self.loss_func(self.sigmoid(train_x_untransformed), train_y_untransformed)
        self.test_loss = self.loss_func(self.sigmoid(test_x_untransformed), test_y_untransformed)
        # weights = self.net.predict.weight
        residuals = self._residual(y, self.sigmoid(x))

        self.residual_mean = residuals.mean()
        self.residual_std = residuals.std()

    def binary_predict(self, x, noise):
        prediction = self.net.logit_forward(x)
        return F.sigmoid(prediction + noise)

    def continuous_predict(self, x, noise):
        x = x.numpy()
        noise = noise.detach().numpy()
        prediction = self.sigmoid(x) + noise
        return prediction

    # Hill equation
    def sigmoid(self, x):
        betas = self.net.predict.parameters()
        betas = [w.detach().numpy() for w in betas]
        weights = np.array([w[0] for w in betas[:-1]])
        c = betas[-1][0]
        power_part = np.dot(x, weights.T) + c
        output = self.max_abundance / (1 + np.exp(-power_part))
        output = torch.tensor(output.reshape(-1, 1))
        return output

    def _get_train_test_data(self, x, y):
        # train data
        m = self.train_test_split_index
        n = x.size()[1]
        p = x.size()[0] - self.train_test_split_index
        train_x = x[:self.train_test_split_index].reshape(m, n)
        train_y = y[:self.train_test_split_index].reshape(m, 1)

        # test data
        test_x = x[self.train_test_split_index:].reshape(p, n)
        test_y = y[self.train_test_split_index:].reshape(p, 1)

        return train_x, train_y, test_x, test_y

    def _residual(self, observed_value, predicted_value):
        return (observed_value - predicted_value)

    def transform_target_to_log(self, target, max_abundance):
        try:
            output = np.log(target / (max_abundance - target))
        except:
            raise Exception("error caused by", target)
        return np.log(target / (max_abundance - target))

    def transform_parent_to_log(self, parent):
        return np.log(parent)


class ParameterEstimation:
    """
	This class requires non-empty graph with available node_data.
	SCM class uses get_model_for_each_node function for each node after loading bel graph and data.
    """

    def __init__(self, belgraph, config):
        # Dictionary<node_str, TrainNet obj>
        self.trained_networks = dict()

        # Dictionary <node_str, pyro.Distribution>
        self.root_parameters = dict()

        # Dictionary <node_str, std of residual>
        self.exogenous_std_dict = dict()

        if not belgraph.nodes or not belgraph.node_data:
            raise Exception("Empty Graph or data not loaded.")

        self.belgraph = belgraph
        self.config = config

    def get_distribution_for_roots_from_data(self):

        for node_str, features_and_target_data in self.belgraph.node_data.items():

            # if node is a root node, and it has empty feature df
            if (self.belgraph.nodes[node_str].root) and (features_and_target_data["features"].empty):

                # we need node label to search for the corresponding distribution from config
                node_label = self.belgraph.nodes[node_str].node_label
                # Getting corresponding distribution from config
                node_distribution = self.config.node_label_distribution_info[node_label]

                # if the node label belongs to categorical
                if node_label in VARIABLE_TYPE["Categorical"]:
                    self.root_parameters[node_str] = self._get_distribution_for_binary_root(
                        node_distribution,
                        features_and_target_data)

                # Otherwise we assume that the data is continuous and return a distribution with mean and std from data.
                else:
                    self.root_parameters[node_str] = self._get_distribution_for_continuous_root(
                        node_distribution,
                        features_and_target_data)
                self.exogenous_std_dict[node_str] = torch.tensor(features_and_target_data["target"].std())

    def get_model_for_each_non_root_node(self):
        """
		This function iterates through every non-root node and trains a neural network.
		It pushes TrainNet objects to trained_network dictionary.
		"""
        for node_str, features_and_target_data in self.belgraph.node_data.items():

            # if node is not a root node, and it has continuous parents
            if (not self.belgraph.nodes[node_str].root) and (not features_and_target_data["features"].empty):

                # we need node label to search for the corresponding distribution from config
                node_label = self.belgraph.nodes[node_str].node_label

                # Currently, only binary classification is supported.
                # [TODO] Add support for multi-class classification for categorical variable

                if node_label in VARIABLE_TYPE["Categorical"]:
                    print("Start training of node:", node_str)
                    time1 = time.time()
                    trained_network = self._classification(features_and_target_data)
                    print("Finished training of node:", node_str, "in ", (time.time() - time1), " seconds")
                    print("Test error:", trained_network.test_loss)
                else:
                    print("Start training of node:", node_str)
                    time1 = time.time()
                    trained_network = self._regression(features_and_target_data)
                    print("Finished training of node:", node_str, "in ", (time.time() - time1), " seconds")
                    print("Test error:", trained_network.test_loss)

                self.trained_networks[node_str] = trained_network
                self.exogenous_std_dict[node_str] = torch.tensor(trained_network.test_residual_std)

    def _regression(self, features_and_target_data):
        # convert features dataframe to float tensor.
        feature_data = torch.tensor(features_and_target_data["features"].values).float()
        number_of_features = feature_data.size()[1]

        # convert target series to float tensor.
        target_data = torch.tensor(features_and_target_data["target"].values).float()

        train_network = TrainNet(n_feature=number_of_features,
                                 max_abundance=self.config.continuous_max_abundance,
                                 n_output=1,
                                 isRegression=True)

        train_network.fit(feature_data, target_data)

        return train_network

    def _classification(self, features_and_target_data):
        feature_data = torch.tensor(features_and_target_data["features"].values).float()
        number_of_features = feature_data.size()[1]

        target_data = torch.tensor(features_and_target_data["target"].values).float()

        # new instance of TrainNet with isRegression=false.
        train_network = TrainNet(n_feature=number_of_features,
                                 max_abundance=1.0,
                                 n_output=1,
                                 isRegression=False)
        train_network.fit(feature_data, target_data)

        return train_network

    def _get_distribution_for_binary_root(self, node_distribution, features_and_target_data):

        # convert target series to float tensor.
        mean = torch.tensor(features_and_target_data["target"].mean())

        # mean should be a probability that becomes the parameter for Bernoulli
        if 0 <= mean <= 1:
            try:
                return (mean)
            except:
                raise Exception(
                    "The schema from _get_distribution_for_binary_root does not match for the pyro distribution for node ",
                    features_and_target_data["target"].name)
        else:
            raise Exception("Something wrong with data for ", features_and_target_data["target"].name)

    def _get_distribution_for_continuous_root(self, node_distribution, features_and_target_data):

        # convert target series to float tensor.
        mean = torch.tensor(features_and_target_data["target"].mean())
        std = torch.tensor(features_and_target_data["target"].std())

        try:
            return (mean, std)
        except:
            raise Exception(
                "The schema from _get_distribution_for_continuous_root does not match for the pyro distribution for node ",
                features_and_target_data["target"].name)
