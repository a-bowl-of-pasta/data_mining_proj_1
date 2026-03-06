# ======= Tony Note ======
#
# This file is the main API
# all the logic and math
# will be done in the other files.
# this one just pieces it together

from .model_backend import Backend
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np


# can be changed to whatever model we are actually using
class Knn_model:

    def __init__(self, k_value):
        self.K_value = k_value
        self.backend = Backend()

    # 1: setup model
    def build_model(self, dataFile):
        print("Currently building model ... ")
        print()

        dataset = self.backend.load_dataset(dataFile)
        dataset = self.backend.binary_encoding(dataset)
        dataset = self.backend.normalize_data(dataset)
        self.dataset = dataset
        print("Normalized Data:")
        print(dataset.head())

        print(" ... model built | ready to learn")

    # 4: Gear Data for ML
    def data_split(self):
        self.train_dataset, self.test_dataset = self.backend.train_test_split(self.dataset)

        # ---- Split output Field out of data
        self.train_features, self.test_features = self.backend.features_train_test(self.train_dataset,
                                                                                   self.test_dataset)
        self.train_labels, self.test_labels = self.backend.labels_train_test(self.train_dataset, self.test_dataset)

        print("model ready to test")

    # in progress ::
    # [] fold training | train the model in the fold
    # [] fold testing  | test the model in the fold
    # [] fold evals    | final evaluations per fold, so we can pick optimal k value
    def run_k_fold_validation(self, KFolds):
        # 4: K-Fold split
        feature_fold = self.backend.generate_k_folds(self.train_features, KFolds)
        label_fold = self.backend.generate_k_folds(self.train_labels, KFolds)

        # runs fold validation K times
        for i in range(KFolds):
            # data held out to test current Kfold run
            val_feature_data = feature_fold[i]
            val_label_data = label_fold[i]

            # data used to train during current Kfold run
            training_feature_data = pd.concat(feature_fold[:i] + feature_fold[i + 1:])
            training_label_data = pd.concat(label_fold[:i] + label_fold[i + 1:])

            print(f"\nFold {i + 1}")
            print("Training size:", len(training_feature_data))
            print("Validation size:", len(val_feature_data))

    # 5: final model evaluation
    def model_evaluation(self):
        y_true = self.test_labels.values
        y_pred = self.predictions

        # True Pos: sum up when true is 1 and predicted is 1
        truepos= np.sum((y_true == 1) & (y_pred == 1))
        # True Neg: sum up when true is 0 and predicted is 0
        trueneg = np.sum((y_true == 0) & (y_pred == 0))
        #-------- This could be Backwards
        # False Pos: sum up when true is 0 and predicted is 1
        false_pos = np.sum((y_true == 0) & (y_pred == 1))
        # True Neg: sum up when true is 1 and predicted is 0
        false_neg = np.sum((y_true == 1) & (y_pred == 0))

        accuracy = (truepos+ trueneg) / len(y_true)
        precision = truepos / (truepos + false_pos) 
        sensitivity = truepos / (truepos + false_neg) 
        specificity = trueneg / (trueneg + false_pos) 

        print("Model Evaluation")
        print("True Positive:", truepos)
        print("False Positive:", false_pos)
        print("False Negative:", false_neg)
        print("True Negative:", trueneg)

        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Sensitivity:", sensitivity)
        print("Specificity:", specificity)



    # 6: display final clusters / classifications
    def peek_final_clusters():
        pass

    # 7 SVM
    def train_svm(self, epochs=100):

        print("Starting Training of the SVM Model...")
        # At this point self has Pandas Object, convert dataframe to numpy array
        x_input = torch.tensor(self.train_features.values, dtype=torch.float32)
        y_output = torch.tensor(self.train_labels.values, dtype=torch.float32)

        # SVM requirement convert labels 0 -> -1
        y_output = torch.where(y_output == 0, -1, 1)
        #define model object with SVM contruction of the
        self.svm_model = SVM(x_input.shape[1])

        # Now we start our Stochastic Gradient Descent Engine
        optimizer = optim.SGD(self.svm_model.parameters(), lr=0.01)
        # Start Loop for Datapasses
        for epoch in range(epochs):
            optimizer.zero_grad()

            outputs = self.svm_model(x_input).squeeze()
            # calculate Hinge Weight update
            loss = torch.mean(torch.clamp(1 - y_output * outputs, min=0))
            # Do backwards propagation to calculate Gradient
            loss.backward()
            # Step based on Gradient cal
            optimizer.step()

        print("SVM Training Complete")

    def test_svm(self):

        print("Testing SVM Model...")
        # Cast Datafram to nummpy array
        x_input = torch.tensor(self.test_features.values, dtype=torch.float32)
        
        outputs = self.svm_model(x_input).detach().numpy()

        self.predictions = (outputs >= 0).astype(int).flatten()

        print("Testing Complete")
    # =================================== delete all of this after meeting

    def get_neighbors(self):
        return self.current_state.neighbors

    def test_model_api(self):
        return "model api works"

    def test_backend_api(self):
        return self.backend.test_backend_api()



class SVM(nn.Module):

    def __init__(self, input_dim):
        super(SVM, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)
