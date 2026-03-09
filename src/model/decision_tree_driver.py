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


class tree_node:

    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature          # index of feature to split on
        self.threshold = threshold      # threshold value
        self.left = left                # left child
        self.right = right              # right child
        self.value = value              # class label if leaf
   
    def is_leaf_node(self):
        return self.value is not None



class Decision_Tree_Model: 
   
    def __init__(self, max_depth=10, min_samples_split=2):
        self.backend = Backend()
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    # = = = = = = = = = = = = = = = = = main API methods
    # 1 & 2: builds model & sets up data for training
    def build_model(self, dataFile, peekRawDataset=False):
        print("Currently building model ... ")
        print()

        print("loading dataset...")
        dataset = self.backend.load_dataset(dataFile, peekRawDataset)

        print("normalizing dataset...")
        dataset = self.backend.normalize_data(dataset)
        
        print("encoding dataset...")
        dataset = self.backend.binary_encoding(dataset)

        self.dataset = dataset
        print()
        print("splitting data...")

        self._data_split()
        print("data split & (feature / ground truth) sets created...")
        print()
        print(" ... model built | ready to learn")

    # 3: learns model | training phase
    def learn_model(self):  
        print("learning model | building tree...")

        # convert pandas dataframe into numpy array
        # array    <---    dataframe 
        feature_np_array = self.train_features.values
        label_np_array = self.train_labels.values

        self.root = self._build_tree(feature_np_array, label_np_array, depth=0)
        
        print("decision tree built...")

    # 4: performs k fold validation | either run this or training, not both
    def k_fold_validation(self, k : int):
        pass

    # 5: final evaluation 
    def model_eval(self):
        print("testing and evaluating model...")
        # generate predictions by traversing tree for each test row
        feature_array = self.test_features.values
        ground_truth = self.test_labels.values

        predictions = np.array([self._traverse_tree(x, self.root) for x in feature_array])

        predictions = predictions.flatten()
        ground_truth = ground_truth.flatten()

        print(predictions[:10])
        print(ground_truth[:10])

        trueneg, truepos, false_neg, false_pos = self._confusion_matrix(predictions, ground_truth)

        self.accuracy = (truepos+ trueneg) / len(ground_truth)
        self.precision = truepos / (truepos + false_pos) 
        self.sensitivity = truepos / (truepos + false_neg) 
        self.specificity = trueneg / (trueneg + false_pos) 

        print("confusion matrix breakdown :: ")
        print(f"True Positive: {truepos}\nFalse Positive: {false_pos}")
        print(f"True Negative: {trueneg}\nFalse Negative: {false_neg}")

        print("... evaluation metrics ready | run <model.evaluation_summary()>")
                
    # = = = = = = = = = = = = = = = = = API helper methods 
    
    
    # ----- phase 1 & 2 helpers : data loading, processing, and splits / learn prep
    # splits the data | first into train & test | second into features & labels
    def _data_split(self):
        ''' 
                            /-- 80% train split --- training set feature & label separation - used in training
        processed dataset--|
                            \-- 20% test split  --- test set feature & label separation - used in eval
         '''
        print("splitting | 80% train : 20% test")
        self.train_dataset, self.test_dataset = self.backend.train_test_split(self.dataset)

        # ---- Split output Field out of data
        print("generating :: feature and ground truth sets for train and test")
        self.train_features, self.test_features = self.backend.features_train_test(self.train_dataset,self.test_dataset)
        self.train_labels, self.test_labels = self.backend.labels_train_test(self.train_dataset, self.test_dataset)
        

    # ----- phase 3 helpers : model learning
    # sets up the tree structure 
    def _build_tree(self, features, labels, depth):

        # useful numbers 
        total_rows, total_features = features.shape
        total_uniq_labels = len(np.unique(labels))

        # stop if max depth reached | only 1 class exists | or rows can't be split anymore
        if ( depth >= self.max_depth or total_uniq_labels == 1 or total_rows < self.min_samples_split):
            
            leaf_value = self.backend.most_common_label(labels)
            
            return tree_node(value=leaf_value)

        best_feature, best_threshold = self.backend.best_split(features, labels, total_features)
        
        # stop if a best feature cannot be found
        if best_feature is None:
           
            leaf_value = self.backend.most_common_label(labels)
           
            return tree_node(value=leaf_value)

        # indices for the left and right sub trees
        # left tree = condition 1 | right tree = condition 2
        left_idxs, right_idxs = self.backend.split(features[:, best_feature], best_threshold)

        left_subtree = self._build_tree(features[left_idxs, :], labels[left_idxs], depth + 1)
        right_subtree = self._build_tree(features[right_idxs, :], labels[right_idxs], depth + 1)

        # nodes for tree construction 
        return tree_node(
            feature=best_feature,
            threshold=best_threshold,
            left=left_subtree,
            right=right_subtree
        )


    # ----- phase 4 helpers : k fold validation 



    # ----- phase 5 helpers : model evaluations   
    # tree traversal, used when predicting 
    def _traverse_tree(self, x, node):
            if node.is_leaf_node():
                return node.value

            if x[node.feature] <= node.threshold:
                return self._traverse_tree(x, node.left)
            return self._traverse_tree(x, node.right)

    # confusion matrix for | matrix of TP, FP, TN, FN  
    def _confusion_matrix(self, y_pred, y_true):
        # True Pos: sum up when true is 1 and predicted is 1
        truepos= np.sum((y_true == 1) & (y_pred == 1))
       
        # True Neg: sum up when true is 0 and predicted is 0
        trueneg = np.sum((y_true == 0) & (y_pred == 0))
        #-------- This could be Backwards
        
        # False Pos: sum up when true is 0 and predicted is 1
        false_pos = np.sum((y_true == 0) & (y_pred == 1))
        
        # True Neg: sum up when true is 1 and predicted is 0
        false_neg = np.sum((y_true == 1) & (y_pred == 0))

        return trueneg, truepos, false_neg, false_pos

    # = = = = = = = = = = = = = = = = basic output methods 
    def peek_processed_data(self):
        print("peeking processed data | data frame head")
        
        print("train dataset")
        print(self.train_dataset.head())
        
        print("test dataset")
        print(self.test_dataset.head())

    def evaluation_summary(self):

        print(f"Accuracy:    {self.accuracy}")
        print(f"Precision:   {self.precision}")
        print(f"Sensitivity: {self.sensitivity}")
        print(f"Specificity: {self.specificity}")

    



'''
class svm_model():
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
        """
        # Cast Datafram to nummpy array
        x_input = torch.tensor(self.test_features.values, dtype=torch.float32)
        
        outputs = self.svm_model(x_input).detach().numpy()

        self.predictions = (outputs >= 0).astype(int).flatten()
        """
        print("Testing Complete")


class SVM(nn.Module):

    def __init__(self, input_dim):
        super(SVM, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)
'''
