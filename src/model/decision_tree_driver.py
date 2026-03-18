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

    def load_train_set(self, trainSet):
        self.train_dataset = self.backend.load_dataset(trainSet, False)
        

    def load_test_set(self, testSet):
        self.test_dataset = self.backend.load_dataset(testSet, False)


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
        print(" ... model built | ready to learn")

    # 2: data processing
    def feature_label_split(self, label_choice):
        self.train_features, self.test_features = self.backend.features_train_test(self.train_dataset,self.test_dataset, 'alt')
        self.train_ground_truths, self.test_ground_truths = self.backend.truth_train_test(self.train_dataset, self.test_dataset, label_choice)


    def process_data(self, label_choice):
        print()
        print("splitting data...")

        self._data_split(label_choice)

        print("data split & (feature / ground truth) sets created...")
        print()
        print(f"chosen class for model prediction :: {label_choice}...")
        print()

    # 3: learns model | training phase
    def learn_model(self ):  
        print("learning model | building tree...")

        # convert pandas dataframe into numpy array
        # array    <---    dataframe 
        feature_np_array = self.train_features.values
        truth_np_array = self.train_ground_truths.values

        self.root = self._build_tree(feature_np_array, truth_np_array, depth=0)
        
        print("decision tree built...")


    # 4: performs k fold validation | either run this or training, not both
    def k_fold_validation(self, labelChoice,  k : int):
        
        # 4: train / test split && fold creation 
        train_dataset, test_dataset = self.backend.train_test_split(self.dataset)
        folded_train_set = self.backend.generate_k_folds(train_dataset, k)

        eval_output_truths = ['Accuracy', 'precision', 'sensitivity', 'specificity']
        ave_eval_scores = [0,0,0,0]

        # runs cross validation K times
        for i in range(k):

            # test = fold at index i | train = all folds excluding the one at index i                        
            k_test = folded_train_set[i]
            k_train = pd.concat([folded_train_set[j] for j in range(k) if j != i])
            
            # == train phase
            k_train_feature, k_test_feature = self.backend.features_train_test(k_train, k_test)
            k_train_truth, k_test_truth = self.backend.truth_train_test(k_train, k_test, labelChoice)

            k_feature_np_array = k_train_feature.values
            k_truth_np_array = k_train_truth.values

            k_root = self._build_tree(k_feature_np_array, k_truth_np_array, depth=0)

            # == test & eval phase
            k_feature_array = k_test_feature.values
            k_ground_truth = k_test_truth.values

            k_predictions = np.array([self._traverse_tree(x, k_root) for x in k_feature_array])

            k_predictions = k_predictions.flatten()
            k_ground_truth = k_ground_truth.flatten()

            tn, tp, fn, fp = self._confusion_matrix(k_predictions, k_ground_truth)

            ave_eval_scores[0] += (tp + tn) / len(k_ground_truth)
            ave_eval_scores[1] += tp / (tp + fp) 
            ave_eval_scores[2] += tp / (tp + fn) 
            ave_eval_scores[3] += tn / (tn + fp)

            print(f"finished k-fold itearation {i + 1} of {k} iterations...")
        
        ave_eval_scores = [scores / k for scores in ave_eval_scores]
        
        print()
        print(f"K fold cross validation eval summary for:")
        print(f"k = {k} | max depth = {self.max_depth} | min samp = {self.min_samples_split} ")
        for i in range(len(ave_eval_scores)): 
            print(f"average {eval_output_truths[i]}:\t{ave_eval_scores[i]}")
        print()


    # 5: final evaluation 
    def model_eval(self):
        print("testing and evaluating model...")

        # --- model prediction / tree traversal  
        feature_array = self.test_features.values
        ground_truth = self.test_ground_truths.values

        predictions = np.array([self._traverse_tree(x, self.root) for x in feature_array])

        predictions = predictions.flatten()
        ground_truth = ground_truth.flatten()

        print()
        print("sample prediction & ground truth outputs")
        print("predictions:  ",predictions[:20])
        print("Ground Truth: ",ground_truth[:20])

        # --- 
        trueneg, truepos, false_neg, false_pos = self._confusion_matrix(predictions, ground_truth)
        self._evaluation_metric_calculations(truepos, trueneg, false_pos, false_neg, ground_truth)

        print()
        print("confusion matrix breakdown :: ")
        print(f"True Positive: {truepos}\nFalse Positive: {false_pos}")
        print(f"True Negative: {trueneg}\nFalse Negative: {false_neg}")
        print()
        print("... evaluation metrics ready | run <model.evaluation_summary()>")
                
    # = = = = = = = = = = = = = = = = = API helper methods 
    
    # ============ splits the data | first into train & test | second into features & gorund truths
    def _data_split(self, label_choice):
        ''' 
                            /-- 80% train split --- training set feature & ground truth separation - used in training
        processed dataset--|
                            \-- 20% test split  --- test set feature & ground truth separation - used in eval
         '''
        print("splitting | 80% train : 20% test")
       
        self.train_dataset, self.test_dataset = self.backend.train_test_split(self.dataset)

        print("generating :: feature and ground truth sets for train and test")
        
        self.train_features, self.test_features = self.backend.features_train_test(self.train_dataset,self.test_dataset)
        self.train_ground_truths, self.test_ground_truths = self.backend.truth_train_test(self.train_dataset, self.test_dataset, label_choice)
        
    # ============ sets up the tree structure 
    def _build_tree(self, features, ground_truths, depth):

        # useful numbers 
        total_rows, total_features = features.shape
        total_uniq_labels = len(np.unique(ground_truths))

        # stop if max depth reached | only 1 class exists | or rows can't be split anymore
        if ( depth >= self.max_depth or total_uniq_labels == 1 or total_rows < self.min_samples_split):
            
            leaf_value = self.backend.most_common_label(ground_truths)
            
            return tree_node(value=leaf_value)

        best_feature, best_threshold = self.backend.best_split(features, ground_truths, total_features)
        
        # stop if a best feature cannot be found
        if best_feature is None:
           
            leaf_value = self.backend.most_common_label(ground_truths)
           
            return tree_node(value=leaf_value)

        # indices for the left and right sub trees
        # left tree = condition 1 | right tree = condition 2
        left_idxs, right_idxs = self.backend.split(features[:, best_feature], best_threshold)

        left_subtree = self._build_tree(features[left_idxs, :], ground_truths[left_idxs], depth + 1)
        right_subtree = self._build_tree(features[right_idxs, :], ground_truths[right_idxs], depth + 1)

        # nodes for tree construction 
        return tree_node(feature=best_feature, threshold=best_threshold,
                            left=left_subtree, right=right_subtree)

    # ============ tree traversal, used when predicting 
    def _traverse_tree(self, x, node):
            if node.is_leaf_node():
                return node.value

            if x[node.feature] <= node.threshold:
                return self._traverse_tree(x, node.left)
            return self._traverse_tree(x, node.right)

    # ============ confusion matrix for | matrix of TP, FP, TN, FN  
    def _confusion_matrix(self, y_pred, y_true):
        '''
        tp : predict class 1 & truth is class 1             | 1 & 1 | + & +
        tn : predict class is not 1 & truth class is not 1  | 2 & 2 | - & -
        fp : predict class 1 & truth class is not 1         | 1 & 2 | + & -
        fn : predict class is not 1 & truth is class 1      | 2 & 1 | - & +
        '''
        truepos= np.sum((y_true == 1) & (y_pred == 1))
        trueneg = np.sum((y_true == 0) & (y_pred == 0))

        false_pos = np.sum((y_true == 0) & (y_pred == 1))
        false_neg = np.sum((y_true == 1) & (y_pred == 0))

        return trueneg, truepos, false_neg, false_pos

    # ============ finds the evaluation scores 
    def _evaluation_metric_calculations(self, tp, tn, fp, fn, ground_truth):
        
        self.accuracy = (tp + tn) / len(ground_truth)
        self.precision = tp / (tp + fp) 
        self.sensitivity = tp / (tp + fn) 
        self.specificity = tn / (tn + fp) 


    # = = = = = = = = = = = = = = = = basic output methods 
    def peek_processed_data(self):
        print()
        print("peeking processed data | data frame head")
        
        print()
        print("train dataset")
        print(self.train_dataset.head())
        
        print()
        print("test dataset")
        print(self.test_dataset.head())

    def evaluation_summary(self):
        print()
        print(f"Accuracy:    {self.accuracy: .5f}")
        print(f"Precision:   {self.precision: .5f}")
        print(f"Sensitivity: {self.sensitivity: .5f}")
        print(f"Specificity: {self.specificity: .5f}")

    