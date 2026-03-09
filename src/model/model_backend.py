# ======== tony Note ========
# this is the backend file. 
# all of the calculations and 
# heavy logic work goes in here

import pandas as pd
import numpy as np
from math import log2
from collections import Counter


class Backend:

    # =============== phase 1 : data loading and processing
    #1.0: Read CSV file in 
    def load_dataset(self, datasetFile, viewRawDataset):

        print("Reading in File: ", datasetFile)
        
        try:
            Mental_Health_Data = pd.read_csv(datasetFile)
            
            if(viewRawDataset):
                print("raw dataset | peeking data frame head:")
                print(Mental_Health_Data.head())

        except Exception as e:
            print("Ah Hamburgers:", e)
            raise ValueError("Try Again Friend")

        return Mental_Health_Data

    #1.1: Normalization   
    def normalize_data(self, Mental_Health_Data): 

        Normalize_cols = Mental_Health_Data.select_dtypes(include=['int64', 'float64']).columns

        #print("Integer Columns:")
        #print(numeric_cols)

        #loop through my Columns
        for col in Normalize_cols:
            # equation is X - min / max - min
            # Step 1 Find min and max
            min_val = Mental_Health_Data[col].min()
            max_val = Mental_Health_Data[col].max()

            # run equation: Current - min / max - min
            if max_val != min_val:
                Mental_Health_Data[col] = (Mental_Health_Data[col] - min_val) / (max_val - min_val)

        return Mental_Health_Data    

    #1.2: Convert to Binary 
    def binary_encoding(self, Mental_Health_Data):

        #        -- Activity Type
        Mental_Health_Data["Activity_Type"] = Mental_Health_Data["Activity_Type"].map({
            "Active": 1,
            "Passive": 0
        })

        #-- Gender
        Mental_Health_Data["Gender"] = Mental_Health_Data["Gender"].map({
            "Male": 1,
            "Female": 0
        })
        
        #--User_Archetype
        Mental_Health_Data["User_Archetype"] = Mental_Health_Data["User_Archetype"].map({
            "Digital Minimalist": 0,
            "Passive Scroller": 0,
            "Hyper-Connected": 1,
            "Average User": 1
        })
        
        #---- Age Normalize it?
        """Mental_Health_Data["User_Archetype"] = Mental_Health_Data["User_Archetype"].map({
            "18": 0,
            "19": 0,
            "20": 1,
            "21": 1,
            "22": 1
        })"""
        
      #-- Primary_Platform
        Mental_Health_Data["Primary_Platform"] = Mental_Health_Data["Primary_Platform"].map({
            "Facebook": 0,
            "Twitter/X": 0,
            "LinkedIn": 0,
            "YouTube": 1,
            "Snapchat": 1,
            "TikTok": 1,
            "Instagram": 1
        })
        
        # Content type
        Mental_Health_Data["Dominant_Content_Type"] = Mental_Health_Data["Dominant_Content_Type"].map({
            "Gaming": 0,
            "Lifestyle/Fashion": 0,
            "Entertainment/Comedy": 0,
            "News/Politics": 1,
            "Self-Help/Motivation": 1,
            "Educational/Tech": 1
        })

        #-- GAD_7_Severity
        #0 = No/Low anxiety → Minimal, Mild
        #1 = Clinically significant anxiety → Moderate, Severe
        Mental_Health_Data["GAD_7_Severity"] = Mental_Health_Data["GAD_7_Severity"].map({
            "Mild": 0,
            "Minimal": 0,
            "Moderate": 1,
            "Severe": 1
        })

         #-- PHQ_9 Severity
         #Target Label
        Mental_Health_Data["PHQ_9_Severity"] = Mental_Health_Data["PHQ_9_Severity"].map({
          "Mild":0,
          'None-Minimal':0,
          'Moderate':1,
          'Moderately Severe':1,
          'Severe':1
        })

        Mental_Health_Data = Mental_Health_Data.drop(columns=["User_ID"])
        return Mental_Health_Data

    # ================ phase 2 : data splitting & prep for training
    #2.0: 80/20 Split
    def train_test_split(self, Mental_Health_Data):

        #---- 80/20 Split
        # find my index for 80% of dataset
        split_index = int(0.8 * len(Mental_Health_Data))
        
        #split based on range of 80%
        training_data = Mental_Health_Data.iloc[:split_index]
        testing_data = Mental_Health_Data.iloc[split_index:]

        return training_data, testing_data
    
    #2.1: Generate Train/Test Inputs  
    def features_train_test(self, training_data, testing_data):
       
        features_train = training_data.drop("PHQ_9_Severity", axis=1)
        features_test = testing_data.drop("PHQ_9_Severity", axis=1)

        return features_train, features_test

    #2.2: Generate Train/Test Outputs
    def labels_train_test(self, training_data, testing_data):
        
        labels_train = training_data["PHQ_9_Severity"]
        labels_test = testing_data["PHQ_9_Severity"]

        return labels_train, labels_test

    # ================ phase 3 : model calculations & training
    
    def split(self, feature_column, threshold):

        # left and right sub tree indices 
        left_tree_idxs = np.argwhere(feature_column <= threshold).flatten()
        right_tree_idxs = np.argwhere(feature_column > threshold).flatten()
        
        return left_tree_idxs, right_tree_idxs
    
    def most_common_label(self, labels):
        counter = Counter(labels)
        return counter.most_common(1)[0][0]

    def entropy(self, condition_prob):
       
        if(condition_prob == 0):
            return 0
       
        else:
            entropy_val = -(condition_prob * log2(condition_prob))
            return entropy_val

    def best_split(self, features, labels, total_features):
        
        best_gain = -1 
        split_idx, split_threshold = None, None
        
        # find threshold & index that splits into the best subtree
        for feature_idx in range(total_features):
            
            # isolated feature column
            feature_column = features[:, feature_idx]

            # the unique values for each feature column
            unique_vals = np.unique(feature_column)

            # array of thresholds used to partition 
            # example_uniq_vals           = [1 , 2, 3, 4]
            # example_partition_threshold = [1.5, 2.5, 3.5] -->|(1+2)/2 = 1.5 |(2+3)/2 = 2.5 
            partition_thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2

            # loops through threshold array in order to partition and find info gain 
            for threshold in partition_thresholds:
                gain = self.information_gain(labels, feature_column, threshold)

                # compare information gain with current best information gain
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feature_idx
                    split_threshold = threshold

        return split_idx, split_threshold

    def information_gain(self, labels, features, threshold):
        
        # left and right subtree index split | left tree = condition 1 & right tree = condition 2
        left_tree_idxs, right_tree_idxs = self.split(features, threshold)

        # no existing left or right sub tree
        if len(left_tree_idxs) == 0 or len(right_tree_idxs) == 0:
            return 0
       
        # length of :: label column; left subtree; and right subtree
        label_column_length = len(labels)
        left_subtree_length, right_subtree_length = len(left_tree_idxs), len(right_tree_idxs)

        # entropy for the labels
        label_counts = np.bincount(labels.astype(int))
        label_entropy = self.entropy(label_counts[0] / label_column_length)
        label_entropy += self.entropy(label_counts[1] / label_column_length)

        # frequency of each label occurring with each condition (left tree / right tree)
        left_label_counts = np.bincount(labels[left_tree_idxs].astype(int), minlength=2)
        right_label_counts = np.bincount(labels[right_tree_idxs].astype(int), minlength=2)

        left_entropy = self.entropy(left_label_counts[0] /left_subtree_length)
        left_entropy += self.entropy(left_label_counts[1] /left_subtree_length)

        right_entropy = self.entropy(right_label_counts[0] /right_subtree_length)
        right_entropy += self.entropy(right_label_counts[1] /right_subtree_length)

        # weighted entropy = 
        # (condition_1 prob)(entropy_1) + (condition_2 prob)(entropy_2) + ... +(condition_n prob)(entropy_n)
        prob_1 = left_subtree_length / label_column_length
        prob_2 = right_subtree_length / label_column_length
        
        weighted_entropy = (prob_1 * left_entropy) + (prob_2 * right_entropy)

        information_gain = label_entropy - weighted_entropy
        return information_gain
    
    # ================ phase 4 : K fold validation 
    # 4.0: Define Folder Function
    def generate_k_folds(self, data, k):
    
        ##Gent Length of size of each fold
        f_size = len(data) // k
        fold_container = []

        # loop for each section
        for i in range(k):
            # We want to start where the last chunk left off
            # fold pos * chunk size
            start = i * f_size
            # end is start + size
            end = start + f_size

            if i == k - 1:  # last fold gets remainder
                fold = data.iloc[start:]
            else:
                fold = data.iloc[start:end]
            # pop on to container
            fold_container.append(fold)

        return fold_container

    # ================ phase 5 : model evaluations 