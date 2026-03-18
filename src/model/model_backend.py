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
    # Read CSV file in 
    def load_dataset(self, datasetFile, viewRawDataset):

        print("Reading in File: ", datasetFile)

        # --- tries to read file
        try:
            Mental_Health_Data = pd.read_csv(datasetFile)

            # peek raw data, dependent on flag
            if (viewRawDataset):
                print("raw dataset | peeking data frame head:")
                print(Mental_Health_Data.head())

        # --- exception for if file can't be read
        except Exception as e:
            print("Ah Hamburgers:", e)
            raise ValueError("Try Again Friend")

        return Mental_Health_Data

    # Normalization   
    def normalize_data(self, Mental_Health_Data):

        ''' 
        X = features
        min = min value in that specific feature column
        max = max value in that specific feature column

        normalized X = (X - min)/(max - min)
        '''
        Normalize_cols = Mental_Health_Data.select_dtypes(include=['int64', 'float64']).columns

        # loop through my Columns
        for col in Normalize_cols:
            # Step 1 Find min and max
            min_val = Mental_Health_Data[col].min()
            max_val = Mental_Health_Data[col].max()

            # run equation: Current - min / max - min
            if max_val != min_val:
                Mental_Health_Data[col] = (Mental_Health_Data[col] - min_val) / (max_val - min_val)

        return Mental_Health_Data

        # Convert to Binary

    def binary_encoding(self, Mental_Health_Data):

        #        -- Activity Type
        Mental_Health_Data["Activity_Type"] = Mental_Health_Data["Activity_Type"].map({
            "Active": 1,
            "Passive": 0
        })

        # -- Gender
        Mental_Health_Data["Gender"] = Mental_Health_Data["Gender"].map({
            "Male": 1,
            "Female": 0
        })

        # --User_Archetype
        Mental_Health_Data["User_Archetype"] = Mental_Health_Data["User_Archetype"].map({
            "Digital Minimalist": 0,
            "Passive Scroller": 0,
            "Hyper-Connected": 1,
            "Average User": 1
        })

        # ---- Age Normalize it?
        """Mental_Health_Data["User_Archetype"] = Mental_Health_Data["User_Archetype"].map({
            "18": 0,
            "19": 0,
            "20": 1,
            "21": 1,
            "22": 1
        })"""

        # -- Primary_Platform
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

        # -- GAD_7_Severity
        # 0 = No/Low anxiety → Minimal, Mild
        # 1 = Clinically significant anxiety → Moderate, Severe
        Mental_Health_Data["GAD_7_Severity"] = Mental_Health_Data["GAD_7_Severity"].map({
            "Mild": 0,
            "Minimal": 0,
            "Moderate": 1,
            "Severe": 1
        })

        # -- PHQ_9 Severity
        # Target Label
        Mental_Health_Data["PHQ_9_Severity"] = Mental_Health_Data["PHQ_9_Severity"].map({
            "Mild": 1,
            'None-Minimal': 0,
            'Moderate': 1,
            'Moderately Severe': 1,
            'Severe': 1
        })

        Mental_Health_Data = Mental_Health_Data.drop(columns=["User_ID"])
        return Mental_Health_Data

    # ================ phase 2 : data splitting & prep for training
    # 80/20 Split
    def train_test_split(self, Mental_Health_Data):

        # ---- 80/20 Split
        # find my index for 80% of dataset
        split_index = int(0.8 * len(Mental_Health_Data))

        # split based on range of 80%
        training_data = Mental_Health_Data.iloc[:split_index]
        testing_data = Mental_Health_Data.iloc[split_index:]

        return training_data, testing_data

    # Generate Train/Test Inputs  
    def features_train_test(self, training_data, testing_data, drop_label='default'):

        # remove severity & score from training | score heavily predicts severity
        if drop_label == 'default':
            drop_cols = ["PHQ_9_Severity", "PHQ_9_Score", "GAD_7_Severity", "GAD_7_Score"]
        elif drop_label == 'alt':
            drop_cols = ['Wine Names']

        features_train = training_data.drop(drop_cols, axis=1)
        features_test = testing_data.drop(drop_cols, axis=1)

        return features_train, features_test

    # Generate Train/Test Outputs
    def truth_train_test(self, training_data, testing_data, label_choice):

        if label_choice == 'anxiety_severity':
            ground_truth_train = training_data["GAD_7_Severity"]
            ground_truth_test = testing_data["GAD_7_Severity"] 

        elif label_choice == 'depression_severity':
            ground_truth_train = training_data["PHQ_9_Severity"]
            ground_truth_test = testing_data["PHQ_9_Severity"]      
              
        else:
            ground_truth_test = testing_data[label_choice]
            ground_truth_train = training_data[label_choice]

        return ground_truth_train, ground_truth_test

    # ================ phase 3 : model calculations & training
    # gets left and right tree indexes | left is condition 1 right is condition 2
    def split(self, feature_column, threshold):
        ''' 
        condition : new_data <= best found column / feature threshold 

        left tree = condition is true 
        right tree = condition is false
        '''
        left_tree_idxs = np.argwhere(feature_column <= threshold).flatten()
        right_tree_idxs = np.argwhere(feature_column > threshold).flatten()

        return left_tree_idxs, right_tree_idxs

    # finds most common label | ground_truths = labels
    def most_common_label(self, ground_truths):

        counter = Counter(ground_truths.flatten())
        return counter.most_common(1)[0][0]

    # calculates entropy for information gain
    def entropy(self, condition_prob):
        if (condition_prob == 0):
            return 0

        else:
            # entropy = -sum( P_i * log_2(P_i) )
            entropy_val = -(condition_prob * log2(condition_prob))
            return entropy_val

    # generates informatin gain 
    def information_gain(self, ground_truth, features, threshold):

        # gather indices of left & right sub trees
        left_tree_idxs, right_tree_idxs = self.split(features, threshold)

        # stop if subtree is empty
        if len(left_tree_idxs) == 0 or len(right_tree_idxs) == 0:
            return 0

        # length of :: truth column; left subtree; and right subtree
        truth_column_length = len(ground_truth)
        left_subtree_length, right_subtree_length = len(left_tree_idxs), len(right_tree_idxs)

        # gather frequencies 
        truth_frequencies = np.bincount(ground_truth.flatten().astype(int), minlength=2)
        left_truth_frequencies = np.bincount(ground_truth[left_tree_idxs].flatten().astype(int), minlength=2)
        right_truth_frequencies = np.bincount(ground_truth[right_tree_idxs].flatten().astype(int), minlength=2)

        # calculate entropy 
        truth_entropy = sum(self.entropy(freq / truth_column_length) for freq in truth_frequencies)
        left_entropy = sum(self.entropy(freq / left_subtree_length) for freq in left_truth_frequencies)
        right_entropy = sum(self.entropy(freq / right_subtree_length) for freq in right_truth_frequencies)

        # weighted entropy = 
        # (condition_1 prob)(entropy_1) +...+ (condition_n prob)(entropy_n)
        prob_1 = left_subtree_length / truth_column_length
        prob_2 = right_subtree_length / truth_column_length

        weighted_entropy = (prob_1 * left_entropy) + (prob_2 * right_entropy)

        # information gain = traget entropy - weighted entropy
        information_gain = truth_entropy - weighted_entropy
        return information_gain

    # uses the information gain to find the best partition  
    def best_split(self, features, ground_truth, total_features):

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
                gain = self.information_gain(ground_truth, feature_column, threshold)

                # compare information gain with current best information gain
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feature_idx
                    split_threshold = threshold

        return split_idx, split_threshold

    # ================ phase 4 : K fold validation 
    # 4.0: Define Folder Function
    def generate_k_folds(self, data, K):

        # fold size = data size / number of folds
        # data = 100 | k = 5 | fold_size = 100 / 5 = 20
        fold_size = len(data) // K
        fold_container = []

        # loop K times & build fold container
        for i in range(K):
            '''
            data partitioning into folds | start & stop indexing
            start = 1 * 20 = 20 | i = 1 ; fold = 20  
            end = 20 + 20 = 40  | fold = indx 20 -> indx 40 
            '''
            start = i * fold_size
            end = start + fold_size

            # last fold remainder | i = k -1
            if i == K - 1:
                fold = data.iloc[start:]
            else:
                fold = data.iloc[start:end]

            # adds each partition (fold) to container
            fold_container.append(fold)

        return fold_container

    # ================ phase 5 : model evaluations
