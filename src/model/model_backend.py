# ======== tony Note ========
# this is the backend file. 
# all of the calculations and 
# heavy logic work goes in here

import pandas as pd
import numpy as np

class Backend: 

    def get_jaccard_dist(self):
        pass


    # Define Folder Function
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


    #1: Read CSV file in 
    def load_dataset(self, datasetFile):

        print("Read in CSV")
        print("Reading in File: ", datasetFile)
        
        try:
            Mental_Health_Data = pd.read_csv(datasetFile)
            print("DataFrame top 15 Records:")
            print(Mental_Health_Data.head())

        except Exception as e:
            print("Ah Hamburgers:", e)
            raise ValueError("Try Again Friend")

        return Mental_Health_Data


    #2: Convert to Binary 
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
          "Mild":'M',
          'None-Minimal':'NM',
          'Moderate':'MO',
          'Moderately Severe':'MS',
          'Severe':'S'
        })


        return Mental_Health_Data


    #2: Normalization   
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
    

    #4: 80/20 Split
    def train_test_split(self, Mental_Health_Data):

        #---- 80/20 Split
        # find my index for 80% of dataset
        split_index = int(0.8 * len(Mental_Health_Data))
        
        #split based on range of 80%
        training_data = Mental_Health_Data.iloc[:split_index]
        testing_data = Mental_Health_Data.iloc[split_index:]

        return training_data, testing_data
    

    #4: Generate Train/Test Inputs  
    def features_train_test(self, training_data, testing_data):
       
        features_train = training_data.drop("PHQ_9_Severity", axis=1)
        features_test = testing_data.drop("PHQ_9_Severity", axis=1)

        return features_train, features_test


    #4: Generate Train/Test Outputs
    def labels_train_test(self, training_data, testing_data):
        
        labels_train = training_data["PHQ_9_Severity"]
        labels_test = testing_data["PHQ_9_Severity"]

        return labels_train, labels_test

    # entropy : -sum(P_i * log_2(P_i))
    def entropy_calc(self, dataset):
        pass
    
    def test_backend_api(self):
        return "backend works fine"