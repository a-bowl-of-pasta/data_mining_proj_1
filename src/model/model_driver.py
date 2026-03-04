# ======= Tony Note ======
#
# This file is the main API
# all the logic and math
# will be done in the other files. 
# this one just pieces it together

from .model_backend import Backend 

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
    def learn_model(self):

        train_dataset, test_dataset = self.backend.train_test_split(self.dataset)

        #---- Split output Field out of data
        self.train_input,    self.test_input = self.backend.input_train_test(train_dataset, test_dataset)
        self.train_output,   self.test_output = self.backend.output_train_test(train_dataset,test_dataset)

        print("model ready to test")


    # in progress ::
    def run_k_fold_validation(self, KFolds):
        
        # 4: K-Fold split   
        input_fold = self.backend.generate_k_folds(self.train_input, KFolds)
        output_fold = self.backend.generate_k_folds(self.train_output, KFolds)

        for i in range(KFolds):
            
            val_input_data = input_fold[i]
            val_output_data = output_fold[i]
            
            training_input_data = pd.concat(input_fold[:i] + input_fold[i + 1:])
            training_output_data = pd.concat(output_fold[:i] + output_fold[i + 1:])


            print(f"\nFold {i + 1}")
            print("Training size:", len(training_input_data))
            print("Validation size:", len(val_input_data))
    
        
# =================================== delete all of this after meeting

    def get_neighbors(self):
        return self.current_state.neighbors
    
    def test_model_api(self):
        return "model api works"

    def test_backend_api(self):
        return self.backend.test_backend_api()



