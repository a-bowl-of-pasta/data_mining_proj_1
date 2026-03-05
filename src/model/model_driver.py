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
    def data_split(self):

        self.train_dataset, self.test_dataset = self.backend.train_test_split(self.dataset)

        #---- Split output Field out of data
        self.train_features, self.test_features = self.backend.features_train_test(self.train_dataset, self.test_dataset)
        self.train_labels,   self.test_labels   = self.backend.labels_train_test(self.train_dataset,self.test_dataset)

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
    def model_evaluation():
        pass

    
    # 6: display final clusters / classifications
    def peek_final_clusters():
        pass
        
# =================================== delete all of this after meeting

    def get_neighbors(self):
        return self.current_state.neighbors
    
    def test_model_api(self):
        return "model api works"

    def test_backend_api(self):
        return self.backend.test_backend_api()



