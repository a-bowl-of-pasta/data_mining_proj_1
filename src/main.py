from model.model_driver import decision_tree_model 

import os

# =================== things that need to get done =================
#  
#  [] finish k-fold validation method
#  [] test k-fold validation method
#  [] write a model evaluation method
#  [] clean up project code & debug
#  [] SVM



if __name__ == "__main__": 

    print('Jonathan McClain, Oko Kenechukwu, Anthony Hernandez')
    print('DataMining: Project')
    print()
    
    # ===== config variables
    datasetFile = "dataset/Social_Media_Mental_Health.csv"
    
    Kvalue = 5
    Kfolds = 5

    # ==== build model
    decision_tree = decision_tree_model()

    decision_tree.build_model(datasetFile)
    input("press <enter> to continue program")
    os.system('cls')  
    
    # === splits data & preps for k_fold / testing
    decision_tree.data_split()
    input("press <enter> to continue program")
    os.system('cls')  
    
    # === k fold validation 
    decision_tree.run_k_fold_validation(Kfolds)

    # === final model evaluation
    #knn.model_evaluation()
    

# DataPoint = each piece of data from the dataset
# DataGroup = the final classifications 

# backend    = where all the math calculations and logic go
# modelState = any model variables and configurations | dataset list, K value, etc

# knn_model  = the main API | all the pieces from the other classes get joined here

