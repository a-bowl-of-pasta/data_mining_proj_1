from readline import backend

from model import DecisionTree
from model import model_backend
from model import model_driver
from model.model_driver import Knn_model as model
import os

# =================== things that need to get done =================
#  
#  [] finish k-fold validation method
#  [] test k-fold validation method
#  [] write a data loader that assigns ground truths
#  [] write a model evaluation method
#  [] clean up project code & debug



if __name__ == "__main__": 

    print('Jonathan McClain, Oko Kenechukwu, Anthony Hernandez')
    print('DataMining: Project')
    print()
    
    #== instance of model_driver
    backend = model_backend.Backend()
    # ===== config variables
    datasetFile = "dataset/Social_Media_Mental_Health.csv"
    
    Kvalue = 5
    Kfolds = 5

    # ==== build model
    knn = model(Kvalue)

    knn.build_model(datasetFile)
    input("press <enter> to continue program")
    os.system('cls')  
    
    # === splits data & preps for k_fold / testing
    knn.data_split()
    input("press <enter> to continue program")
    os.system('cls')  
    
    # === k fold validation 
    knn.run_k_fold_validation(Kfolds)

    # ===Decision Tree
    training_data, testing_data = backend.train_test_split(datasetFile)

    features_train, features_test = backend.features_train_test(training_data, testing_data)

    labels_train, labels_test = backend.labels_train_test(training_data, testing_data)

    X_train = features_train.to_numpy()
    y_train = labels_train.to_numpy()

    tree = DecisionTree(max_depth=5)
    tree.fit(X_train, y_train)

    X_test = features_test.to_numpy()
    y_test = labels_test.to_numpy()

    predictions = tree.predict(X_test)

    for i in range(len(predictions)):
        print("Prediction:", predictions[i], "| Actual:", y_test[i])

    # === final model evaluation
    #knn.model_evaluation()

    backend.test_labels = labels_test
    backend.predictions = predictions

    backend.model_evaluation()
    

# DataPoint = each piece of data from the dataset
# DataGroup = the final classifications 

# backend    = where all the math calculations and logic go
# modelState = any model variables and configurations | dataset list, K value, etc

# knn_model  = the main API | all the pieces from the other classes get joined here

