from model.decision_tree_driver import Decision_Tree_Model
import os
from model.SVM import run_svm_experiments

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
    
    tree_max_depth = 3
    Kfolds = 5

    # ==== build model
    decision_tree = Decision_Tree_Model(max_depth=tree_max_depth , min_samples_split=2)

    decision_tree.build_model(datasetFile)
    input("press <enter> to continue program")
    os.system('cls')

    decision_tree.peek_processed_data()
    input("press <enter> to continue program")
    os.system('cls')   
    
    # === learn model
    decision_tree.learn_model()
    input("press <enter> to continue program")
    os.system('cls')  
    
    # === kfold validation
    #decision_tree.run_k_fold_validation(Kfolds)

    # === evaluate model 
    decision_tree.model_eval()
    input("press <enter> to continue program")
    os.system('cls')  

    decision_tree.evaluation_summary()

    print("---------------SVM---------------")
    run_svm_experiments(datasetFile)



'''
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
'''

# DataPoint = each piece of data from the dataset
# DataGroup = the final classifications 

# backend    = where all the math calculations and logic go
# modelState = any model variables and configurations | dataset list, K value, etc

# knn_model  = the main API | all the pieces from the other classes get joined here

