from model.model_driver import Knn_model as model

if __name__ == "__main__": 

    print('Jonathan McClain, Oko Kenechukwu, Anthony Hernandez')
    print('DataMining: Project')
    print()
    
    # ===== config variables
    datasetFile = "dataset/Social_Media_Mental_Health.csv"
    
    Kvalue = 5
    Kfolds = 5

    # ==== build and learn model
    knn = model(Kvalue)

    knn.build_model(datasetFile)
    input("press <enter> to continue program")


    knn.learn_model()
    input("press <enter> to continue program")


    # === k fold validation 
    knn.run_k_fold_validation(Kfolds)
    

# DataPoint = each piece of data from the dataset
# DataGroup = the final classifications 

# backend    = where all the math calculations and logic go
# modelState = any model variables and configurations | dataset list, K value, etc

# knn_model  = the main API | all the pieces from the other classes get joined here

