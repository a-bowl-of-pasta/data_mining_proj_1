from model.decision_tree_driver import Decision_Tree_Model
from model.svm import run_svm_experiments
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
    
    # PHQ_9_Severity = depression, GAD_7_Severity = anxiety
    label_options = ['depression_severity','anxiety_severity']

    tree_max_depth = 3
    Kfolds = [5, 10, 15, 20] 
    run_kfold = True

    # ==== build model
    decision_tree = Decision_Tree_Model(max_depth=tree_max_depth , min_samples_split=2)

    decision_tree.build_model(datasetFile, peekRawDataset=False)
    input("press <enter> to continue program")
    os.system('cls')
     
   
    if run_kfold == True:

        # === kfold validation
        decision_tree.k_fold_validation(Kfolds)
        input("press <enter> to continue program")
        os.system('cls')

    else:
        # === learn model
        decision_tree.learn_model(labelChoice = label_options[0])
        input("press <enter> to continue program")
        os.system('cls') 
        
        # === view processed data from the learn stage
        decision_tree.peek_processed_data()
        input("press <enter> to continue program")
        os.system('cls')  
        
        # === evaluate model 
        decision_tree.model_eval()
        input("press <enter> to continue program")
        os.system('cls')  

        decision_tree.evaluation_summary()
    
    print()
    print("---------------SVM---------------")
    #run_svm_experiments(datasetFile)


