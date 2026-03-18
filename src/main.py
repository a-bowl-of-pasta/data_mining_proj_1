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

def alt_dataset(train_file, test_file, truth_labels, tree_depth, min_samples):
    # ==== build model
    decision_tree = Decision_Tree_Model(tree_depth , min_samples)

    decision_tree.load_train_set(train_file)
    decision_tree.load_test_set(test_file)
    decision_tree.feature_label_split(truth_labels)

    input("press <enter> to continue program")
    os.system('cls')
     
    # === learn model
    decision_tree.learn_model()
    input("press <enter> to continue program")
    os.system('cls') 
         
    # === evaluate model 
    decision_tree.model_eval()
    input("press <enter> to continue program")
    os.system('cls')  

    decision_tree.evaluation_summary()


def main_dataset(full_dataset, peek_data, truth_labels, tree_depth, min_samples, kfold, run_kfold ):
    # ==== build model
    decision_tree = Decision_Tree_Model(tree_depth , min_samples)

    decision_tree.build_model(full_dataset, peek_data)

    input("press <enter> to continue program")
    os.system('cls')
     
    if run_kfold == True:

        # === kfold validation
        decision_tree.k_fold_validation(truth_labels, kfold)
        input("press <enter> to continue program")
        os.system('cls')

    else:
        # === learn model
        decision_tree.process_data(truth_labels)
        decision_tree.learn_model()
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



if __name__ == "__main__": 

    print('Jonathan McClain, Oko Kenechukwu, Anthony Hernandez')
    print('DataMining: Project')
    print()
    
    # ===== files 
    dataset_file = "dataset/Social_Media_Mental_Health.csv"
    label_options = ['depression_severity','anxiety_severity']

    train_file = "dataset/new_train_set.csv"
    test_file = "dataset/new_test_set.csv"
    alt_file_truth_col = "labels"
    # config options 
    tree_max_depth = 3
    min_samples_split = 5
    kfold = 10    
    run_kfold = False
    peek_data = False
    chosen_label = label_options[0]


    # main dataset = social media and mental health | depth = 11, min_samp = {24, 26, or 27} is best
    # alt dataset  = wine dataset

    #main_dataset(dataset_file, peek_data, chosen_label, tree_max_depth, min_samples_split, kfold, run_kfold )
    alt_dataset(train_file, test_file, alt_file_truth_col, tree_max_depth, min_samples_split)


