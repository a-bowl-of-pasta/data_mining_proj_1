import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

from model.model_backend import Backend

# ---------------------------- SVM Model

class SVMModel(nn.Module):
    # Construct linear SVM model using PyTorch
    # This represents the decision function: f(x) = w·x + b
    def __init__(self, input_size):
        super(SVMModel, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.linear(x)



# ---------------------------- Train SVM

def train_svm(input_training, output_training, input_test, output_test, epochs=2000):

    #------------------ Data conversion
    # Ok we have our Pandas Data structure Convert to PyTouch containers called tensors
    # Inputs
    input_training = torch.tensor(input_training.values, dtype=torch.float32)
    input_test = torch.tensor(input_test.values, dtype=torch.float32)
    # Outputs
    output_training = torch.tensor(output_training.values, dtype=torch.float32).view(-1,1)
    output_test = torch.tensor(output_test.values, dtype=torch.float32).view(-1,1)


    # ------------------ Convert 0 to -1 for SVM

    # Convert labels from {0,1} to {-1,1} since SVM hinge loss requires signed labels
    output_training = torch.where(output_training == 0, -1, 1)
    output_test = torch.where(output_test == 0, -1, 1)
    #print(output_training)
    #print(output_test)
    # ------------------ SVM Contruct
    # define model object with SVM construction of the
    model = SVMModel(input_training.shape[1])

    # ------------------ Gradient Decent
    # Initialize Graident Optimizer engine
    # LR = Learning rate start with .1
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    for epoch in range(epochs):
        perm = torch.randperm(input_training.size(0))
        X = input_training[perm]
        y = output_training[perm]

        outputs = model(X)
        #Regulaization Parameter Lamda?
        C = 10
        # hinge_loss For our Gradient Decent
        hinge_loss = torch.mean(torch.clamp(1 - y * outputs, min=0))
        #Regulaization L2
        loss = 0.5 * torch.sum(model.linear.weight ** 2) + C * hinge_loss
        #clear out gradients for calculation ( have to do each loop )
        optimizer.zero_grad()
        #Calculate the Gradent through back propigation
        # This is the Derviative of the lose with respect to weight
        loss.backward()
        # and Adjust Weights w = w - a * gradient
        optimizer.step()
    # HUGE ISSUE TURN OFF BACK PROP FOR THIS PART
    with torch.no_grad():
        # w * x + b
        predictions = model(input_test)
        # Convert Raw Scores to a set prediction
        # Fancy way to say prediction > 0 then 1 else -1
        predicted = torch.where(predictions >= 0, 1, -1)
    # Print out Unique Values to ensure predicted is outputing both
    # had issue with values going to 1 and was right 63% of the time.
    #print("Predicted classes:", np.unique(predicted.numpy()))
    #print("True classes:", np.unique(output_test.numpy()))
    return predicted.numpy(), output_test.numpy()


# ---------------------------- Evaluation Metrics

def matrix_metrics(pred, true):

    pred = pred.flatten()
    true = true.flatten()
    #true = np.where(true == 0, -1, 1)
    #true = torch.where(true == 0, -1, 1)
    #print(pred[1])
    #print(true[1])

    # True Pos: sum up when true is 1 and predicted is 1
    truepos = np.sum((pred == 1) & (true == 1))
    # True Neg: sum up when true is -1 and predicted is -1
    trueneg = np.sum((pred == -1) & (true == -1))
    # False Pos: sum up when true is -1 and predicted is 1
    falsepos = np.sum((pred == 1) & (true == -1))
    # True Neg: sum up when true is 1 and predicted is -1
    falseneg = np.sum((pred == -1) & (true == 1))

    #Accuracy
    accuracy = (truepos + trueneg) / len(true)
    #precision
    precision = truepos / (truepos + falsepos + 1e-10)
    #Sensitivity
    Sensitivity = truepos / (truepos + falseneg + 1e-10)

    f1 = 2 * (precision * Sensitivity) / (precision + Sensitivity + 1e-10)

    return accuracy, precision, Sensitivity, f1

# ---------------------------- Run One off
# Choo Choo
def run_train(train_data, test_data, backend):
    # Construct ML Data sets
    input_training, input_test = backend.features_train_test(train_data, test_data)
    output_training, output_test = backend.labels_train_test(train_data, test_data)
    # run Raw training
    pred, true = train_svm(input_training, output_training, input_test, output_test)

    return matrix_metrics(pred, true)


# ---------------------------- K-Fold Cross Validation

def run_kfold(dataset, backend, k=5):
    folds = backend.generate_k_folds(dataset, k)

    accuracy = []
    precision = []
    Sensitivity = []
    f1_container = []

    for i in range(k):
        # Ok for this Put test in a side pocket
        test_data = folds[i]
        # take non index folds and put together
        train_folds = [folds[j] for j in range(k) if j != i]
        train_data = pd.concat(train_folds)
        # run training
        acc, prec, rec, f1 = run_train(train_data, test_data, backend)

        accuracy.append(acc)
        precision.append(prec)
        Sensitivity.append(rec)
        f1_container.append(f1)

    return np.mean(accuracy), np.mean(precision), np.mean(Sensitivity), np.mean(f1_container)

#----------------------------- MAIN
def run_svm_experiments(dataset_file):
    # Generate Backend Engine
    backend = Backend()
    # load
    dataset = backend.load_dataset(dataset_file, False)
    # Normalize ints
    dataset = backend.normalize_data(dataset)
    # convert to binary 
    dataset = backend.binary_encoding(dataset)

    # Need Randomizer!
    dataset = dataset.sample(frac=1).reset_index(drop=True)

    print("\n----- SVM Evaluation -----")

    # 80 / 20
    train_data, test_data = backend.train_test_split(dataset)

    acc, prec, rec, f1 = run_train(train_data, test_data, backend)
    print()
    print("80/20 Split")
    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("F1 Score:", f1)

    # 20 / 80
    split_index = int(0.2 * len(dataset))

    train_data = dataset.iloc[:split_index]
    test_data = dataset.iloc[split_index:]

    acc, prec, rec, f1 = run_train(train_data, test_data, backend)
    print()
    print("20/80 Split")
    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("F1 Score:", f1)


    # K-Fold
    acc, prec, rec, f1 = run_kfold(dataset, backend, 5)
    print()
    print("K Fold")
    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("F1 Score:", f1)



