import pandas as pd
import numpy as np

# Define Folder Function
def generate_k_folds(data, k):
    #Gent Length of size of each fold
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




#1: Python Project-----------------------------------------------
print('Jonathan McClain')
print('DataMining: Project')
print()
#2: Read CSV file in --------------------------------------------
print("Read in CSV")
print("Reading in File: social_media_mental_health.csv")
try:
    Mental_Health_Data = pd.read_csv("social_media_mental_health.csv")
    print("DataFrame top 15 Records:")
    print(Mental_Health_Data.head())
except Exception as e:
    print("Ah Hamburgers:", e)
    raise ValueError("Try Again Friend")


#3: Convert to Binary --------------------------------------------

#-- Activity Type
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
    "Passive scroller": 0,
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
    "Youtube": 1,
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

#3: Normalization   --------------------------------------------
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
    Mental_Health_Data[col] = (Mental_Health_Data[col] - min_val) / (max_val - min_val)


print("Normalized Data:")
print(Mental_Health_Data.head())

#4: Gear Data for ML   --------------------------------------------

#---- 80/20 Split
# find my index for 80% of dataset
split_index = int(0.8 * len(Mental_Health_Data))
#split based on range of 80%
training_data = Mental_Health_Data.iloc[:split_index]
testing_data = Mental_Health_Data.iloc[split_index:]

#---- Split output Field out of data
# Generate Inputs and outputs
input_train = training_data.drop("PHQ_9_Severity", axis=1)
output_train = training_data["PHQ_9_Severity"]

input_test = testing_data.drop("PHQ_9_Severity", axis=1)
output_test = testing_data["PHQ_9_Severity"]

#4: K-Fold split   --------------------------------------------

k = 5
input_fold = generate_k_folds(input_train, k)
output_fold = generate_k_folds(output_train, k)

for i in range(k):
    val_input_data = input_fold[i]
    val_output_data = output_fold[i]
    training_input_data = pd.concat(input_fold[:i] + input_fold[i + 1:])
    training_output_data = pd.concat(output_fold[:i] + output_fold[i + 1:])


    print(f"\nFold {i + 1}")
    print("Training size:", len(training_input_data))
    print("Validation size:", len(val_input_data))

