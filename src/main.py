from model.our_model import Knn_model as model


if __name__ == "__main__": 
   
    knn = model(10)
    numOfNeighbors = knn.get_neighbors()


    print( knn.TEST_KNN_METHODS())
    print()
    print( knn.TEST_BACKEND_API())
    print( knn.TEST_MODEL_STATE())
    print()
    print( knn.TEST_DATA_GROUP())
    print( knn.TEST_DATA_POINT())
    print()
    print("num of neighbors :: ", numOfNeighbors)
    print()

# DataPoint = each piece of data from the dataset
# DataGroup = the final classifications 

# backend    = where all the math calculations and logic go
# modelState = any model variables and configurations | dataset list, K value, etc

# knn_model  = the main API | all the pieces from the other classes get joined here