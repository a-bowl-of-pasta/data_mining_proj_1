# ======  Tony Note ======
# this file is where we take care
# of the classes / clusters. 
# 
# essentially, these are the 
# clusters that datapoints get
# assigned to
#
# this is also where


# the data points themselves
class DataPoint:

    def __init__(self, data = None):
        self.data = data
    
    def TEMP_DATA_POINT_TEST(self):
        return "data point works fine"

# the classes <grouped Data Points> | Class1, Class2, Class3
class DataGroups:
    
    def __init__(self):
        self.grouping = []

    def TEMP_DATA_GROUP_TEST(self):
        return "data groups work A okay"
    
