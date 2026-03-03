# ========= Tony Notes =======
# this file is simply for keeping track 
# of important model elements. 
# it is for anything that is repeatedly
# used by the model <such as a dataset vector> 
# 
# for example : 
# dataset vectors
# model configuration values
# etc

class ModelState:

    
    def __init__(self, Kneighbors):
        self.neighbors = Kneighbors

    def get_neighbors(self): 
        return(self.neighbors)

    def TEMP_MODEL_STATE_TEST(self):
        return "Model State works"


