# ======= Tony Note ======
#
# This file is the main API
# all the logic and math
# will be done in the other files. 
# this one just pieces it together

from .core.model_backend import Backend 
from .core.model_state import ModelState 
from .model_classes import DataGroups, DataPoint

# can be changed to whatever model we are actually using
class Knn_model:
    
    def __init__(self, K_init): 
        self.current_state = ModelState(K_init)  
        self.backend = Backend()

    def load_dataset(self, filePath, normalize):
        self.backend.load_dataset()

    def get_neighbors(self):
        return self.current_state.neighbors

    def TEST_KNN_METHODS(self):
        return "knn works well"

    def TEST_BACKEND_API(self):
        return self.backend.TEST_BACKEND()

    def TEST_MODEL_STATE(self):
        return self.current_state.TEMP_MODEL_STATE_TEST()

    def TEST_DATA_POINT(self):
        return DataPoint().TEMP_DATA_POINT_TEST()

    def TEST_DATA_GROUP(self):
        return DataGroups().TEMP_DATA_GROUP_TEST()
    



