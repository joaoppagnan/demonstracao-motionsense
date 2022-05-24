from librep.datasets.basic_dataset_types import Flattenable_DataSet
import pandas as pd
  
##########################################################################################
# Base Embedding dataset abstract classes
# - All embedding classes must derive from these classes and implement their 
#   respective constructors to convert from canonical data sets.
# - Sample types must derive from Flattenable Samples.
##########################################################################################

class Evaluation_Result:
    def show(self): pass

class Evaluation_Result_Text(Evaluation_Result):   
    def __init__(self, text : str):
        self.text = text
    def show(self): 
        print(self.text)
    def __str__(self): 
        return self.text

class DataFrame_Result(Evaluation_Result):
    def __init__(self, df : pd.DataFrame):
        self.df = df
    def show(self): 
        print(self.df)
    def __str__(self): 
        return self.df
    
class Embedding_Evaluator_Base_Class:

    evaluator_name = "Embedding_Evaluator_Base_Class"

    evaluator_description = "Not defined"

    def evaluate_embedding(self, ds : Flattenable_DataSet) -> Evaluation_Result:
        """The evaluate_embedding method must evaluate the dataset
           ds and return the results on a Evaluation_Result object"""
        raise NameError("Child class must implement evaluate_embedding()")
