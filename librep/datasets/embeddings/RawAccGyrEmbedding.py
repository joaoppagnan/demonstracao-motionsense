# mobile_sensors_dataset_base.py
import numpy as np
from ..embeddings.basic_emb_dataset_types import *

##########################################################################################
# RawAccGyrEmbedding: concatenates the x, y, and y samples from the accelerometer
##########################################################################################
class RawAccGyrEmbedding_DataSubSet(Embedding_DataSubSet): 
    def __init__(self, dss: Canonical_DataSubSet):
        """Initialize the data subset converting from the canonical data subset"""
        self.samples = [self.convert_sample(i) for i in dss.samples]
        self.labels = dss.labels
        self.embedding_name = "RawAccGyrEmbedding"
        
    def convert_sample(self,s: Canonical_Sample):
        """Concatenates the x, y, and y accelerometer samples"""
        return np.concatenate((s.acc.x.ravel(),s.acc.y.ravel(),s.acc.z.ravel(),
                               s.gyr.x.ravel(),s.gyr.y.ravel(),s.gyr.z.ravel()))

class RawAccGyrEmbedding_DataSet(Embedding_DataSet): 
    def __init__(self):
        """Initialize the dataset converting from the canonical dataset"""
        self.embedding_name = "RawAccGyrEmbedding"
        self.train      = None
        self.validation = None
        self.test       = None

    def import_from_canonical(self, ds: Canonical_DataSet):
        self.train      = RawAccGyrEmbedding_DataSubSet(ds.train)
        self.validation = RawAccGyrEmbedding_DataSubSet(ds.validation)
        self.test       = RawAccGyrEmbedding_DataSubSet(ds.test)
