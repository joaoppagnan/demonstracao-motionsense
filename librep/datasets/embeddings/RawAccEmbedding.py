# mobile_sensors_dataset_base
import numpy as np
from ..embeddings.basic_emb_dataset_types import *

##########################################################################################
# RawAccEmbedding: concatenates the x, y, and y samples from the accelerometer
##########################################################################################
class RawAccEmbedding_DataSubSet(Embedding_DataSubSet): 
    def __init__(self, dss: Canonical_DataSubSet):
        """Initialize the data subset converting from the canonical data subset"""
        self.samples = [self.convert_sample(i) for i in dss.samples]
        self.labels = dss.labels
        self.embedding_name = "RawAccEmbedding"
        
    def convert_sample(self,s: Canonical_Sample):
        """Concatenates the x, y, and y accelerometer samples"""
        return np.concatenate((s.acc.x.ravel(),s.acc.y.ravel(),s.acc.z.ravel()))

class RawAccEmbedding_DataSet(Embedding_DataSet): 
    def __init__(self):
        """Initialize the dataset converting from the canonical dataset"""
        self.embedding_name = "RawAccEmbedding"
        self.train      = None
        self.validation = None
        self.test       = None

    def import_from_canonical(self, ds: Canonical_DataSet):
        self.train      = RawAccEmbedding_DataSubSet(ds.train)
        self.validation = RawAccEmbedding_DataSubSet(ds.validation)
        self.test       = RawAccEmbedding_DataSubSet(ds.test)

