# Require %pip install tabulate
# Require %pip install umap-learn

from librep.embeddings_eval.embeddings_eval_base import *
import numpy as np
import umap
import matplotlib.pyplot as plt
import io
from PIL import Image

class UMAP_Result(DataFrame_Result):
    def fig2img(self,fig):
        """Convert a Matplotlib figure to a PIL Image and return it"""
        buf = io.BytesIO()
        fig.savefig(buf)
        buf.seek(0)
        img = Image.open(buf)
        return img

    def show(self, ignore_labels : [] = None, label_to_str = None):
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(1,1,1)
        ax.set_xlabel('X', fontsize=15)
        ax.set_ylabel('Y', fontsize=15)
        ax.set_title("UMAP", fontsize=14)
        if not ignore_labels:
            df = self.df
        else:
            df = self.df[~self.df["class"].isin(ignore_labels)]
        labels = []
        for c, ds in df.groupby(["class"]):
            ax.scatter(ds["X"], ds["Y"])
            if label_to_str:
                labels.append(label_to_str(c))
            else:
                labels.append(f"{c}")
        ax.legend(labels)
        ax.grid()
        plt.show()   

class UMAP_Eval(Embedding_Evaluator_Base_Class):

    def __init__(self, umap_model = None):
        if not umap_model:
            self.umap_model = umap.UMAP()
        else:
            self.umap_model = umap_model
        
    evaluator_name = "UMAP_Eval"

    evaluator_description = "TODO"

    def evaluate_embedding(self, ds : Flattenable_DataSet):
        X = np.concatenate((ds.train.get_X(),ds.validation.get_X(),ds.test.get_X()))
        y = np.concatenate((ds.train.get_y(),ds.validation.get_y(),ds.test.get_y()))
        umap_df = self.umap_model.fit_transform(X)
        umap_df = pd.DataFrame(umap_df, columns=["X", "Y"])
        umap_df["class"] = y
        return UMAP_Result(umap_df)
