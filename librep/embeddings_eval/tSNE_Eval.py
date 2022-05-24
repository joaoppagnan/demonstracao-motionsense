from librep.embeddings_eval.embeddings_eval_base import *
import numpy as np
from sklearn import manifold
import matplotlib.pyplot as plt
import io
from PIL import Image

class tSNE_Result(DataFrame_Result):
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
        ax.set_title("T-SNE", fontsize=14)
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

class tSNE_Eval(Embedding_Evaluator_Base_Class):

    def __init__(self, tsne_model = None):
        if not tsne_model:
            self.tsne_model = manifold.TSNE() # n_components=2, init="pca", perplexity=200.0)
        else:
            self.tsne_model = tsne_model
        
    evaluator_name = "tSNE_Eval"

    evaluator_description = "TODO"

    def evaluate_embedding(self, ds : Flattenable_DataSet):
        X = np.concatenate((ds.train.get_X(),ds.validation.get_X(),ds.test.get_X()))
        y = np.concatenate((ds.train.get_y(),ds.validation.get_y(),ds.test.get_y()))
        tsne_df = self.tsne_model.fit_transform(X)
        tsne_df = pd.DataFrame(tsne_df, columns=["X", "Y"])
        tsne_df["class"] = y
        return tSNE_Result(tsne_df)

    
