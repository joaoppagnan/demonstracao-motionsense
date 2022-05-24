from librep.embeddings_eval.embeddings_eval_base import *
import tabulate
import sklearn.metrics 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np

class RF_KNN_SVM_Scores_Result(Evaluation_Result_Text): pass
        
class RF_KNN_SVM_Scores(Embedding_Evaluator_Base_Class):

    evaluator_name = "RF_KNN_SVM_Scores"

    evaluator_description = "RF_KNN_SVM_Scores: Trains RF, KNN, and SVM machine learning models using " + \
                         "the ds.train data subset and report the f1-score and accuracy metrics using " + \
                         "the ds.test data subset. "

    def eval_model(self, model, X, y_test):
        y_pred = model.predict(X)
        f1  = sklearn.metrics.f1_score(y_test, y_pred, average='macro')
        acc = sklearn.metrics.accuracy_score(y_test, y_pred)
        return (float(f1), float(acc))

    models_to_evaluate = [
     { 
         "model" : SVC(C=3.0, kernel="rbf"), 
         "desc"  : "SVC(C=3.0, kernel=\"rbf\")"
     }, 
     {
         "model" : KNeighborsClassifier(n_neighbors=1), 
         "desc"  : "KNeighborsClassifier(n_neighbors=1)"     
     },
     {
         "model" : RandomForestClassifier(n_estimators=100),
         "desc"  : "RandomForestClassifier(n_estimators=100)"     
     }    
    ]
    
    def evaluate_embedding(self, ds : Flattenable_DataSet):
        X_train = np.concatenate((ds.train.get_X(),ds.validation.get_X()))
        y_train = np.concatenate((ds.train.get_y(),ds.validation.get_y()))
        #X_train = ds.train.get_X()
        #y_train = ds.train.get_y()
        X_test  = ds.test.get_X()
        y_test  = ds.test.get_y() 
        #if len(X_train) > 0:
        #    print("RF_KNN_SVM report: X_train[0] shape = ", X_train[0].shape)
        #else:
        #    print("RF_KNN_SVM report: no elements at X_train")
        table = [("Model description", "f1-score", "accuracy")]
        for m in self.models_to_evaluate:
            model = m["model"]
            model.fit(X_train, y_train)
            f1_score, acc = self.eval_model(model,X_test, y_test)
            if "desc" in m: description = m["desc"]
            else:  description = str(model)
            table.append((description,"{0:.2f}%".format(100*f1_score), "{0:.2f}%".format(100*acc)))    
        output_text = tabulate.tabulate(table, headers="firstrow",colalign=("left","right","right"))        
        return RF_KNN_SVM_Scores_Result(output_text) 