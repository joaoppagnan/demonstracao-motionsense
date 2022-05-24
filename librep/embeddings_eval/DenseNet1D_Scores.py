from librep.embeddings_eval.embeddings_eval_base import *
import tabulate
import sklearn.metrics 
from librep.neural_networks import DenseNet1D
import numpy as np
import statistics

import tensorflow as tf 
from tensorflow import keras

# setup the GPU for tensorflow
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

class DenseNet1D_Scores_Result(Evaluation_Result_Text): pass
        
class DenseNet1D_Scores(Embedding_Evaluator_Base_Class):

    evaluator_name = "DenseNet1D_Scores"

    signal_length = None            # Number of Features (or length of the signal)
    model_width = 8                # Number of Filter or Kernels in the Input Layer (Power of 2 to avoid error)
    num_channel = None                 # Number of Input Channels
    problem_type = 'Classification' # Regression or Classification
    n_labels = None         # Number of Output Class in Classification Mode (>=2)

    evaluator_description = "DenseNet1D_Scores: Trains DenseNet1D neural network model using " + \
                         "the ds.train data subset and report the f1-score and accuracy metrics using " + \
                         "the ds.test data subset. "

    def eval_model(self, model, X, y_test):
        y_pred = tf.argmax(model.predict(X), axis=1)
        f1  = sklearn.metrics.f1_score(y_test, y_pred, average='macro')
        acc = sklearn.metrics.accuracy_score(y_test, y_pred)
        return (float(f1), float(acc))

    def set_model_parameters(self):
        self.models_to_evaluate = [
            {
                "model" : DenseNet1D.DenseNet(length = self.signal_length, num_channel = self.num_channel,
                                              num_filters = self.model_width, problem_type=self.problem_type,
                                              output_nums=self.n_labels).DenseNet169(),
                "desc"  : "DenseNet169_1D(DenseNet169_1D, n_kernels = 8)"     
            }]
        return
    
    def evaluate_embedding(self, ds : Flattenable_DataSet, n_eval_times : int = 5):
        # get train, test and validation sets
        X_train = np.array(ds.train.get_X())
        y_train = np.array(ds.train.get_y())
        X_validation = np.array(ds.validation.get_X())
        y_validation = np.array(ds.validation.get_y())
        X_test  = np.array(ds.test.get_X())
        y_test  = np.array(ds.test.get_y())

        # convert the train and validation labels to one-hot-encoding format
        y_train = keras.utils.to_categorical(y_train)
        y_validation = keras.utils.to_categorical(y_validation)

        # get the number of labels
        self.n_labels = y_train.shape[1]
        # get the signal length
        self.signal_length = X_train.shape[1]
        # get the number of channels
        try:
            self.num_channel = X_train.shape[2]
        except:
            self.num_channel = 1

        # set dense net parameters
        self.set_model_parameters()

        callbacks = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)

        for m in self.models_to_evaluate:
            model = m["model"]
            model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['accuracy'])

            # for dense net, train and evaluate the model n_eval_times to get an average and a standard deviation for the classification metrics
            if n_eval_times > 1: 
                table = [("Model description", "f1-score-mean", "f1-score-std", "accuracy-mean", "accuracy-std")]
                f1_score_set, acc_set = [], []
                for i in range(n_eval_times):
                    model.fit(X_train, y_train, epochs=300, batch_size=128, verbose=0,
                              validation_data=(X_validation, y_validation), callbacks=callbacks)
                    f1_score, acc = self.eval_model(model,X_test, y_test)
                    f1_score_set.append(f1_score), acc_set.append(acc)
                f1_score_mean_std = [statistics.mean(f1_score_set), statistics.stdev(f1_score_set)]
                acc_mean_std = [statistics.mean(acc_set), statistics.stdev(acc_set)]
                if "desc" in m: description = m["desc"]
                else:  description = str(model)
                table.append((description,"{0:.2f}%".format(100*f1_score_mean_std[0]), "{0:.2f}%".format(100*f1_score_mean_std[1]),
                                          "{0:.2f}%".format(100*acc_mean_std[0]), "{0:.2f}%".format(100*acc_mean_std[1])))
                output_text = tabulate.tabulate(table, headers="firstrow",colalign=("left","right","right","right","right"))

            else:
                table = [("Model description", "f1-score", "accuracy")]
                model.fit(X_train, y_train, epochs=300, batch_size=128, verbose=0,
                            validation_data=(X_validation, y_validation), callbacks=callbacks)
                f1_score, acc = self.eval_model(model,X_test, y_test)
                if "desc" in m: description = m["desc"]
                else:  description = str(model)
                table.append((description,"{0:.2f}%".format(100*f1_score), "{0:.2f}%".format(100*acc)))    
                output_text = tabulate.tabulate(table, headers="firstrow",colalign=("left","right","right"))

        return DenseNet1D_Scores_Result(output_text) 