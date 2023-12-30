import os
import numpy as np
import mlflow.sklearn
import tensorflow as tf
import sklearn
import keras
from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix
from  mlflow.tracking import MlflowClient
import matplotlib.pyplot as plt 
# from sklearn.metrics import confusion_matrix
# from sklearn.utils.class_weight import compute_class_weight
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn import metrics

class CNNModel:
    
    def __init__(self, params={}):
        """
        Constructor for Convolutional Neural Network
        :param params: dictionary to CNN Model
        """
        self._params = params
        self._cnn = self.build_sequential_model(params)

    @classmethod
    def new_instance(cls, params={}):
        return cls(params)
    
    @classmethod
    def build_sequential_model(self, model_parameters):
        """
        This method builds a sequential TF model from a 
        dictionary of parameters. It uses layer types,
        number of filters, kernel size, activation,
        and regularization.
        """
               
        model = tf.keras.Sequential()
        for layer_params in model_parameters['layers']:
            layer_type = layer_params['type']

            if layer_type == 'Rescaling':
                model.add(layers.Rescaling(scale=layer_params['scaling_factor']))
            elif layer_type == 'Conv2D':
                model.add(layers.Conv2D(
                    filters=layer_params['filters'],
                    kernel_size=layer_params['kernel_size'],
                    activation=layer_params['activation']
                ))
            elif layer_type == 'MaxPooling2D':
                model.add(layers.MaxPooling2D())
            elif layer_type == 'Flatten':
                model.add(layers.Flatten())
            elif layer_type == 'Dense':
                model.add(layers.Dense(
                    units=layer_params['units'],
                    activation=layer_params['activation'],
                    kernel_regularizer=layer_params.get('kernel_regularizer')
                ))
            elif layer_type == 'Dropout':
                model.add(layers.Dropout(rate=layer_params['rate']))
            else:
                raise ValueError(f"Unsupported layer type: {layer_type}")

        return model
    
    @classmethod
    def compile_model(self, passed_model, model_params):
        """
        Compiles a model using the hyperparameters from the model's parameters
        using an optimizer, loss function, and metrics.
        """
        params = model_params['hyperparams']
        
        passed_model.compile(
        optimizer = getattr(tf.keras.optimizers, params['optimizer'])(learning_rate = params['learning_rate'], weight_decay = params['decay']),     
        loss = params['loss_func'],
        metrics = params['metrics'])

        
        
    def mlflow_run(self, dataset, r_name="CNN-Model-Experiment"):
        """
        This method trains, computes metrics, and logs all metrics, parameters,
        and artifacts for the current run using the MLflow APIs
        :param df: pandas dataFrame
        :param r_name: Name of the run as logged by MLflow
        :return: MLflow Tuple (ExperimentID, runID)
        """
        with mlflow.start_run(run_name=r_name) as run:

            # get current run and experiment id
            runID = run.info.run_uuid
            experimentID = run.info.experiment_id
            
            # Create Train and Test Data
            X_train = datasets['X_train']
            y_train = datasets['y_train']
            X_val = datasets['X_val']
            y_val = datasets['y_val']

            # train and predict
            self.compile_model(self.model, self.params)
            num_epochs = self.params['training_config']['epochs']
            class_weights = self.params['training_config']['class_weights']
            batch_size = self.params['training_config']['batch_size']
            history = self.model.fit(X_train, y_train, epochs=num_epochs, batch_size = batch_size class_weight=class_weights, validation_data=(X_val, y_val))
                                            
            # Log model and params using the MLflow APIs
            mlflow.sklearn.log_model(self.model, "cnn-class-model")
            mlflow.log_params(self.params)

            # compute  regression evaluation metrics 
#             mae = metrics.mean_absolute_error(y_test, y_pred)
#             mse = metrics.mean_squared_error(y_test, y_pred)
#             rmse = np.sqrt(mse)
#             r2 = metrics.r2_score(y_test, y_pred)

            y_pred = model.predict(X_val)
            y_pred = np.argmax(y_pred, axis=1)
            
            confusion_matrix = confusion_matrix(y_val, y_pred)
            
            FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)  
            FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
            TP = np.diag(confusion_matrix)
            TN = confusion_matrix.values.sum() - (FP + FN + TP)
            
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            f1_score = ( 2 * precision * recall ) / ( precision + recall )
            accuracy = ( TP + TN ) / ( TP + TN + FP + FN )

            # Log metrics
            mlflow.log_metric("Precision", precision)
            mlflow.log_metric("Recall", recall)
            mlflow.log_metric("F1 Score", f1_score)
            mlflow.log_metric("Accuracy", accuracy)
            mlflow.log_metric("Confusion Matrix", confusion_matrix)
            
            # plot graphs and save as artifacts
#             (fig, ax) = Utils.plot_graphs(self.estimators, self.rmse, "Random Forest Estimators", 
#                                           "Root Mean Square", "Root Mean Square vs Estimators")

            # create temporary artifact file name and log artifact
#             temp_file_name = Utils.get_temporary_directory_path("cnn-", ".png")
#             temp_name = temp_file_name.name
#             try:
#                 fig.savefig(temp_name)
#                 mlflow.log_artifact(temp_name, "rmse_estimators_plots")
#             finally:
#                 temp_file_name.close()  # Delete the temp file

            # print some data
#             print("-" * 100)
            print("Inside MLflow Run with run_id {} and experiment_id {}".format(runID, experimentID))
            print("Precision        :", precision)
            print('Recall           :', recall)
            print('F1 Score         :', f1_score)
            print('Accuracy         :', accuracy)
            print('Confusion Matrix :', confusion_matrix)
            
            plt.plot(history.history['acc'])
            plt.plot(history.history['val_acc'])
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'validation'], loc='upper left')
            plt.show()
            
            return (experimentID, runID)

    
    @property
    def model(self):
        """
        Return the model created
        :return: instance of CNN
        """
        return self._cnn
    
    @property
    def params(self):
      """
      Getter for model parameters 
      """
      return self._params