import os
import numpy as np
# import mlflow.sklearn
import tensorflow as tf
# import sklearn
import keras
from tensorflow.keras import layers, models
# from sklearn.metrics import confusion_matrix
# from sklearn.utils.class_weight import compute_class_weight
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn import metrics
# from  mlflow.tracking import MlflowClient

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
        print('here')
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
            self._rf.fit(X_train, y_train)
            self._cnn.fit(X_train, y_train, epochs=30, class_weight=dict(enumerate(class_weights)), validation_data=(X_val, y_val))
            
            
            y_pred = self._rf.predict(X_test)
                      
               
            # Log model and params using the MLflow APIs
            mlflow.sklearn.log_model(self.model, "cnn-class-model")
            mlflow.log_params(self.params)

            # compute  regression evaluation metrics 
            mae = metrics.mean_absolute_error(y_test, y_pred)
            mse = metrics.mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = metrics.r2_score(y_test, y_pred)

            # Log metrics
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)

            # update global class instance variable with values
            self.rmse.append(rmse)
            self.estimators.append(self.params["n_estimators"])

            # plot graphs and save as artifacts
            (fig, ax) = Utils.plot_graphs(self.estimators, self.rmse, "Random Forest Estimators", 
                                          "Root Mean Square", "Root Mean Square vs Estimators")

            # create temporary artifact file name and log artifact
            temp_file_name = Utils.get_temporary_directory_path("rmse_estimators-", ".png")
            temp_name = temp_file_name.name
            try:
                fig.savefig(temp_name)
                mlflow.log_artifact(temp_name, "rmse_estimators_plots")
            finally:
                temp_file_name.close()  # Delete the temp file

            # print some data
            print("-" * 100)
            print("Inside MLflow Run with run_id {} and experiment_id {}".format(runID, experimentID))
            print("Estimator trees        :", self.params["n_estimators"])
            print('Mean Absolute Error    :', mae)
            print('Mean Squared Error     :', mse)
            print('Root Mean Squared Error:', rmse)
            print('R2                     :', r2)
            
            return (experimentID, runID)

    
    @property
    def model(self):
        """
        Return the model created
        :return: instance of CNN
        """
        return self._cnn
    