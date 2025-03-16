import numpy as np
import tensorflow as tf

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy, Reduction
from tensorflow.keras.metrics import BinaryAccuracy, AUC

from PREDR.Utils import Recorder
from PREDR.Loader import create_dataset, load_dataset
from PREDR.Models import get_model

class Trainer:
    
    def __init__(self, args):
        self.P = dict()
        self.P.update(vars(args))
        self.recorder = Recorder()
        
        self.model = get_model(self.P['model'])(self.P)
        self.optimizer = Adam(learning_rate=self.P['learning_rate'])
        self.loss = BinaryCrossentropy(reduction=self.P['reduction_policy'])
        
        self.accuracy_metric = BinaryAccuracy()
        self.auroc_metric = AUC(curve='ROC')
        self.auprc_metric = AUC(curve='PR')
        
    
    def apply_metrics(labels, prediction):
        self.accuracy_metric.update_state(labels, prediction)
        accuracy = self.accuracy_metric.result().numpy()
        self.auroc_metric.update_state(labels, prediction)
        auroc = self.auroc_metric.result().numpy()
        self.auprc_metric.update_state(labels, prediction)
        auprc = self.auprc_metric.result().numpy()
        
        self.accuracy_metric.reset_state()
        self.auroc_metric.reset_state()
        self.auprc_metric.reset_state()
        
        return accuracy, auroc, auprc
        
    
    def training_step(self, inputs, labels):
        with tf.GradientTape() as tape:
            prediction = self.model(inputs)
            loss = self.loss(labels, prediction)
            
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        self.recorder.update_state('train', *self.apply_metrics(labels, prediction))
        
    
    def test_step(self, inputs, labels, val=True):
        prediction = self.model(inputs)
        loss = self.loss(labels, prediction)
        
        self.recorder.update_state('val' if val else 'test', *self.apply_metrics(labels, prediction))
        
        
    def evaluate(self, dataset, val=True):
        for X, y in dataset:
            self.test_step(X, y, val)
            
    
    def train(self):
        # main train sequence
        
        pass