import numpy as np
import scipy.sparse as sp
import tensorflow as tf

from tensorflow.keras import Input
from tensorflow.keras.layers import Layer, Dense, Flatten
from tensorflow.keras.models import Model

from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC



def get_model(model_name):
    model_catalogs = {
        'predr': PREDR,
        'dtinet': Baseline_DTINet,
        'mlp': Baseline_MLP,
        'svm': Baseline_SVM
    }
    return model_catalogs[model_name]



'''Layer for calculate pairwise concatenation for given feature matrix'''
class pairwise_concat(Layer):
    
    def __init__(self, dim=5):
        super().__init__()
        self.dim = dim
        
    def call(self, inputs):
        X_l, X_r = inputs
        num_l, num_r = len(X_l), len(X_r)
        
        X_lhs = tf.reshape(tf.tile(X_l, multiples=[1, num_r]), [num_l * num_r, self.dim])
        X_rhs = tf.tile(X_r, multiples=[num_l, 1])
        X_cat = tf.concat([X_lhs, X_rhs], axis=1)
        return X_cat
    
    
    

'''Main model PREDR'''
class PREDR(Model):
    
    def __init__(self, P):
        super().__init__()
        
        self.P = P
        chg_dims, dg_dims = self.P['chg_dims'], self.P['dg_dims']
        
        self.pairwise_concat_layer = pairwise_concat()
        
        self.ChG_embedding_layers = list()
        for idx, dim in enumerate(chg_dims):
            self.ChG_embedding_layers.append(
                Dense(dim, activation=self.P['activ'], name=f'ChG_Dense_{idx}'))
        self.ChG_relation_recon = Dense(1, activation='elu', name='ChG_relation')
        
        self.DG_embedding_layers = list()
        for idx, dim in enumerate(dg_dims):
            self.DG_embedding_layers.append(
                Dense(dim, activation=self.P['activ'], name=f'DG_Dense_{idx}'))
        self.DG_relation_recon = Dense(1, activation='elu', name='DG_relation')

        self.sigmoid = tf.keras.activations.sigmoid
       
    
    def call(self, inputs):
        X_ch, X_g, X_d = inputs
        num_ch = tf.shape(X_ch).numpy()[0]
        num_g = tf.shape(X_g).numpy()[0]
        num_d = tf.shape(X_d).numpy()[0]
        
        X_chg = self.pairwise_concat_layer([X_ch, X_g])
        for chg_dnn in self.ChG_embedding_layers:
            X_chg = chg_dnn(X_chg)
        Relation_chg = self.ChG_relation_recon(X_chg)
        A_chg = tf.reshape(Relation_chg, [num_ch ,num_g])
        
        X_gd = self.pairwise_concat_layer([X_g, X_d])
        for dg_dnn in self.DG_embedding_layers:
            X_gd = dg_dnn(X_gd)
        Relation_gd = self.DG_relation_recon(X_gd)
        A_gd = tf.reshape(Relation_gd, [num_g, num_d])
        
        A_chd = tf.matmul(A_chg, A_gd)
        A_chd_flat = tf.reshape(A_chd, (-1, 1))
        output = self.sigmoid(A_chd_flat)
        
        return output, A_chg, A_gd
    

    
    
'''Baseline model DTINet(A network integration approach for drug-target interaction prediction and computational drug repositioning from heterogeneous information [Yunan Luo et. al., nature communications, 2017])'''
class Baseline_DTINet(Model):
    
    def __init__(self, dims):
        super().__init__()
        
        fe_dims, gnn_dims = dims
        
        self.feature_embedding = list()
        for dim in fe_dims:
            self.feature_embedding.append(Dense(dim, activation='relu'))
        
    def call(self, inputs):
        X_ch, X_g, X_d, A = inputs
        num_ch, num_g, num_d = len(X_ch), len(X_g), len(X_d)
        
        Z_ch = self.feature_embedding[0](X_ch)
        Z_d = self.feature_embedding[2](X_d)
        
        Z_chd = tf.linalg.matmul(Z_ch, Z_d, transpose_b=True)
        A_chd_flat = tf.reshape(Z_chd, (-1, 1))
        
        output =tf.keras.activations.sigmoid(A_chd_flat)
        return output, Z_chd
    
    
    
    
'''Baseline model MLP'''
class Baseline_MLP:
    
    def __init__(self, hidden_layers):
        self.mlp = MLPClassifier(hidden_layer_sizes=hidden_layers, random_state=42)
    
    
    def __call__(self, inputs):
        return self.mlp.predict(inputs)
    
    
    def fit(self, inputs, labels):
        self.mlp.fit(inputs, labels)

        
        
        
'''Baseline model SVM'''
class Baseline_SVM:
    
    def __init__(self):
        self.svm = LinearSVC()
        
        
    def __call__(self, inputs):
        return self.svm.predict(inputs)
    
    
    def fit(self, inputs, labels):
        self.svm.fit(inputs, labels)

