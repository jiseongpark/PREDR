import numpy as np
import tensorflow as tf
from scipy import sparse as sp
import datetime


def create_batch(X_list, A_list, sparse=False):
    A_out = sp.block_diag(list(A_list))
    if not sparse:
        A_out = A_out.toarray()
    X_out = np.vstack(X_list)
    n_nodes = np.array([np.shape(a_)[0] for a_ in A_list])
    I_out = np.repeat(np.arange(len(n_nodes)), n_nodes)
    return X_out, A_out, I_out


class Recorder:
    
    def __init__(self, name='record'):
        self.rec = {
            'train_loss': [],
            'val_loss': [],
            'test_loss': [[]],

            'train_acc': [],
            'val_acc': [],
            'test_acc': [[]],

            'train_auroc': [],
            'val_auroc': [],
            'test_auroc': [[]],

            'train_auprc': [],
            'val_auprc': [],
            'test_auprc': [[]],
        }
        
        self.name = name
        
    
    def print_progress(self, epoch):
        ep = epoch + 1
        tr_loss = np.mean(self.rec['train_loss'][epoch])
        tr_acc = np.mean(self.rec['train_acc'][epoch])
        tr_roc = np.mean(self.rec['train_auroc'][epoch])
        tr_prc = np.mean(self.rec['train_auprc'][epoch])
        val_loss = np.mean(self.rec['val_loss'][epoch])
        val_acc = np.mean(self.rec['val_acc'][epoch])
        val_roc = np.mean(self.rec['val_auroc'][epoch])
        val_prc = np.mean(self.rec['val_auprc'][epoch])

        print(f'Ep{ep:02d}, '
              f'Train Loss {tr_loss:.5f}, '
              f'ARP {tr_acc:.3f}|{tr_roc:.3f}|{tr_prc:.3f},  '
              f'Val Loss {val_loss:.5f}, '
              f'ARP {val_acc:.3f}|{val_roc:.3f}|{val_prc:.3f}'
             )
        
        
    def print_test_perf(self):
        tr_loss = np.mean(self.rec['test_loss'][-1])
        tr_acc = np.mean(self.rec['test_acc'][-1])
        tr_roc = np.mean(self.rec['test_auroc'][-1])
        tr_prc = np.mean(self.rec['test_auprc'][-1])

        print(f'Test '
              f'Loss {tr_loss:.5f}, '
              f'ARP {tr_acc:.3f}|{tr_roc:.3f}|{tr_prc:.3f},  '
             )
    
    
    def reset_record(self):
        for key in self.rec:
            self.rec[key] = []
            
    
    def save_record(self, fname=None):
        if fname is None:
            now = datetime.datetime.now()
            fname = now.strftime('%Y%m%d-%H%M')
        
        with open(f'./log/{fname}.json', 'w') as fp:
            json.dump(self.rec, fp)
    
    
    def update_epoch(self):
        for key in self.rec:
            if key.startswith('test'): continue
            self.rec[key].append([])
            
    
    def update_state(self, proc='train', loss=0.0, acc=0.0, roc=0.0, pr=0.0):
        self.rec[proc+'_loss'][-1].append(loss)
        self.rec[proc+'_acc'][-1].append(acc)
        self.rec[proc+'_auroc'][-1].append(roc)
        self.rec[proc+'_auprc'][-1].append(pr)