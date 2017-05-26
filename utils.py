import keras
from keras.layers import Activation, Dropout, Dense, BatchNormalization
import numpy as np
import pandas as pd
import pylab as pl
import seaborn as sns
from IPython.display import SVG

def plot_model(model):
    return SVG(keras.utils.vis_utils.model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))


def plot_history(history, ymax=None):
    pl.figure(figsize=(15, 7))
    pl.plot(history.history['loss'])
    pl.plot(history.history['val_loss'])
    pl.title('model loss')
    pl.ylabel('loss')
    pl.xlabel('epoch')
    if ymax is not None:
        pl.ylim(ymax=ymax)
    pl.legend(['train', 'validation'], loc='upper left')


def compute_metrics(y_true, y_pred, clf_name):
    result = pd.Series()
    delta_znorm = (y_pred - y_true)/(1 + y_true)
    result.loc['RMSE_znorm'] = np.sqrt(np.mean((delta_znorm)**2))
    result.loc['bias_znorm'] = np.mean(delta_znorm)
    result.loc['std_znorm'] = np.std(delta_znorm)
    result.loc['RMSE'] = np.sqrt(np.mean((y_pred - y_true)**2))
    result.loc['|znorm| > 0.15 (%)'] = 100*np.sum(np.abs(delta_znorm) > 0.15)/y_true.shape[0]
    result.loc['|znorm| > 3std (%)'] = 100*np.sum(np.abs(delta_znorm) > 3*np.std(delta_znorm))/y_true.shape[0]
    result.name = clf_name
    return result 


def metrics_with_object_class(y_true, y_pred, object_class, clf_name):
    class_order = ['GALAXY', 'STAR', 'QSO']
    temp = pd.DataFrame(columns=['RMSE_znorm', 'bias_znorm', 'std_znorm', 'RMSE','|znorm| > 0.15 (%)', '|znorm| > 3std (%)'])
    temp = temp.append(compute_metrics(y_true, y_pred, clf_name))
    for c in class_order:
        mask = object_class == c
        if mask.sum() > 0:
        	temp = temp.append(compute_metrics(y_true[mask], y_pred[mask], clf_name + ' (' + c + ' only)'))
    return temp


def plot_quality(y_test, predict, object_class_test, kde=False):
    class_order = ['GALAXY', 'STAR', 'QSO']
    pl.figure(figsize=(15, 15))
    if kde:
    	sns.kdeplot(y_test, predict, gridsize=100, n_levels=5, cmap='gray')
    pl.plot([-0.5, 7.5], [-0.5, 7.5], c='k', linewidth=0.5)
    pl.ylim(-0.5, 7.5)
    pl.xlim(-0.5, 7.5)
    pl.plot([0.15/0.85, 7], [0, 0.85*7 - 0.15], '--', c='k', linewidth=0.5, label='|z_norm| < 0.15')
    pl.plot([0, (7-0.15)/1.15], [0.15, 7], '--', c='k', linewidth=0.5)
    pl.legend(loc='upper center')
    for c, color in zip(class_order, ['b', 'g', 'r']):
        pl.scatter(y_test[object_class_test == c], predict[object_class_test == c], c=color, s=0.1)
        pl.xlabel('specz')
        pl.ylabel('photoz')

    for c, color, cmap, (x_min, x_max), (y_min, y_max) in zip(class_order, ['b', 'g', 'r'], ['Blues', 'Greens', 'Reds'],
                                                              [(0, 1.5), (-0.05, 0.2), (0, 3.5)],
                                                              [(0, 1.5), (-0.05, 0.2), (0, 3.5)]):    
        pl.figure(figsize=(15, 15))
        pl.scatter(y_test[object_class_test == c], predict[object_class_test == c], c=color, s=0.1)
        if kde:
        	sns.kdeplot(y_test[object_class_test == c], predict[object_class_test == c], gridsize=100, n_levels=10, cmap=cmap)
        pl.xlabel('specz') 
        pl.ylabel('photoz')
        pl.plot([-0.5, 7.5], [-0.5, 7.5], c=color, linewidth=0.5)
        pl.ylim(y_min, y_max)
        pl.xlim(x_min, x_max)
        pl.plot([0.15/0.85, 7], [0, 0.85*7 - 0.15], '--', c='k', linewidth=0.5, label='|z_norm| < 0.15')
        pl.plot([0, (7-0.15)/1.15], [0.15, 7], '--', c='k', linewidth=0.5)
        pl.legend(loc='upper center')


def rmse_loss_keras(y_true, y_pred):
        diff = keras.backend.square((y_pred - y_true) / (keras.backend.abs(y_true) + 1))
        return keras.backend.sqrt(keras.backend.mean(diff))


def model_nn(input_dim, n_hidden_layers, dropout=0, batch_normalization=False,
			 activation='relu', neurons_decay=0, starting_power=1, l2=0,
             compile_model=True, trainable=True):
    assert dropout >= 0 and dropout < 1
    assert batch_normalization in {True, False}
    model = keras.models.Sequential()
    
    for layer in range(n_hidden_layers):
        n_units = 2**(int(np.log2(input_dim)) + starting_power - layer*neurons_decay)
        if n_units < 8:
            n_units = 8
        if layer == 0:
            model.add(Dense(units=n_units, input_dim=input_dim, name='Dense_' + str(layer + 1), 
                            kernel_regularizer=keras.regularizers.l2(l2)))
        else:
            model.add(Dense(units=n_units, name='Dense_' + str(layer + 1), 
                            kernel_regularizer=keras.regularizers.l2(l2)))
        if batch_normalization:
            model.add(BatchNormalization(name='BatchNormalization_' + str(layer + 1)))
        model.add(Activation('relu', name='Activation_' + str(layer + 1)))
        if dropout > 0:
            model.add(Dropout(dropout, name='Dropout_' + str(layer + 1)))
            
    model.add(Dense(units=1, name='Dense_' + str(n_hidden_layers+1), 
    				kernel_regularizer=keras.regularizers.l2(l2)))
    model.trainable = trainable
    if compile_model:
        model.compile(loss=rmse_loss_keras, optimizer=keras.optimizers.Adam())
    
    return model