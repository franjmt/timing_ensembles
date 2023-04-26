import pickle
import numpy as np
import os.path
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import jaccard_score
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.utils import shuffle
from sklearn.impute import SimpleImputer

from scipy import stats
from scipy.stats import f_oneway



def get_trials_all(sc, si, lc, li, n_total):
    s_total = len(sc) + len(si)
    proportion_sc = round(len(sc)/s_total*100)
    l_total = len(lc) + len(li)
    proportion_lc = round(len(lc)/l_total*100)

    proportion_c_total = round(np.mean([proportion_sc, proportion_lc]))

    n_correct = int(n_total*((proportion_c_total/2)/100)) #number of trial to select for either short or long correct trials
    
    sc_select = np.random.choice(sc, n_correct, replace=False)
    lc_select = np.random.choice(lc, n_correct, replace=False)
    
    proportion_i_total = 100 - proportion_c_total
    
    n_incorrect = int(n_total*((proportion_i_total/2)/100))
    
    if n_incorrect <= len(si) and n_incorrect <= len(li):
        si_select = np.random.choice(si, n_incorrect, replace=False)
        li_select = np.random.choice(li, n_incorrect, replace=False)
    elif n_incorrect <= len(si) and n_incorrect > len(li):
        si_select = np.random.choice(si, n_incorrect, replace=False)
        li_select = li
    elif n_incorrect > len(si) and n_incorrect <= len(li):
        si_select = si
        li_select = np.random.choice(li, n_incorrect, replace=False)
    else:
        si_select = si
        li_select = li
    
    return sc_select, si_select, lc_select, li_select



def get_trials_correct(sc, lc, n_total):
    sc_select = np.random.choice(sc, int(n_total/2), replace=False)
    lc_select = np.random.choice(lc, int(n_total/2), replace=False)
    
    return sc_select, lc_select


def X_mean(align_dff, zero, interval_epoch, fb, fa, trials):
    # fb: frames to take before first tone
    # fa: frames to take after tone
    
    dff = align_dff[:, zero-fb:zero+interval_epoch+fa, trials]
    dff_red = np.mean(dff, axis=1)
    
    return(dff_red)


def classifier(X, y, X_shift, n_neu):
    svm = SVC(C=1/np.log(n_neu+1), kernel='linear', degree=2)
    LR = LogisticRegression(penalty="l2", C=1/np.log(n_neu+1), max_iter = 5000)
    cv = ShuffleSplit(n_splits=100, test_size=0.25, random_state=0)
    
    accu_svm = cross_val_score(svm, X, y, cv=cv)
    accu_svm_shift = cross_val_score(svm, X_shift, y, cv=cv)
    
    accu_LR = cross_val_score(LR, X, y, cv=cv)
    accu_LR_shift = cross_val_score(LR, X_shift, y, cv=cv)
    
    return accu_svm, accu_svm_shift, accu_LR, accu_LR_shift



def classifier_y(X, y, y_shuffle, n_neu):
    svm = SVC(C=1/np.log(n_neu+1), kernel='linear', degree=2)
    LR = LogisticRegression(penalty="l2", C=1/np.log(n_neu+1), max_iter = 5000)
    cv = ShuffleSplit(n_splits=100, test_size=0.25, random_state=0)
    
    accu_svm = cross_val_score(svm, X, y, cv=cv)
    accu_svm_shuffle = cross_val_score(svm, X, y_shuffle, cv=cv)
    
    accu_LR = cross_val_score(LR, X, y, cv=cv)
    accu_LR_shuffle = cross_val_score(LR, X, y_shuffle, cv=cv)
    
    return accu_svm, accu_svm_shuffle, accu_LR, accu_LR_shuffle



def X_activity(align_dff, zero, interval_epoch, fb, fa, trials):
    # fb: frames to take before first tone
    # fa: frames to take after tone
    dff = align_dff[:, zero-fb:zero+interval_epoch+fa, trials]
    dff_red = np.trapz(dff, axis=1)
    
    return(dff_red)


def classification_frame(align_dff, shift_dff, zero, short_trials, long_trials, mean_short_epoch, mean_long_epoch):

    acc_svm_all = []
    accu_svm_shift_all = []
    accu_LR_all = []
    accu_LR_shift_all = []

    neu = align_dff.shape[0]

    for j in range(-1, mean_short_epoch):

        dff_short_red = X_activity(align_dff, zero, j, 0, 0, short_trials)
        dff_short_red_shift = X_activity(shift_dff, zero, j, 0, 0, short_trials)

        dff_long_red = X_activity(align_dff, zero, j, 0, 0, long_trials)
        dff_long_red_shift = X_activity(shift_dff, zero, j, 0, 0, long_trials)

        dff_all = np.hstack((dff_short_red, dff_long_red))
        y_all = np.hstack((np.zeros(dff_short_red.shape[1]), np.ones(dff_long_red.shape[1])))
        dff_all_shift = np.hstack((dff_short_red_shift, dff_long_red_shift))

        X, X_shift, y = shuffle(dff_all.T, dff_all_shift.T, y_all)
#         y_shuffle = shuffle(y)

        accu_svm, accu_svm_shift, accu_LR, accu_LR_shift = classifier(X, y, X_shift, neu)
#         accu_svm, accu_svm_shift, accu_LR, accu_LR_shift = classifier_y(X, y, y_shuffle, neu_rMO)

        acc_svm_all.append(np.nanmean(accu_svm))
        accu_svm_shift_all.append(np.nanmean(accu_svm_shift))
        accu_LR_all.append(np.nanmean(accu_LR))
        accu_LR_shift_all.append(np.nanmean(accu_LR_shift))


    for i in range(mean_short_epoch, mean_long_epoch):

        dff_long_red = X_activity(align_dff, zero, i, 0, 0, long_trials)
        dff_long_red_shift = X_activity(shift_dff, zero, i, 0, 0, long_trials)

        dff_all = np.hstack((dff_short_red, dff_long_red))
        y_all = np.hstack((np.zeros(dff_short_red.shape[1]), np.ones(dff_long_red.shape[1])))
        dff_all_shift = np.hstack((dff_short_red_shift, dff_long_red_shift))

        X, X_shift, y = shuffle(dff_all.T, dff_all_shift.T, y_all)

        accu_svm, accu_svm_shift, accu_LR, accu_LR_shift = classifier(X, y, X_shift, neu)

        acc_svm_all.append(np.nanmean(accu_svm))
        accu_svm_shift_all.append(np.nanmean(accu_svm_shift))
        accu_LR_all.append(np.nanmean(accu_LR))
        accu_LR_shift_all.append(np.nanmean(accu_LR_shift))

    return acc_svm_all, accu_svm_shift_all, accu_LR_all, accu_LR_shift_all



def classifier_test_train(X_train, y_train, X_test, y_test, X_test_shuffle, n_neu):
    svm = SVC(C=1/np.log(n_neu+1), kernel='linear', degree=2).fit(X_train, y_train)
    LR = LogisticRegression(penalty="l2", C=1/np.log(n_neu+1), max_iter = 5000).fit(X_train, y_train)

    accu_svm = svm.score(X_test, y_test)
    accu_svm_shuffle = svm.score(X_test_shuffle, y_test)
    accu_LR = LR.score(X_test, y_test)
    accu_LR_shuffle = LR.score(X_test_shuffle, y_test)


    return accu_svm, accu_svm_shuffle, accu_LR, accu_LR_shuffle


def classification_frame_w(align_dff, shift_dff, zero, short_trials, long_trials, mean_long_epoch):

    n_neu = align_dff.shape[0]

    dff_short_red = X_activity(align_dff, zero-1, zero+4, 0, 0, short_trials)
    dff_short_red_shift = X_activity(shift_dff, zero-1, zero+4, 0, 0, short_trials)

    dff_long_red = X_activity(align_dff, zero-1, zero+4, 0, 0, long_trials)
    dff_long_red_shift = X_activity(shift_dff, zero-1, zero+4, 0, 0, long_trials)

    dff_all = np.hstack((dff_short_red, dff_long_red))
    y_all = np.hstack((np.zeros(dff_short_red.shape[1]), np.ones(dff_long_red.shape[1])))
    dff_all_shift = np.hstack((dff_short_red_shift, dff_long_red_shift))

    X, X_shift, y = shuffle(dff_all.T, dff_all_shift.T, y_all)
    y_shuffle = shuffle(y)

#     acc_svm_short_epoch, accu_svm_shift_short_epoch, accu_LR_short_epoch, accu_LR_shift_short_epoch = classifier(X, y, X_shift, n_neu)
    acc_svm_short_epoch, accu_svm_shift_short_epoch, accu_LR_short_epoch, accu_LR_shift_short_epoch = classifier_y(X, y, y_shuffle, n_neu)

    svm_short_epoch = [np.mean(acc_svm_short_epoch), np.mean(accu_svm_shift_short_epoch)]
    lr_short_epoch = [np.mean(accu_LR_short_epoch), np.mean(accu_LR_shift_short_epoch)]
    
    acc_svm_long_epoch = []
    acc_svm_shift_long_epoch = []
    acc_lr_long_epoch = []
    acc_lr_shift_long_epoch = []
    
    for i,j in enumerate(np.arange(4,mean_long_epoch)):

        dff_long_red = X_activity(align_dff, zero+4+i, zero+4+j, 0, 0, long_trials)
        dff_long_red_shift = X_activity(shift_dff, zero+4+i, zero+4+j, 0, 0, long_trials)

        dff_all = np.hstack((dff_short_red, dff_long_red))
        y_all = np.hstack((np.zeros(dff_short_red.shape[1]), np.ones(dff_long_red.shape[1])))
        dff_all_shift = np.hstack((dff_short_red_shift, dff_long_red_shift))

        X, X_shift, y = shuffle(dff_all.T, dff_all_shift.T, y_all)

#         accu_svm, accu_svm_shift, accu_LR, accu_LR_shift = classifier(X, y, X_shift, n_neu)
        accu_svm, accu_svm_shift, accu_LR, accu_LR_shift = classifier_y(X, y, y_shuffle, n_neu)

        acc_svm_long_epoch.append(np.mean(accu_svm))
        acc_svm_shift_long_epoch.append(np.mean(accu_svm_shift))
        acc_lr_long_epoch.append(np.mean(accu_LR))
        acc_lr_shift_long_epoch.append(np.mean(accu_LR_shift))
    

    return svm_short_epoch, lr_short_epoch, acc_svm_long_epoch, acc_svm_shift_long_epoch, acc_lr_long_epoch, acc_lr_shift_long_epoch


def classification_add_frame(align_dff, zero, interval_types, mean_epoch):


    n_neurons, time_bin, trials = align_dff.shape
    
    svm = SVC(C=1/np.log(n_neurons+1), kernel='linear', degree=2)
    model = LogisticRegression(penalty="l2", C=1/np.log(n_neurons+1), max_iter = 5000)
    cv = ShuffleSplit(n_splits=100, test_size=0.25, random_state=0)
    
    acc_array_lr = []
    acc_shuffle_array_lr = []
    acc_array_svm = []
    acc_shuffle_array_svm = []

    for i in range(-1, mean_epoch+1):
        X = np.trapz(align_dff[:,zero-1:zero-1+i,:], axis=1)
        y = interval_types
        y_shuffle = shuffle(y)

        accu_lr = cross_val_score(model, X.T, y, cv=cv)
        acc_array_lr.append(np.nanmean(accu_lr))
        
        accu_shuffle_lr = cross_val_score(model, X.T, y_shuffle, cv=cv)
        acc_shuffle_array_lr.append(np.nanmean(accu_shuffle_lr))
        
        accu_svm = cross_val_score(svm, X.T, y, cv=cv)
        acc_array_svm.append(np.nanmean(accu_svm))
        
        accu_shuffle_svm = cross_val_score(svm, X.T, y_shuffle, cv=cv)
        acc_shuffle_array_svm.append(np.nanmean(accu_shuffle_svm))
    
    lr_dat_cv = np.array(acc_array_lr)
    lr_shu_cv = np.array(acc_shuffle_array_lr)
    svm_dat_cv = np.array(acc_array_svm)
    svm_shu_cv = np.array(acc_shuffle_array_svm)

    return svm_dat_cv, svm_shu_cv, lr_dat_cv, lr_shu_cv
    


def classifier_time(neu_traces, interval_types, w):
    ###  w: window of neural activity to be used
    
    n_neurons, time_bin, trials = neu_traces.shape
    
    svm = SVC(C=1/np.log(n_neurons+1), kernel='linear', degree=2)
    model = LogisticRegression(penalty="l2", C=1/np.log(n_neurons+1), max_iter = 5000)
    cv = ShuffleSplit(n_splits=100, test_size=0.25, random_state=0)
    
    acc_array_lr = np.zeros((100))
    acc_shuffle_array_lr = np.zeros((100))
    acc_array_svm = np.zeros((100))
    acc_shuffle_array_svm = np.zeros((100))


    for i in range(time_bin-w):
        X = np.trapz(neu_traces[:,i:i+w,:], axis=1)
        y = interval_types
        y_shuffle = shuffle(y)

        accu_lr = cross_val_score(model, X.T, y, cv=cv)
        acc_array_lr = np.vstack((acc_array_lr, accu_lr))
        
        accu_shuffle_lr = cross_val_score(model, X.T, y_shuffle, cv=cv)
        acc_shuffle_array_lr = np.vstack((acc_shuffle_array_lr, accu_shuffle_lr))
        
        accu_svm = cross_val_score(svm, X.T, y, cv=cv)
        acc_array_svm = np.vstack((acc_array_svm, accu_svm))
        
        accu_shuffle_svm = cross_val_score(svm, X.T, y_shuffle, cv=cv)
        acc_shuffle_array_svm = np.vstack((acc_shuffle_array_svm, accu_shuffle_svm))

    acc_array_lr = np.delete(acc_array_lr, 0, axis=0)
    acc_shuffle_array_lr = np.delete(acc_shuffle_array_lr, 0, axis=0)
    acc_array_svm = np.delete(acc_array_svm, 0, axis=0)
    acc_shuffle_array_svm = np.delete(acc_shuffle_array_svm, 0, axis=0)
    
    lr_dat_cv = np.nanmean(acc_array_lr, axis=1)
    lr_shu_cv = np.nanmean(acc_shuffle_array_lr, axis=1)
    svm_dat_cv = np.nanmean(acc_array_svm, axis=1)
    svm_shu_cv = np.nanmean(acc_shuffle_array_svm, axis=1)

    return lr_dat_cv, lr_shu_cv, svm_dat_cv, svm_shu_cv




def plot_distribution(dff_sc, dff_lc, dff_sc_shift, dff_lc_shift, cortex, file_name):
    global store_location, mouse_name, session

    t_stats, p_value = stats.ttest_ind(np.mean(dff_sc, axis=1), np.mean(dff_lc, axis=1))
    t_stats_shift, p_value_shift = stats.ttest_ind(np.mean(dff_sc_shift, axis=1), np.mean(dff_lc_shift, axis=1))


    fig, ax = plt.subplots(1,2, dpi=100)
    a = ax[0].hist(np.mean(dff_sc, axis=1), color='violet', alpha=0.7)
    ax[0].hist(np.mean(dff_lc, axis=1), color='gold', alpha=0.6)
    ax[0].set_title('population integral')
    ax[0].text(0, -11, 'p_value = '+'{:.2e}'.format(p_value), fontsize=8)

    b = ax[1].hist(np.mean(dff_sc_shift, axis=1), color='violet', alpha=0.7)
    ax[1].hist(np.mean(dff_lc_shift, axis=1), color='gold', alpha=0.6)
    ax[1].set_title('population integral\nshifted')
    ax[1].text(0, -11, 'p_value = '+'{:.2e}'.format(p_value_shift), fontsize=8)
    plt.tight_layout()
    plt.savefig(store_location+'/' + mouse_name + session + '_' + cortex + '_' + file_name + '.png')



