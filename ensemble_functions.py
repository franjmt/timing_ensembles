
import numpy as np
from scipy import stats
from sklearn.utils import shuffle
from classifiers_functions import X_mean, classifier
from scipy.stats import mannwhitneyu
from smooth import smooth


def ensemble_epoch(ens_trial_matrix, interval_epoch, delay_epoch, tone_frames, resp_time, trials):
    """
    input:
    ens_trial_matrix: 3D array, neurons x time x trials
    interval_epoch: int, number of frames in interval epoch
    delay_epoch: int, number of frames in delay epoch
    tone_frames: 2D array, tone onset and offset frames for each trial
    resp_time: 1D array, response time for each trial
    trials: 1D array, trial numbers
    output:
    ens_interval: 2D array, ensemble x trials, ensemble activity in interval epoch
    ens_delay: 2D array, ensemble x trials, ensemble activity in delay epoch
    ens_resp: 2D array, ensemble x trials, ensemble activity in response epoch
    """
    ens_interval = np.zeros((interval_epoch+1, len(trials)))
    ens_delay = np.zeros((delay_epoch+1, len(trials)))
    ens_resp = np.zeros((8+1, len(trials)))

    for j,i in enumerate(trials):
        if i>=ens_trial_matrix.shape[2]:
            break
        else:
            ens_loc_interval = np.where(ens_trial_matrix[:,tone_frames[i,0]:tone_frames[i,0]+interval_epoch+1, i]==1)
            ens_interval[ens_loc_interval[1],j] = ens_loc_interval[0]+1

            ens_loc_delay = np.where(ens_trial_matrix[:,tone_frames[i,1]:tone_frames[i,1]+delay_epoch+1, i]==1)
            ens_delay[ens_loc_delay[1],j] = ens_loc_delay[0]+1

            ens_loc_resp = np.where(ens_trial_matrix[:,resp_time[i].astype(int):resp_time[i].astype(int)+8+1, i]==1)
            ens_resp[ens_loc_resp[1],j] = ens_loc_resp[0]+1
        
    return ens_interval, ens_delay, ens_resp


def ens_prob_trial(ens_epoch, n_ens):
    """
    input:
    ens_epoch: 2D array, ensemble x trials, ensemble activity in epoch
    n_ens: int, number of ensembles
    output:
    ens_prob_trials: 2D array, ensemble x trials, probability of each ensemble in each trial
    """
    for i in range(ens_epoch.shape[1]):
        this_ens_cnts = np.bincount(ens_epoch[:,i].astype(int), minlength=len(n_ens))
        this_ens_prob = this_ens_cnts/np.sum(this_ens_cnts)
        if i==0:
            ens_prob_trials = this_ens_prob
        else:
            ens_prob_trials = np.vstack((ens_prob_trials, this_ens_prob))
    
    return ens_prob_trials


def each_ens_stats(each_ens_prob_trails, each_ens_shu_prob_trails):
    """
    input:
    each_ens_prob_trails: 2D array, ensemble x trials, probability of each ensemble in each trial
    each_ens_shu_prob_trails: 2D array, ensemble x trials, probability of each ensemble in each trial in shuffled data
    output:
    p_value: 1D array, p value of each ensemble
    diff: 1D array, difference of probability of each ensemble between real data and shuffled data
    """
    s_ttest, p_value = mannwhitneyu(each_ens_prob_trails, each_ens_shu_prob_trails, axis=0, method='exact')
    diff = np.mean(each_ens_prob_trails, axis=0)/np.mean(each_ens_shu_prob_trails, axis=0)
    
    return p_value, diff


def get_CV_classifiers(alignCdec, shift_alignCdec, zero, mean_short_epoch, mean_long_epoch, sc_trials, lc_trials):
    """
    input:
    alignCdec: 3D array, neurons x time x trials, aligned neural activity to first tone
    shift_alignCdec: 3D array, neurons x time x trials, aligned neural activity to first tone in shuffled data  
    zero: int, frame number of first tone  
    mean_short_epoch: int, number of frames in short epoch
    mean_long_epoch: int, number of frames in long epoch
    sc_trials: 1D array, trial numbers of short interval trials 
    lc_trials: 1D array, trial numbers of long interval trials
    output:
    Dict_classifier: dictionary, keys are 'accu_svm', 'accu_svm_shuffle', 'p_value', values are accuracy of each classifier
    """
    
    dff_sc_red = X_mean(alignCdec, zero, mean_short_epoch, 0, 1, sc_trials)
    dff_lc_red = X_mean(alignCdec, zero, mean_long_epoch, 0, 1, lc_trials)
    shu_dff_sc_red = X_mean(shift_alignCdec, zero, mean_short_epoch, 0, 1, sc_trials)
    shu_dff_lc_red = X_mean(shift_alignCdec, zero, mean_long_epoch, 0, 1, lc_trials)

    dff_all_red = np.hstack((dff_sc_red, dff_lc_red))
    shu_dff_all_red = np.hstack((shu_dff_sc_red, shu_dff_lc_red))
    type_all = np.hstack((np.zeros(len(sc_trials)), np.ones(len(lc_trials))))

    X, shu_X, y = shuffle(dff_all_red.T, shu_dff_all_red.T, type_all)

    accu_svm, accu_svm_shuffle, accu_LR, accu_LR_shuffle = classifier(X, y, shu_X, alignCdec.shape[0])
    t_stats_svm, p_value_svm = stats.ttest_ind(accu_svm_shuffle, accu_svm)
    t_stats_LR, p_value_LR = stats.ttest_ind(accu_LR_shuffle, accu_LR)
    
    Dict_classifier = {'CV_svm' : accu_svm,
                      'CV_svm_shu' : accu_svm_shuffle,
                      'p_value_svm' : p_value_svm}

    return Dict_classifier


def cdec_corr_by_trial(cdec, corr_cell_trials, trials, all_trials):
    """
    input:
    cdec: 3D array, neurons x time x trials, aligned neural activity to first tone
    corr_cell_trials: 2D array, neurons x trials, 1 if the neuron is correlated with the behavior in this trial, 0 if not
    trials: 1D array, trial numbers of trials to be analyzed
    all_trials: 1D array, all trial numbers
    output:
    Cdec_corr_trials: 2D array, time x trials, average neural activity of correlated neurons in each trial
    """
    Cdec_corr_trials = np.zeros((cdec.shape[1]-1, len(trials)))
    
    f=9
    for j,i in enumerate(trials):
        if i>all_trials[-1]:
            break
        else:
            this_corr = corr_cell_trials[:,i]==1
            this_cdec = np.nanmean(cdec[this_corr,:,i],axis=0)
            Cdec_corr_trials[:,j] = smooth(this_cdec, f, 'hamming')
        
    return Cdec_corr_trials



def get_CV_classifiers_ens(cdec, cdec_shu, cells, ens_corr_trials, sc_trials, lc_trials, zeros, mean_epoch, condition):
    """
    input:
    cdec: 3D array, neurons x time x trials, aligned neural activity to first tone
    cdec_shu: 3D array, neurons x time x trials, aligned neural activity to first tone in shuffled data
    cells: 1D array, cell numbers of cells to be analyzed
    ens_corr_trials: 2D array, cell x trials, probability of each cell in each trial
    sc_trials: 1D array, trial numbers of short interval trials
    lc_trials: 1D array, trial numbers of long interval trials
    zeros: 1D array, frame number of first tone in short and long interval
    mean_epoch: int, number of frames in mean epoch
    condition: string, 'normal', 'shuffle', 'align'
    output:
    Dict_classifier: dictionary, keys are 'accu_svm', 'accu_svm_shuffle', 'p_value', values are accuracy of each classifier
    """
    
    dff_sc_red = mean_ens(cdec[cells,:,:], ens_corr_trials[cells,:], sc_trials, zeros[0], mean_epoch, condition)
    dff_lc_red =  mean_ens(cdec[cells,:,:], ens_corr_trials[cells,:], lc_trials, zeros[1], mean_epoch, condition)
    shu_dff_sc_red =  mean_ens(cdec_shu[cells,:,:], ens_corr_trials[cells,:], sc_trials, zeros[0], mean_epoch, condition)
    shu_dff_lc_red =  mean_ens(cdec_shu[cells,:,:], ens_corr_trials[cells,:], lc_trials, zeros[1], mean_epoch, condition)

    dff_all_red = np.hstack((dff_sc_red, dff_lc_red))
    if np.isnan(dff_all_red).any():
        dff_all_red = np.nan_to_num(dff_all_red)

    shu_dff_all_red = np.hstack((shu_dff_sc_red, shu_dff_lc_red))
    if np.isnan(shu_dff_all_red).any():
        shu_dff_all_red = np.nan_to_num(shu_dff_all_red)

    type_all = np.hstack((np.zeros(len(sc_trials)), np.ones(len(lc_trials))))
    X, y, X_shu = shuffle(dff_all_red.T, type_all, shu_dff_all_red.T)

    accu_svm, accu_svm_shuffle, accu_LR, accu_LR_shuffle = classifier(X, y, X_shu, len(cells))
    t_stats_svm, p_value_svm = stats.ttest_ind(accu_svm_shuffle, accu_svm)
    t_stats_LR, p_value_LR = stats.ttest_ind(accu_LR_shuffle, accu_LR)
    
    Dict_classifier = {'CV_svm' : accu_svm,
                      'CV_svm_shu' : accu_svm_shuffle,
                      'p_value_svm' : p_value_svm}

    return Dict_classifier


def mean_ens(cdec, corr_cell_trials, trials, zero, epoch, condition):
    """
    input:
    cdec: 3D array, neurons x time x trials, aligned neural activity to first tone
    corr_cell_trials: 2D array, neurons x trials, 1 if the neuron is correlated with the behavior in this trial, 0 if not
    trials: 1D array, trial numbers of trials to be analyzed
    zero: int, frame number of first tone
    epoch: int, number of frames in mean epoch
    condition: string, 'normal', 'shuffle', 'align'
    output:
    Cdec_corr_trials: 2D array, time x trials, average neural activity of correlated neurons in each trial
    """
    Cdec_corr_trials = np.zeros((epoch+1, len(trials)))

    for j,i in enumerate(trials):
        this_corr = corr_cell_trials[:,i]==1
        this_cdec = cdec[this_corr,zero-1:zero+epoch,i]
        if condition == 'normal':
            Cdec_corr_trials[:,j] = np.nanmean(this_cdec, axis=0)
        elif condition == 'shuffle':
            shu_cdec = this_cdec
            shu_neu = shuffle(np.arange(this_cdec.shape[0]))
            shu_cdec = shu_cdec[shu_neu,:]
            Cdec_corr_trials[:,j] = np.nanmean(shu_cdec, axis=0)
        elif condition == 'align':
            shu_this_cdec = np.zeros_like(this_cdec)
            for n in range(this_cdec.shape[0]):
                shu_this_cdec[n,:] = shuffle(this_cdec[n,:])
            Cdec_corr_trials[:,j] = np.nanmean(shu_this_cdec, axis=0)
    return Cdec_corr_trials

