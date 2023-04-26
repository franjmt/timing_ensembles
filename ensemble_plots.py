import numpy as np

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.colors import ListedColormap
from p_value_star import p_value_star
from scipy import stats


trial_colors = {'sc_trials':'#EE82EE',
                'si_trials':'#DAA6DB',
                'lc_trials':'#FAC205',
                'li_trials':'#E0C666'}
dat_colors = ['#5478C0', '#73AEEB']


def plot_trial_ensemble_matrix(ens_interval, ens_delay, ens_resp, mean_interval_epoch, mean_servo_epoch,
                               x_axis, y_axis, colors_ens_list):


    fig, ax = plt.subplots(figsize=(10,7))
    dat = np.concatenate((ens_interval, ens_delay, ens_resp), axis=0)
    
    this_ens_num = np.unique(dat.astype(int))
    print(this_ens_num)
    
    this_colors = [colors_ens_list[i] for i in this_ens_num]

    ens_cmap = ListedColormap(this_colors)
    
    pos = ax.imshow(dat.T, cmap=ens_cmap)
    ax = plt.gca();

    # Major ticks
    ax.set_yticks(np.arange(0, dat.T.shape[0], 1))
    ax.set_xticks(np.arange(0, dat.T.shape[1], 1))

    # Labels for major ticks
    ax.set_xticklabels(x_axis, rotation=50)
    ax.set_yticklabels(y_axis[:dat.T.shape[0]])

    # Minor ticks
    ax.set_yticks(np.arange(-.5, dat.T.shape[0], 1), minor=True)
    ax.set_xticks(np.arange(-.5, dat.T.shape[1], 1), minor=True)

    ax.grid(which='minor', color='k', linestyle='-', linewidth=0.1)
    ax.vlines(mean_interval_epoch, -0.5, dat.T.shape[0]-1+0.5, colors = 'k')
    ax.vlines(mean_interval_epoch+mean_servo_epoch+1, -0.5, dat.T.shape[0]-1+0.5, colors = 'k')
    # ax.set_title(title+'\nensembles', fontsize=10)
    ax.set_ylabel('trials', fontsize=10)

    fig.colorbar(pos, ax=ax, shrink=0.5, ticks=this_ens_num)
    plt.tight_layout()



def plot_prob_ens_epoch_trial(ens_sign_prob, ens_sign_prob_sem, ens_sigificant, p_values_df, 
                              epoch_label, colors_ens_list, title):
    
    figure = plt.figure(figsize=(5,4.2))
    plt.suptitle(title, y=1.05)
    
    p_value = 0.05
    barwidth = 0.25
    b1 = np.arange(len(epoch_label))*2
    b2 = [x + barwidth*(len(ens_sigificant)/2) for x in b1]
    
    for j, i in enumerate(ens_sigificant):
        plt.bar(b1, ens_sign_prob[j,:], yerr=ens_sign_prob_sem[j,:], 
                color=colors_ens_list[i], width=barwidth, 
                edgecolor='k', lw=1, label='ensemble '+ str(ens_sigificant[j]))

        
        if p_values_df['interval_delay'][i]<=p_value:
            p = p_value_star(p_values_df['interval_delay'][i])
            y_pos = max(ens_sign_prob[j,:])+max(ens_sign_prob_sem[j,:])+0.1
            plt.plot([b1[0], b1[1]], [y_pos, y_pos], lw=1, marker='|', color='k')
            plt.text(np.mean([b1[0], b1[1]]), y_pos+0.005, p,  fontsize=8, ha='center')    

        if p_values_df['interval_resp'][i]<=p_value:
            sig_str = 'p_value < '+str(p_value)
            p = p_value_star(p_values_df['interval_resp'][i])
            y_pos = max(ens_sign_prob[j,:])+max(ens_sign_prob_sem[j,:])+0.15
            plt.plot([b1[0], b1[2]], [y_pos, y_pos], lw=1, marker='|', color='k')
            plt.text(np.mean([b1[0], b1[2]]), y_pos+0.005, p,  fontsize=8, ha='center') 

        if p_values_df['delay_resp'][i]<=p_value:
            p = p_value_star(p_values_df['delay_resp'][i])
            y_pos = max(ens_sign_prob[j,:])+max(ens_sign_prob_sem[j,:])+0.2
            plt.plot([b1[1], b1[2]], [y_pos, y_pos], lw=1, marker='|', color='k')
            plt.text(np.mean([b1[1], b1[2]]), y_pos+0.005, p,  fontsize=8, ha='center') 
        b1 = [x + barwidth for x in b1]

    plt.ylabel('probability')
    lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=8)
    plt.xticks(b2, epoch_label)
    plt.tight_layout()



def classifier_box_plot_ens(labels, svm_data, svm_shu, svm_p_values):
    global dat_colors
    fig, ax = plt.subplots(1, figsize=(2.5*(len(labels)),3))

    data_labels = ['data', 'shuffled data']
    
    barwidth = 1
    b1 = np.arange(len(labels))*2
    b2 = [x + barwidth for x in b1]
    b3 = [x + barwidth/2 for x in b1]

    # plot SVM CV performances
    b = ax.boxplot(svm_data, patch_artist=True, positions=b1, medianprops=dict(color='black'), widths=0.5)
    for patch in b['boxes']:
        patch.set_facecolor(dat_colors[0]) 
        patch.set_alpha(0.8)

    c = ax.boxplot(svm_shu, patch_artist=True, positions=b2, medianprops=dict(color='black'), widths=0.5)
    for patch in c['boxes']:
        patch.set_facecolor(dat_colors[1]) 
        patch.set_alpha(0.8)
    ax.legend([b['boxes'][0], c['boxes'][0]], data_labels, 
                 bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=8)
 
    for i in range(len(svm_p_values)):
        p = p_value_star(svm_p_values[i])
        y_pos = ax.get_ylim()
        ax.plot([b1[i], b2[i]], [y_pos[1], y_pos[1]], lw=1, marker='|', color='k')
        ax.text(np.mean([b1[i], b2[i]]), y_pos[1]+0.005, p,  fontsize=8, ha='center')   

    ax.set_xticks(b3)
    ax.set_xticklabels(labels)
    ax.set_ylabel('CV performances (%)')
    ax.set_xlabel('Ensemble')
    ax.set_title('SVM', fontsize=9, y=1.05)
    plt.tight_layout(pad=3)



def plot_mean_activity_trials(xrange, zero, Cdec_int, data_type, mean_epoch_dict, ctypes):
    global trial_colors, dat_colors

    fig, ax = plt.subplots(2,1,figsize=(6,4))
    x_axis = np.arange((-1)*xrange[0],xrange[1]-xrange[0])*0.125
    
    # c_list = ['short', 'long']
    label_list = ['short interval', 'long interval']

    for i in range(len(ctypes)):
        ax[i].spines['left'].set_visible(False)
        tc = ctypes[i]
        dat_m = np.nanmean(Cdec_int[f'{data_type}_{tc}'][:,zero-xrange[0]-1:zero+xrange[1]-xrange[0]-1], axis=0)
        dat_s = stats.sem(Cdec_int[f'{data_type}_{tc}'][:,zero-xrange[0]-1:zero+xrange[1]-xrange[0]-1], axis=0, nan_policy='omit')
        this_max = np.nanmax([dat_m])+np.nanmax([dat_s])
        this_min = np.nanmin([dat_m])-np.nanmin([dat_s])
        
        ax[i].plot(x_axis, dat_m, color=trial_colors[f'{tc}_trials'], lw=2, label=label_list[i])
        ax[i].fill_between(x_axis, dat_m-dat_s, dat_m+dat_s, color=trial_colors[f'{tc}_trials'], alpha=0.2)
        ax[i].vlines(x_axis[xrange[0]], this_min, this_max, lw=1, label='1st tone', color='#D1D6D9')
        ax[i].vlines(x_axis[xrange[0]+mean_epoch_dict[f'mean_{tc}_interval']], this_min, this_max, lw=1,
                     color=dat_colors[0], label='2nd tone')
        ax[i].vlines(x_axis[xrange[0]+mean_epoch_dict[f'mean_{tc}_interval']+mean_epoch_dict[f'mean_{tc}_servo']+mean_epoch_dict[f'mean_{tc}_resp']], 
                     this_min, this_max, lw=1, color='green', label='response')
        lgd = ax[i].legend(fontsize=8, loc='upper left', ncol=1, bbox_to_anchor=(1, 1.01))
        ax[i].set_yticks([])
    ax[0].spines['bottom'].set_visible(False)
    ax[0].set_xticks([])
    ax[1].vlines(x_axis[0]-0.2, this_min, this_min+0.01, lw=2)
    ax[1].text(x_axis[0]-0.6, this_min, '0.01 z\u0394F/F', rotation=90, fontsize=8)
    ax[1].set_xlabel('time (s)')
    ax[1].tick_params(axis='x')
    ax[0].set_xlim(x_axis[0]-0.8, x_axis[-1]+0.1)
    ax[1].set_xlim(x_axis[0]-0.8, x_axis[-1]+0.1)
    plt.tight_layout()



def plot_mean_ensemble_activity(xrange, zero, Cdec_dict, data_type, mean_epoch_dict):
    global trial_colors, dat_colors

    fig, ax = plt.subplots(2,1,figsize=(4.8,4), sharex=True)
    x_axis = np.arange((-1)*xrange[0],xrange[1]-xrange[0])*0.125
    
    label_list = ['short interval\ncorrect', 
                  'long interval\ncorrect']
    ctypes = ['sc','lc']

    for i, tc in enumerate(ctypes):
        ax[i].spines['left'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
        ax[i].spines['top'].set_visible(False)
        dat_m = np.nanmean(Cdec_dict[f'{data_type}_{tc}'][zero-xrange[0]-1:zero+xrange[1]-xrange[0]-1,:], axis=1)
        dat_s = stats.sem(Cdec_dict[f'{data_type}_{tc}'][zero-xrange[0]-1:zero+xrange[1]-xrange[0]-1,:], axis=1, nan_policy='omit')

        ax[i].plot(x_axis, dat_m, color=trial_colors[f'{tc}_trials'], lw=2, label=label_list[i])
        ax[i].fill_between(x_axis, dat_m-dat_s, dat_m+dat_s, color=trial_colors[f'{tc}_trials'], alpha=0.2)
        this_min, this_max = ax[i].get_ylim()

        ax[i].vlines(0, this_min, this_max, lw=1, label='1st tone', color='k')
        ax[i].vlines(x_axis[xrange[0]+mean_epoch_dict[f'mean_{tc}_interval']], this_min, this_max, lw=1,
                     color=dat_colors[0], label='2nd tone')
        ax[i].vlines(x_axis[xrange[0]+mean_epoch_dict[f'mean_{tc}_interval']+mean_epoch_dict[f'mean_{tc}_servo']+mean_epoch_dict[f'mean_{tc}_resp']], 
                     this_min, this_max, lw=1, color='green', label='response')
        lgd = ax[i].legend(fontsize=8, loc='upper left', ncol=1, bbox_to_anchor=(1, 1.01))
        ax[i].set_yticks([])

    ax[0].spines['bottom'].set_visible(False)
    ax[1].vlines(x_axis[0]-0.1, this_min, this_min+0.1, lw=2, color='k')
    ax[1].text(x_axis[0]-0.4, this_min, '0.1 z\u0394F/F', rotation=90, fontsize=8)
    ax[1].set_xlabel('time (s)')
    ax[1].tick_params(axis='x')
    ax[1].set_xlim(x_axis[0]-0.1, x_axis[-1]+0.1)
    plt.tight_layout()