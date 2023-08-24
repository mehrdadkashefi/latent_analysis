import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pynwb as nwb
from pynwb import NWBHDF5IO
import pandas as pd
from scipy.signal import gaussian, convolve
import multiprocess
from functools import partial

######### Read data #########

def resample(Data, timestamps):
    temp = pd.DataFrame({('Data', 'x'):Data})
    temp['timestamps'] = pd.to_datetime(timestamps, unit='s')
    temp.set_index('timestamps', inplace=True)
    temp_resampled = temp.resample('ms').mean()
    temp_resampled['Data'] = temp_resampled['Data'].interpolate()
    return temp_resampled['Data'].to_numpy()


def read_mc_maze(data_path):
    data = NWBHDF5IO(data_path, "r").read()
    D = pd.DataFrame({
                    ('cursor_pos', 'x'):resample(data.processing['behavior']['cursor_pos'].data[:,0], data.processing['behavior']['cursor_pos'].timestamps[:]).reshape(-1, ),
                    ('cursor_pos', 'y'):resample(data.processing['behavior']['cursor_pos'].data[:,1], data.processing['behavior']['cursor_pos'].timestamps[:]).reshape(-1, ),
                    ('eye_pos', 'x'):resample(data.processing['behavior']['eye_pos'].data[:,0], data.processing['behavior']['eye_pos'].timestamps[:]).reshape(-1, ),
                    ('eye_pos', 'y'):resample(data.processing['behavior']['eye_pos'].data[:,1], data.processing['behavior']['eye_pos'].timestamps[:]).reshape(-1, ),
                    ('hand_pos', 'x'):resample(data.processing['behavior']['hand_pos'].data[:,0], data.processing['behavior']['hand_pos'].timestamps[:]).reshape(-1, ),
                    ('hand_pos', 'y'):resample(data.processing['behavior']['hand_pos'].data[:,1], data.processing['behavior']['hand_pos'].timestamps[:]).reshape(-1, ),
                    ('hand_vel', 'x'):resample(data.processing['behavior']['hand_vel'].data[:,0], data.processing['behavior']['hand_vel'].timestamps[:]).reshape(-1, ),
                    ('hand_vel', 'y'):resample(data.processing['behavior']['hand_vel'].data[:,1], data.processing['behavior']['hand_vel'].timestamps[:]).reshape(-1, ),
                        })
    trial_info = data.trials.to_dataframe()
    units = data.units.to_dataframe()
    units = units.drop(labels='obs_intervals', axis=1)
    units['location'] = ['PMd' if str(unit)[0] == '1' else 'M1' for unit in units.axes[0]]
    units = units.drop(labels='electrodes', axis=1)

    trial_info = trial_info[trial_info['stop_time'] <= len(D)/1000]
    
    return D, trial_info, units

data_path = "./000128/sub-Jenkins/sub-Jenkins_ses-full_desc-train_behavior+ecephys.nwb"
D, trial_info, units = read_mc_maze(data_path)


######### Tools for data analysis #########
class Analysis_tools():
    """ Class for tools often required for data analysis
    """
    def __init__(self, fs):
        self.fs = fs

    def get_stem(self, units, which_unit, t,  **kwargs):
        """ Get spike times for a specific unit, during a specific time
        """
        self.dt = kwargs.get('dt', 1/1000)   # Step size 
        self.t_pad = kwargs.get('t_pad', 0.5)   # How many seconds to pad the data from before and after the trial 
        plot = kwargs.get('plot', False)   # Plot the single trial
        return_padded = kwargs.get('return_padded', False)   # Whether to returned  

        #!!!! Fix this!
        t_low = max(0.0, t[0] - self.t_pad)
        t_high = t[1] + self.t_pad #min(t[1] + self.t_pad, len(D)/self.fs/60)
        t_trial = int(np.round((t_high - t_low)/self.dt))
        spike_times = units.loc[which_unit,:]['spike_times']
        in_range_inx = (spike_times <= t_high) & (spike_times >= t_low)
        in_range_time = spike_times[in_range_inx]
        in_range_time_idx = np.round((in_range_time - t_low) / self.dt).astype('int')

        # Temp zero for continuous rate
        stem = np.zeros((t_trial, ), dtype='bool')
        in_range_time_idx[in_range_time_idx >= len(stem)] = len(stem) - 1

        stem[in_range_time_idx] = 1

        if return_padded == 0:
            num_pad = np.round(self.t_pad/ self.dt).astype('int')
            stem = stem[num_pad:-num_pad]

        # Plot single trial
        if plot:
            plt.plot(stem, c='k')
            plt.xlabel('Time (ms)')
            plt.gca().set_yticks([])

        return stem
    
    def get_fr(self, which_unit, units, t, **kwargs):
        """ Get firing rate for a specific unit, during a specific time
        """
        self.gauss_width = kwargs.get('gauss_width', 50)   # Window size 50 ms
        self.dt = kwargs.get('dt', 1/1000)   # Step size 
        self.t_pad = kwargs.get('t_pad', 0.5)   # How many seconds to pad the data from before and after the trial 
        plot = kwargs.get('plot', False)   # Plot the single trial
        return_padded = kwargs.get('return_padded', False)   # Whether to returned  

        # Get the stem first
        stem = self.get_stem(units, which_unit, t, plot=False, dt=self.dt, t_pad=self.t_pad, return_padded=True)
        # Smooth the stem with Gaussian window
        # Compute Gauss window and std with respect to bins
        bin_width = self.dt * self.fs

        gauss_bin_std = self.gauss_width / bin_width
        # the window extends 3 x std in either direction
        win_len = int(6 * gauss_bin_std)
        # Create Gaussian kernel
        window = gaussian(win_len, gauss_bin_std, sym=True)
        window /=  np.sum(window)
        FR = convolve(stem, window, 'same') / self.dt

        if return_padded == 0:
            num_pad = np.round(self.t_pad/ self.dt).astype('int')
            FR = FR[num_pad:-num_pad]

        if plot:
            plt.plot(FR, c='k')
            plt.xlabel('Time (ms)')
            plt.ylabel('Firing rate (Spk/s)')

        return FR
    
    def align_continuous(self, trial_info, D, condition_columns, align_column, channel_column, t_range):
        """ Aligns continuous data on a specific event 
        """
        conds = trial_info.set_index(condition_columns).index.unique().tolist()
        data_aligned = np.zeros((len(conds),t_range[1] - t_range[0], len(D[channel_column].columns)))
        for cond_i, cond in enumerate(conds):
            # Make a mask to select desired trials
            mask = np.all(trial_info[condition_columns] == cond, axis=1)
            mask_idx = np.where(mask)[0]
            # Get start time, end time and event time for each condition
            # Start and end time in absolute time 
            start_times = trial_info.loc[mask_idx].start_time.to_numpy()
            end_times = trial_info.loc[mask_idx].stop_time.to_numpy()
            # event code time reletive to trial start
            event_times = trial_info.loc[mask_idx][align_column].to_numpy() - start_times
            # Allocate space for trials of the condition
            data_temp = np.zeros((len(start_times) ,t_range[1] - t_range[0], len(D[channel_column].columns)))
            # Loop over trials of the same conditioin
            for i, (ts, tev, te) in enumerate(zip(start_times, event_times, end_times)):
                temp = D[channel_column].iloc[int(ts*self.fs):int(te*self.fs), :].to_numpy()
                t_event = int(tev*self.fs)
                data_temp[i, :, :] = temp[t_event+t_range[0]:t_event+t_range[1], :]
            # Get the mean over trials of condition
            data_aligned[cond_i, :, :] = np.mean(data_temp, axis=0)
        return data_aligned
     
    def align_fr(self, trial_info, units, condition_columns, align_column, unit, t_range):
        """ Aligns units firing rate on a specific event 
        """
        conds = trial_info.set_index(condition_columns).index.unique().tolist()
        data_aligned = np.zeros((len(conds),t_range[1] - t_range[0], len(unit)))

        #for cond_i, cond in enumerate(conds):
        def par_fuction_over_cond(cond):
            # Make a mask to select desired trials
            mask = np.all(trial_info[condition_columns] == cond, axis=1)
            mask_idx = np.where(mask)[0]
            # event code time reletive to trial start
            event_times = trial_info.loc[mask_idx][align_column].to_numpy()
            # Allocate space for trials of the condition
            data_temp = np.zeros((len(event_times) ,t_range[1] - t_range[0], len(unit)))
            
            for u_i, u in enumerate(unit):
                # Loop over trials of the same conditioin
                for i, tev in enumerate(event_times):
                    data_temp[i, :, u_i] = self.get_fr(u, units, [tev + (t_range[0]/self.fs) , tev + (t_range[1]/self.fs)], plot=False)

            return np.expand_dims(np.mean(data_temp, axis=0), axis=0)        
        
        with multiprocess.Pool(processes=10) as pool:
            data_aligned = pool.map(par_fuction_over_cond, conds)
        return np.vstack(data_aligned)
    

######### Tools for dimensionality reduction #########

