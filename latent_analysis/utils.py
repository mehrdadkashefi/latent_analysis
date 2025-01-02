"""
Collection of different functions and tools for reading data from different tasks and tools for data analysis

@Author: Mehrdad Kashefi
"""

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
from scipy.io import loadmat


def resample(Data, timestamps):
    """ Resamples using a linear interpolation 
    Args:
        Data (np.array)
            Data to be resampled
        timestamps (np.array)
            Timestamps for the data
    Returns:
        resampled_data (np.array)
    """
    temp = pd.DataFrame({('Data', 'x'):Data})
    temp['timestamps'] = pd.to_datetime(timestamps, unit='s')
    temp.set_index('timestamps', inplace=True)
    temp_resampled = temp.resample('ms').mean()
    temp_resampled['Data'] = temp_resampled['Data'].interpolate()
    return temp_resampled['Data'].to_numpy()


def read_point2point(data_path, **kwargs):
    """ Read the data from point2point task 
    Args:
        data_path (str)
            Path to the data set 
    Returns:
        D (df)
            Data frame of the continuous data (Hand pos, Hand vel, etc.)
        trial_info (df)
            Data frame of the trial information (Start time, end time, etc.)
        units (df)
            Data frame of the units (Spike times, KS label, etc.)
    """
    KSGood_only = kwargs.get('KSGood_only', False)   # Radius of the target
    KSVersion = kwargs.get('KSVersion', 4)   # Which version of Kilosort to use

    trial_info = pd.read_csv(os.path.join(data_path, "trial_info.csv"))
    # rename last event
    if 'last_event' in trial_info.keys():
        trial_info['last_event'] = trial_info['last_event'].replace(1, 'END_TRIAL')
        trial_info['last_event'] = trial_info['last_event'].replace(2, 'TIMEOUT')
        trial_info['last_event'] = trial_info['last_event'].replace(3, 'BAD_DWELL')
        trial_info['last_event'] = trial_info['last_event'].replace(4, 'EARLY_GO')

    D = loadmat(os.path.join(data_path, "D.mat"))['D']
    D = pd.DataFrame({
        ('hand_pos', 'x'):D[:,0],
        ('hand_pos', 'y'):D[:,1],
        ('hand_vel', 'x'):D[:,2],
        ('hand_vel', 'y'):D[:,3], 
        ('hand_spd', 'xy'):D[:, 4]
        })
    
    if KSVersion == 4:
        units = pd.read_pickle(os.path.join(data_path,  "units_ks40.pkl"))
    elif KSVersion == 2:
        units = pd.read_pickle(os.path.join(data_path,  "units_ks20.pkl"))
    if KSGood_only:
        units = units.loc[units.KSLabel == 'good', :]

    return D, trial_info, units

def read_mc_maze(data_path):
    """ Read the data from the NHP maze task 
    Args:
        data_path (str)
            Path to the data set 
    Returns:
        D (df)
            Data frame of the continuous data (Hand pos, Hand vel, etc.)
        trial_info (df)
            Data frame of the trial information (Start time, end time, etc.)
        units (df)
            Data frame of the units (Spike times, KS label, etc.)
    """
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

def read_nhp_sequence(data_path, **kwargs):
    """ Read the data from the NHP sequence task 
    Args:
        data_path (str)
            Path to the data set 
    Returns:
        D (df)
            Data frame of the continuous data (Hand pos, Hand vel, etc.)
        trial_info (df)
            Data frame of the trial information (Start time, end time, etc.)
        units (df)
            Data frame of the units (Spike times, KS label, etc.)
    """
    KSGood_only = kwargs.get('KSGood_only', False)   # Radius of the target
    KSVersion = kwargs.get('KSVersion', 4)   # Which version of Kilosort to use
    trial_info = pd.read_csv(data_path + "trial_info.csv")

    # check if file exists 
    if os.path.isfile(data_path + "units_ks40.csv") or os.path.isfile(data_path + "units_ks20.csv"):
        if KSVersion == 4:
            units = pd.read_pickle(data_path + "units_ks40.pkl")
        elif KSVersion == 2:
            units = pd.read_pickle(data_path + "units_ks20.pkl")

        if KSGood_only:
            units = units.loc[units.KSLabel == 'good', :]
    else:
        print('Found no units_ksxx.pkl file!')
        print('Check your dataset directory if you were expecting neural data!')
        units = []

    temp = loadmat(data_path + "D.mat")['D']
    D = pd.DataFrame({
    ('hand_pos', 'x'):temp[:,0],
    ('hand_pos', 'y'):temp[:,1],
    ('hand_vel', 'x'):temp[:,2],
    ('hand_vel', 'y'):temp[:,3], 
    ('hand_spd', 'xy'):temp[:, 4]
    })
    # rename last event
    if 'last_event' in trial_info.keys():
        trial_info['last_event'] = trial_info['last_event'].replace(1, 'END_TRIAL')
        trial_info['last_event'] = trial_info['last_event'].replace(2, 'TIMEOUT')
        trial_info['last_event'] = trial_info['last_event'].replace(3, 'BAD_DWELL')
        trial_info['last_event'] = trial_info['last_event'].replace(4, 'EARLY_GO')
        trial_info['last_event'] = trial_info['last_event'].replace(4, 'BAD_TARGET')

    return D, trial_info, units

def read_mc_rtt(data_path):
    """ Read the data from the NHP random target reaching task
    Args:
        data_path (str)
            Path to the data set 
    Returns:
        D (df)
            Data frame of the continuous data (Hand pos, Hand vel, etc.)
        trial_info (df)
            Data frame of the trial information (Start time, end time, etc.)
        units (df)
            Data frame of the units (Spike times, KS label, etc.)
    """
    data = NWBHDF5IO(data_path, "r").read()
    D = pd.DataFrame({
                        ('cursor_pos', 'x'):data.processing['behavior']['cursor_pos'].data[:,0].reshape(-1, ),
                        ('cursor_pos', 'y'):data.processing['behavior']['cursor_pos'].data[:,1].reshape(-1, ),
                        ('finger_pos', 'x'):data.processing['behavior']['finger_pos'].data[:,0].reshape(-1, ),
                        ('finger_pos', 'y'):data.processing['behavior']['finger_pos'].data[:,1].reshape(-1, ),
                        ('finger_vel', 'x'):data.processing['behavior']['finger_vel'].data[:,0].reshape(-1, ),
                        ('finger_vel', 'y'):data.processing['behavior']['finger_vel'].data[:,1].reshape(-1, ),
                        ('target_pos', 'x'):data.processing['behavior']['target_pos'].data[:,0].reshape(-1, ),
                        ('target_pos', 'y'):data.processing['behavior']['target_pos'].data[:,1].reshape(-1, ),
                            })
   
    # Defime trial by target update
    t_pos = np.vstack((D['target_pos']['x'], D['target_pos']['y'])).T
    event = np.where(np.sum(np.abs(np.diff(t_pos,axis=0)), axis=1)>5)[0]

    trial_info = pd.DataFrame({
                        ('start_time','time'):np.append(np.zeros(1,), event[:-1])/1000,
                        ('end_time','time'):event/1000,
                        ('to_target', 'x'):D['target_pos'].iloc[event]['x'],
                        ('to_target', 'y'):D['target_pos'].iloc[event]['y'],
                            })
    
    unique_target_id, unique_target_idx = np.unique(np.vstack((D['target_pos'].iloc[event]['x'].values, D['target_pos'].iloc[event]['y'].values) ), axis=1, return_inverse=True)
    trial_info[('to_target', 'id')] = unique_target_idx
    
    trial_info[('from_target', 'x')] = D['target_pos'].iloc[np.append(event[0], event[:-1])]['x'].values
    trial_info[('from_target', 'y')] = D['target_pos'].iloc[np.append(event[0], event[:-1])]['y'].values
    trial_info[('from_target', 'id')] = np.append(unique_target_idx[0], unique_target_idx[:-1])
    # Add the reach extent
    dist = np.vstack((trial_info['to_target']['x'].values, trial_info['to_target']['y'].values)) 
    - np.vstack((trial_info['from_target']['x'].values, trial_info['from_target']['y'].values))
    trial_info[('reach_extent', ' ')] = np.linalg.norm(dist, axis=0)
    # Remove the first trial: bad from traget
    trial_info = trial_info.drop(index=trial_info.index[0], axis=0)

    units = data.units.to_dataframe()
    units = units.drop(labels='obs_intervals', axis=1)
    units['location'] = [units.loc[u].electrodes.iloc[0].location for u in units.index]
    units = units.drop(labels='electrodes', axis=1)
    return D, trial_info, units
    

######### Tools for data analysis #########
class Analysis_tools():
    """ 
    Class for tools often required for data analysis
    """
    def __init__(self, fs):
        self.fs = fs

    def get_stem(self, units, which_unit, t,  **kwargs):
        """ Get spike times for a specific unit, in a specific time range
        Args:
            units (df)
                Data frame of the units (Spike times, KS label, etc.)
            which_unit (int)
                Which unit to get the spike times
            t (list)
                Time range to get the spike times
        Kwargs:
            df (float)
                Step size, default is 1/1000
            t_pad (float)
                How many seconds to pad the data from before and after the trial, default is 0.5
            plot (bool)
                Plot the single trial, default is False
            return_padded (bool)
                Whether to include padded data in return, default is False
        Returns:
            stem (np.array)
                Spike times for the specific unit, in the specific time range
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
        Args:
            which_unit (int)
                Which unit to get the spike times
            units (df)
                Data frame of the units (Spike times, KS label, etc.)
            t (list)
                Time range to get the spike times
        Kwargs:
            gauss_width (int)
                Window size for the Gaussian window, default is 50
            dt (float)
                Step size, default is 1/1000
            t_pad (float)
                How many seconds to pad the data from before and after the trial, default is 0.5
            plot (bool)
                Plot the single trial, default is False
            return_padded (bool)
                Whether to include padded data in return, default is False
        Returns:
            FR (np.array)
                Continuous firing rate for the specific unit, in the specific time range
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
    
    def align_continuous(self, trial_info, D, condition_columns, align_column, channel_column, t_range, return_mean = True):
        """ Aligns continuous data on a specific event 
        Args:
            trial_info (df)
                Data frame of the trial information (Start time, end time, etc.)
            D (df)
                Data frame of the continuous data (Hand pos, Hand vel, etc.)
            condition_columns (list)
                Columns to group the data
            align_column (str)
                Column to align the data on
            channel_column (str)
                Which data column is selected
            t_range (list)
                Time range to align the data
            return_mean (bool)
                Whether to return the mean of trials of a condition, default is True
        Returns:
            data_aligned (np.array)
                Aligned data (condition x time x channel)
            conds (np.array)
                Unique conditions
            aligned_trials_type (np.array)
                Type of the trials
        """
        conds = trial_info.set_index(condition_columns).index.unique().tolist()
        data_aligned = np.zeros((len(conds),t_range[1] - t_range[0], len(D[channel_column].columns)))
        data_aligned_trials = []
        aligned_trials_type = []
        for cond_i, cond in enumerate(conds):
            # Make a mask to select desired trials
            mask = np.all(trial_info[condition_columns] == cond, axis=1)
            mask_idx = np.where(mask)[0]
            # Get start time, end time and event time for each condition
            # Start and end time in absolute time 
            start_times = trial_info.iloc[mask_idx].start_time.to_numpy()
            end_times = trial_info.iloc[mask_idx].stop_time.to_numpy()
            # event code time reletive to trial start
            event_times = trial_info.iloc[mask_idx][align_column].to_numpy() - start_times
            # Allocate space for trials of the condition
            data_temp = np.zeros((len(start_times) ,t_range[1] - t_range[0], len(D[channel_column].columns)))
            # Loop over trials of the same conditioin
            for i, (ts, tev, te) in enumerate(zip(start_times, event_times, end_times)):
                temp = D[channel_column].iloc[int(ts*self.fs):int(te*self.fs), :].to_numpy()
                t_event = int(tev*self.fs)
                data_temp[i, :, :] = temp[t_event+t_range[0]:t_event+t_range[1], :]
            # Get the mean over trials of condition
            data_aligned[cond_i, :, :] = np.mean(data_temp, axis=0)
            aligned_trials_type.append(cond_i * np.ones((len(mask_idx))))
            if return_mean == False:
                data_aligned_trials.append(data_temp)
        # Return        
        if return_mean == True:
            return data_aligned, np.array(conds), np.hstack(aligned_trials_type).astype(int)
        else:
            return np.vstack(data_aligned_trials), np.array(conds), np.hstack(aligned_trials_type).astype(int)
     
    def align_fr(self, trial_info, units, condition_columns, align_column, unit, t_range, return_mean = True):
        """ Aligns units firing rate on a specific event 
        Args:
            trial_info (df)
                Data frame of the trial information (Start time, end time, etc.)
            units (df)
                Data frame of the units (Spike times, KS label, etc.)
            condition_columns (list)
                Columns to group the data
            align_column (str)
                Column to align the data on
            unit (list)
                Which unit to get the spike times
            t_range (list)
                Time range to align the data
            return_mean (bool)
                Whether to return the mean of trials of a condition, default is True
        Returns: 
            data_aligned (np.array)
                Aligned data (condition x time x unit)
        """
        conds = trial_info.set_index(condition_columns).index.unique().tolist()
        data_aligned = np.zeros((len(conds),t_range[1] - t_range[0], len(unit)))

        #for cond_i, cond in enumerate(conds):
        def par_fuction_over_cond(cond):
            # Make a mask to select desired trials
            mask = np.all(trial_info[condition_columns] == cond, axis=1)
            mask_idx = np.where(mask)[0]
            # event code time reletive to trial start
            event_times = trial_info.iloc[mask_idx][align_column].to_numpy()
            # Allocate space for trials of the condition
            data_temp = np.zeros((len(event_times) ,t_range[1] - t_range[0], len(unit)))
            
            for u_i, u in enumerate(unit):
                # Loop over trials of the same conditioin
                for i, tev in enumerate(event_times):
                    data_temp[i, :, u_i] = self.get_fr(u, units, [tev + (t_range[0]/self.fs) , tev + (t_range[1]/self.fs)], plot=False)
            if return_mean:
                return np.expand_dims(np.mean(data_temp, axis=0), axis=0)
            else:
                return data_temp
        
        with multiprocess.Pool(processes=10) as pool:
            data_aligned = pool.map(par_fuction_over_cond, conds)
            
        return np.vstack(data_aligned)
    
    def align_stem(self, trial_info, units, condition_columns, align_column, unit, t_range):
        """ Aligns units firing rate on a specific event 
        Args:
            trial_info (df)
                Data frame of the trial information (Start time, end time, etc.)
            units (df)
                Data frame of the units (Spike times, KS label, etc.)
            condition_columns (list)
                Columns to group the data
            align_column (str)
                Column to align the data on
            unit (list)
                Which unit to get the spike times
            t_range (list)
                Time range to align the data
        Returns:
            data_aligned (np.array)
                Aligned data (condition x time x unit)
        """
        conds = trial_info.set_index(condition_columns).index.unique().tolist()
        data_aligned = np.zeros((len(conds),t_range[1] - t_range[0], len(unit)))

        #for cond_i, cond in enumerate(conds):
        def par_fuction_over_cond(cond):
            # Make a mask to select desired trials
            mask = np.all(trial_info[condition_columns] == cond, axis=1)
            mask_idx = np.where(mask)[0]
            # event code time reletive to trial start
            event_times = trial_info.iloc[mask_idx][align_column].to_numpy()
            # Allocate space for trials of the condition
            data_temp = np.zeros((len(event_times) ,t_range[1] - t_range[0], len(unit)))
            
            for u_i, u in enumerate(unit):
                # Loop over trials of the same conditioin
                for i, tev in enumerate(event_times):
                    data_temp[i, :, u_i] = self.get_stem(units, u, [tev + (t_range[0]/self.fs) , tev + (t_range[1]/self.fs)], plot=False, return_padded=False)
            else:
                return data_temp
        
        with multiprocess.Pool(processes=10) as pool:
            data_aligned = pool.map(par_fuction_over_cond, conds)
            
        return np.vstack(data_aligned)
    
class Kernel_Gaussian():
    def __init__(self,x_range, y_range, n_kernel, S, do_plot):
        """ Class for generaing 2D Gaussian Kernels
        Args:
            x_range (list)
                Range of x values
            y_range (list)
                Range of y values
            n_kernel (int)
                Number of kernels
            S (np.array)
                Covariance matrix for each Gaussian kernel (2x2)
            do_plot (bool)
                Whether to plot the kernels
        """
        self.x_range = x_range
        self.y_range = y_range
        self.n_kernel = n_kernel
        self.S = S
        self.do_plot = do_plot

        Kernel_x = np.linspace(min(self.x_range)+((max(self.x_range)-min(self.x_range))/(self.n_kernel+1)),max(self.x_range)-((max(self.x_range)-min(self.x_range))/(self.n_kernel+1)), self.n_kernel)
        Kernel_y = np.linspace(min(self.y_range)+((max(self.y_range)-min(self.y_range))/(self.n_kernel+1)),max(self.y_range)-((max(self.y_range)-min(self.y_range))/(self.n_kernel+1)), self.n_kernel)
        [m,n] = np.meshgrid(Kernel_x,Kernel_y)
        Kernel_mu = np.stack((m.flatten(),n.flatten()), axis=0).T

        self.centers = Kernel_mu

        [X,Y] = np.meshgrid(np.arange(min(self.x_range-1),max(self.x_range)+1, 0.3), np.arange(min(self.y_range-1), max(self.y_range)+1, 0.3))

        ZZ = np.zeros(X.shape)

        for K in range(len(Kernel_mu)):
            mu = Kernel_mu[K,:]
            Gaussian_pdf = lambda x,y : np.linalg.det(2*np.pi*self.S) * np.exp(-0.5 * np.sum(((np.concatenate((x,y),axis=-1)-mu).T) * (np.linalg.pinv(self.S)@(np.concatenate((x,y), axis=-1)-mu).T), axis=0))
            Z = Gaussian_pdf(X.reshape(-1,1), Y.reshape(-1,1))
            Z = Z.reshape(X.shape)
            ZZ = ZZ + Z

        if self.do_plot:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(X, Y, ZZ, cmap=sns.color_palette("viridis", as_cmap=True))

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Feature Value')

    def encode(self,x,y):
        """ Encode the data using the Gaussian Kernels
        Args:
            x (np.array)
                X values
            y (np.array)
                Y values
        Returns:
            M (np.array)
                Encoded data
        """
        M = np.zeros((x.shape[0], len(self.centers)))
        for K in range(len(self.centers)):
            mu = self.centers[K,:]
            Gaussian_pdf =  lambda x,y : np.linalg.det(2*np.pi*self.S) * np.exp(-0.5 * np.sum(((np.concatenate((x,y),axis=-1)-mu).T) * (np.linalg.pinv(self.S)@(np.concatenate((x,y), axis=-1)-mu).T), axis=0))
            M[:, K] = Gaussian_pdf(x,y)
        return M

class Kernel_Cosine():
    def __init__(self, n_cos, clip_neg):
        """ Class for generating Cosine Kernels
        Args:
            n_cos (int)
                Number of cosine kernels
            clip_neg (bool)
                Whether to clip the negative values
        """
        self.n_cos = n_cos
        self.phase_steps = np.linspace((2*np.pi)/n_cos, 2*np.pi, n_cos)
        self.clip_neg = clip_neg

        # Plot Kernels
        a = np.linspace(0, 2*np.pi, 1000)
        y = np.zeros((len(a), len(self.phase_steps)))

        for count, ang in enumerate(self.phase_steps):
            y[:, count] = np.cos(a - ang)
        if self.clip_neg:
            y[y<0] = 0
        plt.plot(np.rad2deg(a),y)
        plt.xlabel('Angle (deg)')

    def encode(self, A):
        """ Encode the data using the Cosine Kernels
        Args:
            A (np.array)
                Data to be encoded
        Returns:
            M (np.array)
                Encoded data
        """
        M = np.zeros((A.shape[0], len(self.phase_steps)))
        for count, ang in enumerate(self.phase_steps):
            M[:, count] = np.cos(A - ang)
        if self.clip_neg:
            M[M<0] = 0
        if M.shape[1] != np.linalg.matrix_rank(M):
            print("M is not Full Rank!")
            print('Columns of M: ', M.shape[1])
            print('Rank of M: ', np.linalg.matrix_rank(M))

        return M