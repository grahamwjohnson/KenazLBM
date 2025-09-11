import os
import glob
import re
import pickle
import numpy as np
import pandas as pd
import time
import gc
import pyedflib
import scipy.signal
import datetime
import sys
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Use a relative path to ch_to_ignore.txt in the same directory as this script
CH_TO_IGNORE_PATH = os.path.join(os.path.dirname(__file__), 'ch_to_ignore.txt')

def create_bip_mont(channels: list[str], pat_id: str, ch_names_to_ignore: list, save_dir: str):
    bip_names = []
    mont_idxs = []

    # Delete unused channels, whole strings must match
    ch_idx_to_delete = []
    ch_names_found_to_delete = []
    for j in range(len(channels)):
        for i in range(len(ch_names_to_ignore)):
            if ch_names_to_ignore[i] == channels[j]:
                ch_idx_to_delete = ch_idx_to_delete + [j]
                ch_names_found_to_delete = ch_names_found_to_delete + [channels[j]]
                continue
    
    # TODO Should be sorting channel names now to deal with Edge cases for patients collected on NK 
    # where 2 channels are listed out of order, but this may actually introduce MORE errors than 
    # if we leave it where we assume channels are in order

    # Find all numbers at ends of channel labels
    nums = np.ones(len(channels), dtype=int)*-1
    for i in range(0,len(channels)):

        # Skip unused channels
        if i in ch_idx_to_delete: 
            continue

        str_curr = channels[i]
        still_number = True
        ch_idx = -1
        while still_number:
            curr_chunk = str_curr[ch_idx:]
            if not curr_chunk.isnumeric():
                nums[i] = str_curr[ch_idx+1:]
                still_number = False
            ch_idx = ch_idx - 1
    
    # Base the lead change on when numbers switch because this is more
    # robust to weird naming strategies that use numbers in base name
    for i in range(0,len(nums) - 1):
        if nums[i] + 1 == nums[i+1]:
            # Valid monotonically increasing bipolar pair
            bip_names.append(channels[i] + channels[i+1])
            mont_idxs.append([i,i+1])

    # Save a CSV to output directory with bip names and mont_idxs 
    if not os.path.exists(save_dir): os.mkdir(save_dir)
    df_bipmont = pd.DataFrame({'mont_idxs': mont_idxs, 
                       'bip_names': bip_names})
    df_bipmont.to_csv(save_dir + '/' + pat_id + '_bipolar_montage_names_and_indexes_from_rawEDF.csv')

    return mont_idxs, bip_names

def highpass(data: np.ndarray, cutoff: float, sample_rate: float, poles: int = 8):
    sos = scipy.signal.butter(poles, cutoff, 'highpass', fs=sample_rate, output='sos')
    filtered_data = scipy.signal.sosfiltfilt(sos, data)
    return filtered_data

def bandstop(data: np.ndarray, edges: list[float], sample_rate: float, poles: int = 8):
    sos = scipy.signal.butter(poles, edges, 'bandstop', fs=sample_rate, output='sos')
    filtered_data = scipy.signal.sosfiltfilt(sos, data)
    return filtered_data

def lowpass(data: np.ndarray, cutoff: float, sample_rate: float, poles: int = 8):
    sos = scipy.signal.butter(poles, cutoff, 'lowpass', fs=sample_rate, output='sos')
    filtered_data = scipy.signal.sosfiltfilt(sos, data)
    return filtered_data

def apply_wholeband_filter(y0, fs):

    # Hardcoded filter values, Hz - This is done before splitting into desired freq ranges
    FILT_HP = 1
    FILT_BS_RANGE_1 = [59, 61]
    FILT_BS_RANGE_2 = [119, 121]
    FILT_LP = 179
    
    y1 = highpass(y0, FILT_HP, fs)
    y2 = bandstop(y1, FILT_BS_RANGE_1, fs)
    y3 = bandstop(y2, FILT_BS_RANGE_2, fs)
    y4 = lowpass(y3, FILT_LP, fs)

    return y4

def read_channel(f, i):
    return f.readSignal(i)

def montage_filter_pickle_edfs(
    pat_id: str,
    edf_file: str,
    outfile: str,
    desired_samp_freq: int,
    expected_unit: str,
    montage: str,
    ch_names_to_ignore: list,
    ignore_channel_units: bool
):
    # CREATE BIPOLAR MONTAGE
    print(f"Processing EDF file: {edf_file}")

    # Use PyEDFLib to read in file
    f = pyedflib.EdfReader(edf_file)
    channels = f.getSignalLabels()
    mont_idxs, bip_names = create_bip_mont(channels, pat_id, ch_names_to_ignore, os.path.join(os.path.dirname(outfile), 'metadata'))
    print("Montage created")
    f._close()
    del f

    print("Reading in EDF")
    gc.collect()

    f = pyedflib.EdfReader(edf_file)
    n = f.signals_in_file
    channels = f.getSignalLabels()
    all_samp_freqs = np.array(f.getSampleFrequencies())

    # Ensure all channel units are the same
    if not ignore_channel_units:
        sig_headers = f.getSignalHeaders()
        ch_units = np.empty(len(channels), dtype=object)
        for i in range(len(channels)):
            ch_units[i] = sig_headers[i]['dimension']
        if not np.all(ch_units == ch_units[0]):
            raise Exception("Not all channel units are the same for file: " + edf_file)
        if ch_units[0] != expected_unit:
            raise Exception("Current file's channel units do not equal expected units: " + expected_unit + ". Current file: " + edf_file)

    start_t = time.time()
    raw_data = np.zeros((n, f.getNSamples()[0]), dtype=np.float16)
    for i in np.arange(n):
        raw_data[i, :] = f.readSignal(i)
    print(f"Time read EDF in: {time.time()-start_t}")
    f._close()
    del f
    print("EDF read in")

    # Ensure sampling freq are all equal to each other
    if not np.all(all_samp_freqs == all_samp_freqs[0]):
        raise Exception("Not all sampling freqs units are the same for file: " + edf_file)

    # Check sampling frequency is as desired, resample if not:
    fs = all_samp_freqs[0]
    if fs != desired_samp_freq:
        print(f"Resampling to {desired_samp_freq} for file: {edf_file}, this file was sampled at {fs} Hz")
        if fs % desired_samp_freq == 0:
            fs_mult = int(fs / desired_samp_freq)
            print(f"Sampling frequency of file [{fs}] is exact multiple of desired sampling frequency [{desired_samp_freq}], so simply indexing at interval of {fs_mult}")
            raw_data = raw_data[:, 0::fs_mult]
            fs = desired_samp_freq
        else:
            raise Exception(f"FS [{fs}] NOT MULTIPLE OF DESIRED SAMPLING FREQUENCY - NOT CODED UP YET")
    else:
        print(f"Sampling frequency confirmed to be {fs} Hz")

    # If doing bipolar montage
    if montage == 'BIPOLE':
        print("Assigning bipolar montage")
        new_bip_names = ["" for x in range(len(bip_names))]
        for i in range(len(bip_names)):
            new_bip_names[i] = channels[mont_idxs[i][0]] + channels[mont_idxs[i][1]]
        if new_bip_names != bip_names:
            print("WARNING: Current file's created bip montage does not exactly equal original file montage. Current file: " + edf_file)
            print("Assuming that channels are in proper order, just with wrong names (i.e. bad Natus template)")

        new_raw_data = np.empty([len(bip_names), raw_data.shape[1]], dtype=np.float16)
        for i in range(len(bip_names)):
            new_raw_data[i, :] = raw_data[mont_idxs[i][0], :] - raw_data[mont_idxs[i][1], :]
        raw_data = new_raw_data
        del new_raw_data
        gc.collect()

        channels = bip_names
        print("Bipolar montage assignment complete")

    print("Filtering the data")
    filt_data = np.asarray([apply_wholeband_filter(raw_data[i, :], fs) for i in range(len(channels))], dtype=np.float16)

    del raw_data
    gc.collect()

    # Save directly to the provided outfile path
    with open(outfile, "wb") as f:
        pickle.dump(filt_data, f)
    print(f"Big pickle saved")

    return len(channels)

def find_contiguous_true_indexes(array):
  """Finds the indices of all contiguous regions of True in an array.

  Args:
    array: A NumPy boolean array.

  Returns:
    A list of tuples, where each tuple contains the start and end indices of a
    contiguous region of True values.
  """

  # Find the indices of all True values in the array.
  true_indexes = np.nonzero(array)[0]

  # Find the start and end indices of each contiguous region of True values.
  start_indexes = []
  end_indexes = []
  for i in range(len(true_indexes)):
    if i == 0 or true_indexes[i] - true_indexes[i - 1] > 1:
      start_indexes.append(true_indexes[i])
    if i == len(true_indexes) - 1 or true_indexes[i + 1] - true_indexes[i] > 1:
      end_indexes.append(true_indexes[i])

  # Return a list of tuples containing the start and end indices of each
  # contiguous region of True values.
  return list(zip(start_indexes, end_indexes))

def fill_hist_by_channel(data_in: np.ndarray, histo_bin_edges: np.ndarray, zero_island_delete_idxs: list):

    """
    Computes per-channel histograms for a multi-channel dataset, with optional removal of specified data segments.

    This function calculates histograms for each channel in the input `data_in` using the provided `histo_bin_edges`.
    It also allows for the exclusion of specific index ranges (`zero_island_delete_idxs`) before histogram computation.

    Parameters:
    -----------
    data_in : np.ndarray (shape `[num_channels, num_samples]`)
        The input multi-channel data array where each row corresponds to a channel.

    histo_bin_edges : np.ndarray (shape `[num_bins + 1]`)
        The edges defining the histogram bins.

    zero_island_delete_idxs : list of tuples
        List of index ranges `(start_idx, end_idx)` specifying segments to remove from all channels before computing histograms.

    Returns:
    --------
    histo_bin_counts : np.ndarray (shape `[num_channels, num_bins]`)
        A 2D array containing histogram counts for each channel.

    Notes:
    ------
    - If `zero_island_delete_idxs` is not empty, specified index ranges are removed before histogram computation.
    - Uses `np.histogram` to compute bin counts for each channel.
    - Outputs progress to `sys.stdout` to indicate per-channel processing status.
    """

    if zero_island_delete_idxs != []:
        zero_island_delete_idxs.sort()
        # Delete zero islands
        for ir in reversed(range(len(zero_island_delete_idxs))):
            data_in = data_in[:, np.concatenate([np.arange(0,zero_island_delete_idxs[ir][0]), np.arange(zero_island_delete_idxs[ir][1]+1, data_in.shape[1])], axis=0)]

    # initialize output 2D array
    num_channels = data_in.shape[0]
    histo_bin_counts = np.zeros([num_channels, len(histo_bin_edges)-1]) # Save a histo count for every channel (will just be sum if scaling same for all channels)

    # Store the counts of datapoints within the histogram bins of interest
    for ch_idx in range(0, num_channels):
        sys.stdout.write("\rComputing histogram for channel ID: " + str(ch_idx) + '/' + str(num_channels-1))
        sys.stdout.flush()    
        # Pull out this channel's data
        data_ch = data_in[ch_idx,:]
        histo_bin_counts[ch_idx, :] = np.histogram(data_ch, histo_bin_edges)[0][:]
    
    return histo_bin_counts

def aquire_scale_params(
        num_channels: int,
        files: list,
        file_starts_dt: list,
        save_dir: str,
        scale_epoch_hours: float,
        buffer_start_hours: float,
        resamp_freq: float,
        histo_min: float,
        histo_max: float,
        num_bins: int):    

    """
    Only supports 'HistEqualScale' and 'By_Channel_Scale'.
    Only normalizes to first X hours of data (no seizure-centered logic).
    """

    # Get the first file datetime, sorted
    all_start_seconds = [int((x - x.min).total_seconds()) for x in file_starts_dt]
    sort_idxs = np.argsort(all_start_seconds)
    sorted_dts = [file_starts_dt[i] for i in sort_idxs]
    first_file_start_datetime = sorted_dts[0]

    normalization_epoch_sec = [buffer_start_hours*3600, (buffer_start_hours + scale_epoch_hours)*3600] # seconds

    # Save normalization epoch info
    df_norm_epoch = pd.DataFrame({'first_file_start_datetime': first_file_start_datetime,
                                  'normalization_epoch_start_sec': normalization_epoch_sec[0],
                                  'normalization_epoch_end_sec': normalization_epoch_sec[1]}, index=[0])
    os.makedirs(save_dir + '/metadata/scaling_metadata/', exist_ok=True)
    df_norm_epoch.to_csv(save_dir + '/metadata/scaling_metadata/' + 'normalization_epoch_seconds.csv')

    # Histogram bin setup
    histo_bin_edges = np.linspace(histo_min, histo_max, num_bins + 1)
    histo_bin_counts = np.zeros([num_channels, num_bins])

    print("\nFollowing files COULD contain normalization range of " +
          str(normalization_epoch_sec[0]/3600) + "-" +
          str(normalization_epoch_sec[1]/3600) + " hours from start data for this subject:")
    for i, file in enumerate(files):
        gc.collect()
        if (file_starts_dt[i] < first_file_start_datetime + datetime.timedelta(seconds=normalization_epoch_sec[1])):
            print(f"[{i+1}/{len(files)}]: {file}")
            with open(file, "rb") as f:
                filt_data_norm = pickle.load(f)
            filt_data_norm = np.float16(filt_data_norm)
            seconds_in_file = filt_data_norm.shape[1] / resamp_freq
            file_end_dt = file_starts_dt[i] + datetime.timedelta(seconds=seconds_in_file)
            norm_end_dt = first_file_start_datetime + datetime.timedelta(seconds=normalization_epoch_sec[1])
            norm_start_dt = first_file_start_datetime + datetime.timedelta(seconds=normalization_epoch_sec[0])

            if file_end_dt < norm_start_dt:
                print("No data in normalization range, skipping file")
                continue

            # Only need later portion of file
            elif (file_starts_dt[i] <= norm_start_dt) & (file_end_dt <= norm_end_dt):
                start_samp_needed_idx = max(0, int((norm_start_dt - file_starts_dt[i]).total_seconds() * resamp_freq))
                filt_data_norm = filt_data_norm[:, start_samp_needed_idx:]
            
            # Only need middle portion of file
            elif (file_starts_dt[i] <= norm_start_dt) & (file_end_dt > norm_end_dt):
                start_samp_needed_idx = max(0, int((norm_start_dt - file_starts_dt[i]).total_seconds() * resamp_freq))
                end_samp_needed_idx = min(filt_data_norm.shape[1], int((norm_end_dt - file_starts_dt[i]).total_seconds() * resamp_freq))
                filt_data_norm = filt_data_norm[:, start_samp_needed_idx:end_samp_needed_idx]
            
            # Only need first portion of file
            elif (file_starts_dt[i] > norm_end_dt) & (file_end_dt > norm_end_dt):
                end_samp_needed_idx = min(filt_data_norm.shape[1], int((norm_end_dt - file_starts_dt[i]).total_seconds() * resamp_freq))
                filt_data_norm = filt_data_norm[:, 0:end_samp_needed_idx]

            epoch_data_zero_bool = abs(filt_data_norm) < 1e-7
            true_islands = find_contiguous_true_indexes(epoch_data_zero_bool[0, :])
            zero_island_delete_idxs = []
            for ir in reversed(range(len(true_islands))):
                island = true_islands[ir]
                if ((island[1] - island[0]) > resamp_freq):
                    zero_island_delete_idxs.append(island)

            histo_bin_counts += fill_hist_by_channel(
                data_in=filt_data_norm,
                histo_bin_edges=histo_bin_edges,
                zero_island_delete_idxs=zero_island_delete_idxs
            )

    # Save histogram values
    df_hist_counts = pd.DataFrame(histo_bin_counts, columns=histo_bin_edges[:-1])
    # df_hist_counts.to_csv(save_dir + '/metadata/scaling_metadata/' + 'histo_bin_counts.csv')
    save_name = save_dir + '/metadata/scaling_metadata/' + 'histo_bin_counts.pkl'
    with open(save_name, "wb") as f:
        pickle.dump(df_hist_counts, f)

    # Compute CDF and scaled CDF for each channel
    cdf_x_avg_vals = (histo_bin_edges[:-1] + histo_bin_edges[1:]) / 2
    num_bins = histo_bin_counts.shape[1]
    cdf_by_channel = np.zeros([num_channels, num_bins])
    cdf_by_channel_scaled = np.zeros([num_channels, num_bins])

    for ch_idx in range(num_channels):
        cdf_by_channel[ch_idx, 0] = histo_bin_counts[ch_idx, 0]
        for bin_idx in range(1, num_bins):
            cdf_by_channel[ch_idx, bin_idx] = cdf_by_channel[ch_idx, bin_idx - 1] + histo_bin_counts[ch_idx, bin_idx]

        max_val = cdf_by_channel[ch_idx, num_bins - 1]
        zero_bin_val = cdf_by_channel[ch_idx, int((num_bins - 1) / 2)]
        min_val = cdf_by_channel[ch_idx, 0]

        cdf_by_channel_scaled[ch_idx, :] = cdf_by_channel[ch_idx, :] - zero_bin_val

        # Avoid division by zero
        if zero_bin_val != min_val:
            cdf_by_channel_scaled[ch_idx, 0:int((num_bins - 1) / 2)] /= (zero_bin_val - min_val)
        else:
            cdf_by_channel_scaled[ch_idx, 0:int((num_bins - 1) / 2)] = 0

        if max_val != zero_bin_val:
            cdf_by_channel_scaled[ch_idx, int((num_bins - 1) / 2) + 1:] /= (max_val - zero_bin_val)
        else:
            cdf_by_channel_scaled[ch_idx, int((num_bins - 1) / 2) + 1:] = 0

    # Fit linear interpolation for each channel
    from scipy import interpolate
    linear_interp_by_ch = [
        interpolate.interp1d(
            cdf_x_avg_vals,
            cdf_by_channel_scaled[ch_idx, :],
            kind='linear',
            bounds_error=False,
            fill_value=(-1, 1)
        ) for ch_idx in range(num_channels)
    ]

    # Pickle the interpolations
    save_name = save_dir + '/metadata/scaling_metadata/' + 'linear_interpolations_by_channel.pkl'
    with open(save_name, "wb") as f:
        pickle.dump(linear_interp_by_ch, f)

    print("\nScale parameters acquired and saved")

    gc.collect()
    return

def employ_norm(
    files: list,
    file_starts_dt: list,
    file_buffer_sec: float,
    resamp_freq: float,
    save_dir: str,
    linear_interp_by_ch: list,
    out_dur: float,
    out_stride: float,
    montage: str,
    savename_base: str,
    PROCESS_FILE_DEBUG_LIST: list
    ):

    """
    @author grahamwjohnson
    Developed between 2023-2025

    Script for Normalizing, Scaling, and Epoching Time Series Data

    This script processes a list of time series files by applying different normalization and scaling techniques to the data. The file data is then segmented into epochs with specified duration and stride, and the epochs are saved to disk as pickle files.

    Parameters:
    - files (list): List of file paths to the time series data files to be processed.
    - file_starts_dt (list): List of start datetime objects corresponding to each file, used to calculate timestamps for epochs.
    - num_channels (int): The number of channels in the time series data.
    - file_buffer_sec (float): Buffer time (in seconds) at the beginning and end of each file to exclude from epoching.
    - resamp_freq (float): Resampling frequency of the time series data in Hz.
    - save_dir (str): Directory where the processed files and histograms will be saved.
    - scale_type (str): The type of scaling to apply to the data (options: 'LinearScale', 'CubeRootScale', 'HyperTanScaling', 'HistEqualScale').
    - scale_factors (list): List of scale factors for each channel.
    - linear_interp_by_ch (list): List of interpolation functions for histogram equalization.
    - out_dur (float): Desired duration of each epoch in seconds.
    - out_stride (float): Desired stride (step size) between consecutive epochs in seconds.
    - montage (str): Type of montage ('BIPOLE' or 'MONOPOLE').
    - savename_base (str): Base name used for saving epoch files.
    - PROCESS_FILE_DEBUG_LIST (list): List of indices of files to process for debugging; if empty, all files are processed.

    Outputs:
    - Processed data is saved as pickle files in the specified save_dir.
    - Normalization histograms are saved in the '/metadata/normalization_histograms/' subdirectory.
    - Zero-padded files are saved in the '/metadata/zero_padded_epochs' subdirectory.

    Note:
    - The script allows for debugging certain files if the PROCESS_FILE_DEBUG_LIST is provided.
    - The script handles scaling using different methods and ensures the data is normalized and clipped to a specified range before epoching.
    - Epoching is done based on the specified duration and stride, with data saved in pickle format for further analysis.

    """
    
    zero_pad_dir = save_dir + '/metadata/zero_padded_epochs'
    if not os.path.exists(zero_pad_dir): os.makedirs(zero_pad_dir)

    total_files = len(files)
    print("\nScaling params aquired: Employing scaling on following files and epoching into desired duration and stride")
    # Re-iterate through all files and normnalize

    # Allow for debugging certain files
    if PROCESS_FILE_DEBUG_LIST:
        process_list = PROCESS_FILE_DEBUG_LIST
    else:
        process_list = range(0,len(files))

    for i in process_list: 
        file = files[i]
        print("\n[" + str(i+1) + '/' + str(total_files) + ']: ' + file)

        # ## Use the previously aquired scale factors to norm the data

        # Load the big pickle
        print("Loading big pickle")
        with open(file, "rb") as f:
            filt_data = pickle.load(f)

        if i == 0: num_channels = filt_data.shape[0]
        else:
            if filt_data.shape[0] != num_channels: raise Exception("ERROR: Not all files for subject have the same number of channels after bipolar montage")
            num_channels = filt_data.shape[0]

        # Ensure that data is in float16 format
        filt_data = np.float16(filt_data)

        scaled_filt_data = np.zeros(filt_data.shape, dtype=np.float16) # initialize for hist scaling below

        # Scale the data with histogram equalization 
        for ch_idx in range(0,num_channels):
            scaled_filt_data[ch_idx,:] = linear_interp_by_ch[ch_idx](filt_data[ch_idx,:])
        
        # Clip after scaling
        scaled_filt_data = scaled_filt_data.clip(-1,1)

        # Save output histograms for this file: 
        minISH_file_filtered = np.percentile(filt_data, 0.01, axis=1)
        maxISH_file_filtered = np.percentile(filt_data, 99.99, axis=1)
        
        hist_save_dir = save_dir + '/metadata/normalization_histograms/' + file.split('/')[-1].split('.')[0]
        if not os.path.exists(hist_save_dir): os.makedirs(hist_save_dir)
        # Turn interactive plotting off
        plt.ioff()
        pl.ioff()

        # TAKES A LOT OF MEMORY
        # Save a summary histogram of entire file data points 
        print("Saving histogram for entire file data")
        if np.max(abs(minISH_file_filtered)) > np.max(abs(maxISH_file_filtered)): maxMax = np.max(abs(minISH_file_filtered))
        else: maxMax = np.max(abs(maxISH_file_filtered))
        gs = gridspec.GridSpec(2, 1)
        fig = pl.figure(figsize=(10, 6))
        ax1 = pl.subplot(gs[0, 0])
        ax1.hist(filt_data.flatten(), 200, (-maxMax, maxMax), 'b', alpha = 0.5, label='All Unscaled File Data')
        ax1.title.set_text('Raw Data - All File Data')
        ax1.legend()
        ax2 = pl.subplot(gs[1, 0])
        ax2.hist(filt_data.flatten(), 200, (-1, 1), 'b', alpha = 0.5, label='All Scaled File Data')
        ax2.title.set_text('Histogram Equalized Data:')
        ax2.legend()
        savename_jpg = hist_save_dir + '/all_file_data_hist.jpg'
        pl.savefig(savename_jpg, dpi=400)
        pl.close(fig)
        del fig

        # Save by channel histograms regardless of channel norm style in order to verify scaling effect
        print("Saving normalization histograms for each channel in file")
        for ch in range(0,num_channels):
            if abs(minISH_file_filtered[ch]) > abs(maxISH_file_filtered[ch]): maxMax_ch  = abs(minISH_file_filtered[ch])
            else: maxMax_ch = abs(maxISH_file_filtered[ch])
            
            gs = gridspec.GridSpec(2, 1)
            fig = pl.figure(figsize=(10, 6))
            ax1 = pl.subplot(gs[0, 0])
            ax1.hist(filt_data[ch,:], 200, (-maxMax_ch, maxMax_ch), 'b', alpha = 0.5, label='All Unscaled File Data for Channel')
            ax1.title.set_text('Raw Data - Channel ID:' + str(ch))
            ax1.legend()
            ax2 = pl.subplot(gs[1, 0])
            ax2.hist(scaled_filt_data[ch,:], 200, (-1, 1), 'b', alpha = 0.5, label='All Scaled File Data for Channel')
            ax2.title.set_text('Histogram Equalized Data - Channel ID:' + str(ch))
            ax2.legend()
            savename_jpg = f"{hist_save_dir}/ch{str(ch)}.jpg"
            pl.savefig(savename_jpg, dpi=400)
            pl.close(fig)
            del fig

        print("Data scaled, now epoching the file data into Window Duration: " + str(out_dur) + " seconds, Stride: "  + str(out_stride) + " seconds")
        del filt_data
        gc.collect()


        ###### EPOCHING ######
        fs = resamp_freq

        # Now that we have scaled we will epoch into desired duration and stride                      
        end_of_file_datetime = file_starts_dt[i] + datetime.timedelta(seconds=scaled_filt_data.shape[1]/fs)
        stride = datetime.timedelta(seconds=out_stride)
        duration = datetime.timedelta(seconds=out_dur)

        # Skip a buffer time at the beginning and end of the file
        curr_start_datetime = file_starts_dt[i] + datetime.timedelta(seconds=file_buffer_sec) 
        curr_start_sample = int(file_buffer_sec * fs)
        sample_duration = int(out_dur * fs)
        sample_stride = int(out_stride * fs)
        while curr_start_datetime + duration < end_of_file_datetime - datetime.timedelta(seconds=file_buffer_sec):
            # If in loop, then another epoch exists from [start time]  to [start time + window duration]

            # Pull the epoch's data
            epoch_data = np.float16(scaled_filt_data[:,curr_start_sample:(curr_start_sample+sample_duration)])

            if montage == 'BIPOLE': pole_str = '_bipole_'
            if montage == 'MONOPOLE': pole_str = '_monopole_'

            # Pickle the epoch
            # Check for ZERO PADDING and skip epoch if present (probably want to do <small number to account for float16 precision)
            epoch_data_zero_bool = abs(epoch_data) < 1e-7
            if epoch_data_zero_bool.sum() > 1000: # arbitrary check for approximate zeros
                s = "ZERO PADDED FILE FOUND: " + str(out_dur) + " second epoch starting at " + curr_start_datetime.strftime("%m/%d/%Y-%H:%M:%S")
                print(s)
                save_name = zero_pad_dir + "/" + savename_base + "_" + curr_start_datetime.strftime("%m%d%Y_%H%M%S%f")[:-4] + "_to_" + (curr_start_datetime + duration).strftime("%m%d%Y_%H%M%S%f")[:-4] + pole_str + "scaled_filtered_data.pkl"
                
            else: # epoch should be non-zero padded
                save_name = save_dir + "/" + savename_base + "_" + curr_start_datetime.strftime("%m%d%Y_%H%M%S%f")[:-4] + "_to_" + (curr_start_datetime + duration).strftime("%m%d%Y_%H%M%S%f")[:-4] + pole_str + "scaled_filtered_data.pkl"
            
            with open(save_name, "wb") as f:
                pickle.dump(epoch_data, f)

            # Prepare the next round:
            # Advance the window by the stride, but redefine by actual samples to avoid drift
            curr_start_sample = curr_start_sample + sample_stride
            curr_start_datetime = file_starts_dt[i] + datetime.timedelta(seconds=curr_start_sample/fs)

            del epoch_data
            gc.collect()
            
        print("Little pickles saved")

        del scaled_filt_data
        gc.collect()
        print("Garbage collected")

def preprocess_directory(in_dir, eq_hrs=24, checkpoint=0, out_dir=None, desired_samp_freq=512):
    """
    Preprocess all .EDF/.edf files in the raw/*/* structure
    and save them to preprocessed/*/*_pp.pkl per subject directory.

    Args:
        in_dir (str): Root directory of raw files
        out_dir (str, optional): Root directory to save preprocessed files.
                                 If None, defaults to the input directory.
    """
    if out_dir is None:
        out_dir = in_dir

    if not os.path.exists(in_dir):
        raise FileNotFoundError(f"Input directory does not exist: {in_dir}")

    os.makedirs(out_dir, exist_ok=True)

    # Read in channels to ignore
    if os.path.exists(CH_TO_IGNORE_PATH):
        with open(CH_TO_IGNORE_PATH, 'r') as f:
            channels_to_ignore = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        print(f"Channels to ignore: {channels_to_ignore}")
    else:
        channels_to_ignore = []
        print("No channels to ignore file found, proceeding without ignoring any channels.")

    print(f"Equalization hours: {eq_hrs} hours")

    # Find all subject directories under input root
    subject_dirs = [d for d in glob.glob(os.path.join(in_dir, "*")) if os.path.isdir(d)]

    if not subject_dirs:
        print(f"No subject directories found under {in_dir}.")
        return

    # Go through each subject directory one at a time because need to equalize at subject level
    print(f"Found {len(subject_dirs)} subject directories under {in_dir}.")
    for subj_path in subject_dirs:
        subject_id = os.path.basename(subj_path)
        
        # If checkpoint ==0 start preprocessing from scratch
        if checkpoint <= 0:
            
            # Find all EDF files in this subject directory
            edf_files = glob.glob(os.path.join(subj_path, "*.[Ee][Dd][Ff]"))
            print(f"\nProcessing subject: {subject_id} with {len(edf_files)} EDF files.")
            if not edf_files:
                print(f"  No EDF files found for subject {subject_id}, skipping.")
                continue

            num_channels = np.ones(len(edf_files), dtype=int) * -1  # Placeholder, will be set in montage_filter_pickle_edfs
        
            file_index = 0
            for infile in edf_files:
                filename, ext = os.path.splitext(os.path.basename(infile))
                outfile = os.path.join(out_dir, subject_id, f"{filename}_bipole_filtered.pkl")

                # Ensure the output directory exists
                os.makedirs(os.path.dirname(outfile), exist_ok=True)

                num_channels[file_index] = montage_filter_pickle_edfs(
                    pat_id=subject_id,
                    edf_file=infile,
                    outfile=outfile,
                    desired_samp_freq=desired_samp_freq,
                    expected_unit='uV',
                    montage='BIPOLE',
                    ch_names_to_ignore=channels_to_ignore,
                    ignore_channel_units=False)
                    
                file_index += 1
            
            # Now check if all channel counts are the same
            if not np.all(num_channels == num_channels[0]):
                raise Exception(f"ERROR: Not all files for subject {subject_id} have the same number of channels after bipolar montage. Channel counts: {num_channels}")

        # Now that all of the files for this subject are bipole montaged and filtered,
        # we can do the equalization step across all files for this subject
        preprocessed_files = glob.glob(os.path.join(out_dir, subject_id, "*_bipole_filtered.pkl"))
        if not preprocessed_files:  # Just a safety check
            raise Exception(f"ERROR: No preprocessed files found for subject {subject_id}")
        print(f"  Found {len(preprocessed_files)} preprocessed files for subject {subject_id}, proceeding to equalization.")

        file_starts_dt = []
        for f in preprocessed_files:
            match = re.search(r'(\d{8}_\d{6})', os.path.basename(f))
            if match:
                dt_str = match.group(1)
                file_start_dt = datetime.datetime.strptime(dt_str, '%m%d%Y_%H%M%S')
                file_starts_dt.append(file_start_dt)
            else:
                raise ValueError(f"Filename {f} does not contain a valid datetime string.")

        # Now can call our equaliztion subfunction
        if checkpoint <= 1:

            # Get the number of channels from bipole montage CSV
            csv_name = out_dir + f'/{subject_id}/metadata/{subject_id}_bipolar_montage_names_and_indexes_from_rawEDF.csv'
            num_channels_incsv = sum(1 for line in open(csv_name)) - 1  # subtract 1 for header

            aquire_scale_params(
                num_channels=num_channels_incsv,
                files=preprocessed_files,
                file_starts_dt=file_starts_dt, 
                save_dir=out_dir + '/' + subject_id,
                scale_epoch_hours=eq_hrs,
                buffer_start_hours=0,
                resamp_freq=desired_samp_freq,
                histo_min=-10000,
                histo_max=10000,
                num_bins=100001)

        # Now employ the calculated equalization on all files for this subject
        # Load the saved scaling params
        if checkpoint <= 2:
            with open(out_dir + '/' + subject_id + '/metadata/scaling_metadata/linear_interpolations_by_channel.pkl', "rb") as f:
                linear_interp_by_ch = pickle.load(f)    
            
            # Now can call our scaling and epoching subfunction
            employ_norm(
                files=preprocessed_files,
                file_starts_dt=file_starts_dt,
                file_buffer_sec=0,
                resamp_freq=desired_samp_freq,
                save_dir=out_dir + '/' + subject_id + '/preprocessed_epoched_data',
                linear_interp_by_ch=linear_interp_by_ch,
                out_dur=1024,
                out_stride=1024,
                montage='BIPOLE',
                savename_base=subject_id,
                PROCESS_FILE_DEBUG_LIST=[])

        else: raise Exception(f"ERROR: Checkpoint value {checkpoint} not recognized. Must be 0, 1, or 2.")

    print("\nAll subjects preprocessed successfully.")


# For Development and Debugging
if __name__ == "__main__":
    # preprocess_directory('/home/graham/Downloads/test_raw_inputs')
    preprocess_directory('/home/graham/Downloads/test_raw3', checkpoint=1)