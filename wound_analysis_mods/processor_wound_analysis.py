import os
import glob
import scipy
import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.ndimage as nd
import imageio.v2 as imageio
import matplotlib.pyplot as plt

class ImageProcessor:

    def __init__(self, filename, im_save_path, img, line_coords, line_length, Ch1, Ch2, frame_rate, pixel_size):
        self.filename = filename
        self.im_save_path = im_save_path
        self.img = img      
        self.line_coords = line_coords
        self.line_length = line_length
        self.Ch1 = Ch1
        self.Ch2 = Ch2
        self.frame_rate = frame_rate
        self.pixel_size = pixel_size
        
        self.num_channels = len(img[0][0])
        self.num_frames = len(img)  
        self.num_lines = len(line_coords)
        self.bin_num = len(line_coords[0][0])
        self.bin_length_pixels = (self.line_length / self.bin_num) * self.pixel_size
        self.line_length_pixels = self.bin_length_pixels * self.bin_num
                
        self.indv_line_values, self.x_values = self.calc_indv_line_values()

    def calc_indv_line_values(self):
        """
        Calculate the every individual line value for each channel and frame.

        Returns:
        self.indv_line_values: numpy array
            4-D array of shape (number of channels, number of frames, number of lines, number of bins per line).
        """
        #create empty np array
        self.indv_line_values = np.zeros(
            shape=(self.num_channels, self.num_frames, self.num_lines, self.bin_num))
        
        #calculate the individual line values
        its = self.num_channels*self.num_frames
        with tqdm(total=its, miniters=its/100) as pbar:
            pbar.set_description('Calculating Linescans')
            for channel in range(self.num_channels):
                for frame in range(self.num_frames):
                    pbar.update(1)
                    x = np.vstack(self.line_coords[:, 0])
                    y = np.vstack(self.line_coords[:, 1])

                    # Extract the values along all lines at once, using cubic interpolation
                    signal = scipy.ndimage.map_coordinates(
                        self.img[frame, 0, channel], (x, y))
                    
                    signal = scipy.signal.savgol_filter(signal, window_length=11, polyorder=3)

                    # Reshape the signal array and store it in the output array
                    signal = signal.reshape((self.num_lines, self.bin_num)) # reversed to make the center of the wound "0"
                    self.indv_line_values[channel, frame, :, ::-1] = signal

                    # calculating the x values of line in microns for calculations later
                    self.x_values = np.arange(len(signal[0]))
                    self.x_values = self.x_values * self.bin_length_pixels
                    
                    

        return self.indv_line_values, self.x_values

    ###################################
    ######### INDV LINE MEAS ##########
    ###################################

    def calc_ind_peak_props(self):
        """
        Calculate properties of individual peaks in the fluorescence signal for each channel, frame, and line.

        This method applies a Savitzky-Golay filter to the fluorescence signal, and then detects peaks using the find_peaks
        function from the scipy.signal module. For each peak, it calculates several properties such as the width, maximum,
        and minimum value of the peak, and stores these measurements in separate arrays for each channel, frame, and line.
        It also calculates the peak amplitude and relative amplitude for each peak, and stores all of the peak-related
        arrays and measurements in a dictionary for easy access later.

        Returns:
        --------
        ind_peak_widths : numpy.ndarray
            An array of shape (num_channels, num_frames, num_lines) containing the average width of each peak.
        ind_peak_maxs : numpy.ndarray
            An array of shape (num_channels, num_frames, num_lines) containing the average maximum value of each peak.
        ind_peak_mins : numpy.ndarray
            An array of shape (num_channels, num_frames, num_lines) containing the average minimum value of each peak.
        ind_peak_amps : numpy.ndarray
            An array of shape (num_channels, num_frames, num_lines) containing the average amplitude of each peak.
        ind_peak_rel_amps : numpy.ndarray
            An array of shape (num_channels, num_frames, num_lines) containing the average relative amplitude of each peak.
        ind_peak_props : dict
            A dictionary containing the smoothed signal, peak locations, peak widths, peak heights, left and right indices,
            and peak prominences for each frame in each channel. The keys are formatted as 'Ch {channel} Frame {frame_num} Line {line_num}'.
        """
        # make empty arrays to fill with peak measurements for each channel
        self.ind_peak_widths = np.zeros(
            shape=(self.num_channels, self.num_frames, self.num_lines))
        self.ind_peak_maxs = np.zeros(
            shape=(self.num_channels, self.num_frames, self.num_lines))
        self.ind_peak_mins = np.zeros(
            shape=(self.num_channels, self.num_frames, self.num_lines))
        # make a dictionary to store the arrays and measurements generated by this function so they don't have to be re-calculated later
        self.ind_peak_props = {}

        #generate the signals for each line, then find the peak
        for channel in range(self.num_channels):
            for frame_num in range(self.num_frames):
                for line_num in range(self.num_lines):

                    signal = scipy.signal.savgol_filter(
                        self.indv_line_values[channel, frame_num, line_num], window_length=11, polyorder=3)
                    peaks, _ = scipy.signal.find_peaks(
                        signal, prominence=(np.max(signal)-np.min(signal))*.6)

                    # if peaks detected, calculate properties and return property averages. Otherwise return NaNs
                    if len(peaks) > 0:
                        proms, _, _ = scipy.signal.peak_prominences(
                            signal, peaks)
                        widths, heights, leftIndex, rightIndex = scipy.signal.peak_widths(
                            signal, peaks, rel_height=0.5)
                        mean_width = np.mean(widths, axis=0)
                        mean_max = np.mean(signal[peaks], axis=0)
                        mean_min = np.mean(signal[peaks]-proms, axis=0)
                        self.ind_peak_widths[channel,
                                             frame_num, line_num] = mean_width
                        self.ind_peak_maxs[channel,
                                           frame_num, line_num] = mean_max
                        self.ind_peak_mins[channel,
                                           frame_num, line_num] = mean_min

                        # store the smoothed signal, peak locations, maxs, mins, and widths for each frame in each channel
                        self.ind_peak_props[f'Ch {channel} Frame {frame_num} Line {line_num}'] = {'smoothed': signal,
                                                                                                  'peaks': peaks,
                                                                                                  'proms': proms,
                                                                                                  'heights': heights,
                                                                                                  'leftIndex': leftIndex,
                                                                                                  'rightIndex': rightIndex}

                    else:
                        self.ind_peak_widths[channel,
                                             frame_num, line_num] = np.nan
                        self.ind_peak_maxs[channel,
                                           frame_num, line_num] = np.nan
                        self.ind_peak_mins[channel,
                                           frame_num, line_num] = np.nan

                        # store the smoothed signal, peak locations, maxs, mins, and widths for each frame in each channel
                        self.ind_peak_props[f'Ch {channel} Frame {frame_num} Line {line_num}'] = {'smoothed': np.nan,
                                                                                                  'peaks': np.nan,
                                                                                                  'proms': np.nan,
                                                                                                  'heights': np.nan,
                                                                                                  'leftIndex': np.nan,
                                                                                                  'rightIndex': np.nan}

            # calculate amplitude and relative amplitude
            self.ind_peak_amps = self.ind_peak_maxs - self.ind_peak_mins
            self.ind_peak_rel_amps = self.ind_peak_amps / self.ind_peak_mins

        return self.ind_peak_widths, self.ind_peak_maxs, self.ind_peak_mins, self.ind_peak_amps, self.ind_peak_rel_amps, self.ind_peak_props

    def calc_indv_CCFs(self):
        '''
        Calculates individual cross-correlation functions (CCFs) for each unique combination of channels.
        Returns the shifts and CCFs as arrays, along with the list of channel combinations.

        Returns:
        indv_shifts (numpy.ndarray): Array of shape (num_combos, num_frames, num_lines) containing the delays (in frames) 
            between the two signals for each channel combination, frame, and line.
        indv_ccfs (numpy.ndarray): Array of shape (num_combos, num_frames, num_lines, bin_num*2-1) containing the CCFs 
            for each channel combination, frame, and line.
        channel_combos (list): List of lists containing the unique combinations of channels used to calculate the CCFs.
        '''
        # make a list of unique channel combinations to calculate CCF for
        channels = list(range(self.num_channels))
        self.channel_combos = []
        for i in range(self.num_channels):
            for j in channels[i + 1:]:
                self.channel_combos.append([channels[i], j])
        self.num_combos = len(self.channel_combos)

        # calc shifts and cross-corelation
        self.indv_shifts = np.zeros(
            shape=(self.num_combos, self.num_frames, self.num_lines))
        self.indv_ccfs = np.zeros(
            shape=(self.num_combos, self.num_frames, self.num_lines, self.bin_num*2-1))

        #for each channel combo, calculate the CCF
        for combo_number, combo in enumerate(self.channel_combos):
            for frame_num in range(self.num_frames):
                for line_num in range(self.num_lines):
                    signal1 = self.indv_line_values[combo[0],
                                                    frame_num, line_num]
                    signal2 = self.indv_line_values[combo[1],
                                                    frame_num, line_num]

                    delay_frames, cc_curve = self.calc_shifts(signal1, signal2, prominence=0.1)

                    self.indv_shifts[combo_number, frame_num,
                                     line_num] = delay_frames
                    self.indv_ccfs[combo_number,
                                   frame_num, line_num] = cc_curve

        return self.indv_shifts, self.indv_ccfs, self.channel_combos

    def calc_shifts(self, signal1, signal2, prominence):
        """
        Calculate the delay between two 1D signals based on their cross-correlation.

        This function first applies a Savitzky-Golay filter to each input signal with a window
        length of 11 and a polynomial order of 3 to smooth the signals. It then identifies the
        prominent peaks in each signal using the `find_peaks` function from SciPy's signal module.
        If both signals have at least one prominent peak, it calculates the cross-correlation
        between the two signals using the `correlate` function from NumPy. The resulting
        cross-correlation curve is smoothed and normalized, and the prominent peaks are again
        identified. If there is more than one peak, the peak closest to the center of the curve is
        chosen as the delay. The delay is then converted to bins by subtracting the index of the
        peak from the center of the curve. If either input signal does not have any prominent peaks,
        the function returns NaNs for both the delay and the cross-correlation curve.

        Args:
        signal1 (ndarray): The first input signal.
        signal2 (ndarray): The second input signal.

        Returns:
        tuple: A tuple containing the delay between the two signals in bins (or NaN if no
            prominent peaks were identified), and the cross-correlation curve between the two signals
            (or an array of NaNs if no prominent peaks were identified).
        """
        #smooth signals
        signal1 = scipy.signal.savgol_filter(
            signal1, window_length=11, polyorder=3)
        signal2 = scipy.signal.savgol_filter(
            signal2, window_length=11, polyorder=3)

        # Find peaks in the signals
        peaks1, _ = scipy.signal.find_peaks(
            signal1, prominence=(np.max(signal1)-np.min(signal1))*0.25)
        peaks2, _ = scipy.signal.find_peaks(
            signal2, prominence=(np.max(signal2)-np.min(signal2))*0.25)

        if len(peaks1) > 0 and len(peaks2) > 0:
            corr_signal1 = signal1 - signal1.mean()
            corr_signal2 = signal2 - signal2.mean()
            cc_curve = np.correlate(
                corr_signal1, corr_signal2, mode='full')
            # smooth the curve
            cc_curve = scipy.signal.savgol_filter(
                cc_curve, window_length=11, polyorder=3)
            # normalize the curve
            cc_curve = cc_curve / \
                (self.bin_num * signal1.std() * signal2.std())
            # find peaks
            peaks, _ = scipy.signal.find_peaks(
                cc_curve, prominence=prominence)
            # absolute difference between each peak and zero
            peaks_abs = abs(peaks - cc_curve.shape[0] // 2)
            # if peaks were identified, pick the one closest to the center
            if len(peaks) > 1:
                delay = np.argmin(peaks_abs[np.nonzero(peaks_abs)])
                delayIndex = peaks[delay]
                delay_frames = delayIndex - cc_curve.shape[0] // 2
                delay_frames = delay_frames # negative because values were flipped before so that the wound center wound be considered 0
            # otherwise, return NaNs for both period and autocorrelation curve
            else:
                delay_frames = np.nan
                cc_curve = np.full((self.bin_num * 2 - 1), np.nan)
        else:
            delay_frames = np.nan
            cc_curve = np.full((self.bin_num * 2 - 1), np.nan)

        return delay_frames, cc_curve

    ###################################
    ######### CALC LINE MEANS #########
    ###################################

    def calculate_mean_line_values_per_frame(self):
        """
        Calculates the mean line values per frame and per channel over all lines in the image.

        Returns:
        A numpy array of shape (num_channels, num_frames, bin_num) containing the mean line values 
            per frame and per channel over all lines in the image.
        """
        # Calculate mean values per bin for each frame and channel
        self.mean_line_values_per_frame = np.zeros(
            shape=(self.num_channels, self.num_frames, self.bin_num))
        for channel in range(self.num_channels):
            for frame in range(self.num_frames):
                for bin in range(self.bin_num):
                    #calculate the avg pixel value at each bin location for every line in a frame
                    self.mean_line_values_per_frame[channel, frame,
                                                    bin] = self.indv_line_values[channel, frame, :, bin].mean()

        return self.mean_line_values_per_frame

    def calc_mean_peak_props(self):
        '''
        Calculates mean cross-correlation functions (CCFs) for each unique combination of channels.
        Returns the shifts and CCFs as arrays, along with the list of channel combinations.

        Returns:
        mean_shifts (numpy.ndarray): Array of shape (num_combos, num_frames, num_lines) containing the delays (in frames) 
            between the two signals for each channel combination, frame, and line.
        mean_ccfs (numpy.ndarray): Array of shape (num_combos, num_frames, num_lines, bin_num*2-1) containing the CCFs 
            for each channel combination, frame, and line.
        '''
        # make empty arrays to fill with peak measurements for channels
        self.mean_peak_widths = np.zeros(
            shape=(self.num_channels, self.num_frames))
        self.mean_peak_maxs = np.zeros(
            shape=(self.num_channels, self.num_frames))
        self.mean_peak_mins = np.zeros(
            shape=(self.num_channels, self.num_frames))
        
        # make a dictionary to store the arrays and measurements generated by this function so they don't have to be re-calculated later
        self.mean_peak_props = {}

        #generate the signals for each line, then find the peak
        for channel in range(self.num_channels):
            for frame_num in range(self.num_frames):
                signal = scipy.signal.savgol_filter(
                    self.mean_line_values_per_frame[channel, frame_num], window_length=11, polyorder=3)
                peaks, _ = scipy.signal.find_peaks(
                    signal, prominence=(np.max(signal)-np.min(signal))*0.6)

                # if peaks detected, calculate properties and return property averages. Otherwise return nans
                if len(peaks) > 0:
                    proms, _, _ = scipy.signal.peak_prominences(
                        signal, peaks)
                    widths, heights, leftIndex, rightIndex = scipy.signal.peak_widths(
                        signal, peaks, rel_height=0.5)
                    self.mean_peak_widths[channel, frame_num] = np.mean(
                        widths, axis=0)
                    self.mean_peak_maxs[channel, frame_num] = np.mean(
                        signal[peaks], axis=0)
                    self.mean_peak_mins[channel, frame_num] = np.mean(
                        signal[peaks]-proms, axis=0)

                    # store the smoothed signal, peak locations, maxs, mins, and widths for each frame in each channel
                    self.mean_peak_props[f'Ch {channel} Frame {frame_num}'] = {'smoothed': signal,
                                                                               'peaks': peaks,
                                                                               'proms': proms,
                                                                               'heights': heights,
                                                                               'leftIndex': leftIndex,
                                                                               'rightIndex': rightIndex}

                else:
                    self.mean_peak_widths[channel, frame_num] = np.nan
                    self.mean_peak_maxs[channel, frame_num] = np.nan
                    self.mean_peak_mins[channel, frame_num] = np.nan

                    # store the smoothed signal, peak locations, maxs, mins, and widths for each frame in each channel
                    self.mean_peak_props[f'Ch {channel} Frame {frame_num}'] = {'smoothed': np.nan,
                                                                               'peaks': np.nan,
                                                                               'proms': np.nan,
                                                                               'heights': np.nan,
                                                                               'leftIndex': np.nan,
                                                                               'rightIndex': np.nan}
                    
        # calculate amplitude and relative amplitude
        self.mean_peak_amps = self.mean_peak_maxs - self.mean_peak_mins
        self.mean_peak_rel_amps = self.mean_peak_amps / self.mean_peak_mins

        return self.mean_peak_widths, self.mean_peak_maxs, self.mean_peak_mins, self.mean_peak_amps, self.mean_peak_rel_amps, self.mean_peak_props

    def calc_mean_CCF(self):
        """
        Calculate the mean cross-correlation function (CCF) and delay for each combination of channels and frames.

        This method loops over all channel combinations and frames and calls the `calc_shifts` function to calculate
        the delay and CCF between the two channels for each frame. The resulting mean delay and mean CCF are stored in
        arrays `mean_shifts` and `mean_ccfs`, respectively.

        Returns:
        tuple: A tuple containing the mean delays and mean CCFs for each channel combination and frame.
            The first element of the tuple is a 2D array of shape (num_combos, num_frames), where each row represents
            a channel combination and each column represents a frame. The second element of the tuple is a 3D array
            of shape (num_combos, num_frames, bin_num*2-1), where each row represents a channel combination, each
            column represents a frame, and each depth slice represents the CCF for that combination and frame.
        """
        # calc shifts and cross-corelation
        self.mean_shifts = np.zeros(
            shape=(self.num_combos, self.num_frames))
        self.mean_ccfs = np.zeros(
            shape=(self.num_combos, self.num_frames, self.bin_num*2-1))

        for combo_number, combo in enumerate(self.channel_combos):
            for frame_num in range(self.num_frames):
                signal1 = self.mean_line_values_per_frame[combo[0],
                                                          frame_num]
                signal2 = self.mean_line_values_per_frame[combo[1],
                                                          frame_num]

                delay_frames, cc_curve = self.calc_shifts(signal1, signal2, prominence=0.01)

                self.mean_shifts[combo_number, frame_num] = delay_frames
                self.mean_ccfs[combo_number, frame_num] = cc_curve

        return self.mean_shifts, self.mean_ccfs

    ###################################
    ####### INDV PLOTTING #############
    ###################################

    def plot_ind_ccfs(self):
        """
        Plots individual cross-correlation functions (CCFs) for each channel combination, frame, and line.

        Returns:
        dict: A dictionary containing the figures for each individual CCF.
        """
        def return_figure(ch1: np.ndarray, ch2: np.ndarray, ccf_curve: np.ndarray, ch1_name: str, ch2_name: str, shift: int):
            """
            Returns a figure containing two subplots: one for the two input channels, and one for their cross-correlation function (CCF).

            Args:
            ch1 (np.ndarray): An array containing the pixel values of the first channel.
            ch2 (np.ndarray): An array containing the pixel values of the second channel.
            ccf_curve (np.ndarray): An array containing the cross-correlation function values.
            ch1_name (str): A string containing the name of the first channel.
            ch2_name (str): A string containing the name of the second channel.
            shift (int): An integer representing the shift between the two channels, if one exists.

            Returns:
            matplotlib.figure.Figure: A figure containing two subplots: one for the two input channels, and one for their CCF.
            """
            plt.style.use('dark_background')
            fig, (ax1, ax2) = plt.subplots(2, 1)
            ax1.plot(self.x_values, ch1, color='tab:blue', label=ch1_name)
            ax1.plot(self.x_values, ch2, color='tab:orange', label=ch2_name)
            ax1.set_xlabel('distance (microns)')
            ax1.set_ylabel('Mean px value')
            ax1.legend(loc='upper right', fontsize='small', ncol=1)
            ax2.plot(np.arange(-self.bin_num + 1, self.bin_num) * self.bin_length_pixels, ccf_curve)
            ax2.set_ylabel('Crosscorrelation')
            

            if not shift == np.nan:
                color = 'red'
                ax2.axvline(x=shift * self.bin_length_pixels, alpha=0.5, c=color, linestyle='--')
                if shift < 1:
                    ax2.set_xlabel(
                        f'{ch1_name} leads by {int(abs(shift * self.bin_length_pixels))} microns')
                elif shift > 1:
                    ax2.set_xlabel(
                        f'{ch2_name} leads by {int(abs(shift * self.bin_length_pixels))} microns')
                else:
                    ax2.set_xlabel('no shift detected')
            else:
                ax2.set_xlabel(f'No peaks identified')

            fig.subplots_adjust(hspace=0.5)
            plt.close()
            return(fig)

        def normalize(signal: np.ndarray):
            """
            Normalizes an input signal between 0 and 1.

            Args:
            signal (np.ndarray): An array containing the signal to be normalized.

            Returns:
            np.ndarray: An array containing the normalized signal.
            """

            return (signal - np.min(signal)) / (np.max(signal) - np.min(signal))

        # empty dictionary to fill with figures, in the event that we make more than one
        self.ind_ccf_plots = {}

        # fill the dictionary with plots for each channel
        if self.num_channels > 1:
            its = len(self.channel_combos)*self.num_frames*self.num_lines
            with tqdm(total=its, miniters=its/100) as pbar:
                pbar.set_description('plotting indv ccfs')
                for combo_number, combo in enumerate(self.channel_combos):
                    for frame in range(self.num_frames):
                        for line in range(self.num_lines):
                            pbar.update(1)
                            self.ind_ccf_plots[f'Ch{combo[0] + 1}-Ch{combo[1] + 1} Frame {frame + 1} Line {line + 1} CCF'] = return_figure(ch1=normalize(self.indv_line_values[combo[0], frame, line]),
                                                                                                                                   ch2=normalize(self.indv_line_values[combo[1], frame, line]),
                                                                                                                                   ccf_curve=self.indv_ccfs[combo_number, frame, line],
                                                                                                                                   ch1_name=f'Ch{combo[0] + 1}',
                                                                                                                                   ch2_name=f'Ch{combo[1] + 1}',
                                                                                                                                   shift=self.indv_shifts[combo_number, frame, line])

        return self.ind_ccf_plots

    def plot_ind_peak_props(self):
        """
        Plot individual peak properties for each line of each frame of each channel.

        Calls the `return_figure` function to create a plot for each line of each frame of each channel.

        Returns:
        ind_peak_figs (dict): A dictionary of figures, where the keys are strings describing the channel, 
            frame, and line, and the values are the corresponding figures.
        """
        def return_figure(line_signal: np.ndarray, prop_dict: dict, Ch_name: str):
            """
            Create a plot of the smoothed signal and peak properties.

            Args:
            line_signal (np.ndarray): A 1D numpy array of the raw signal.
            prop_dict (dict): A dictionary of peak properties, where the keys are 'smoothed', 'peaks', 
                'proms', 'heights', 'leftIndex', and 'rightIndex', and the values are numpy arrays.
            Ch_name (str): A string describing the channel, frame, and line.

            Returns:
            fig (matplotlib.figure.Figure): A matplotlib figure object.
            """
            plt.style.use('dark_background')
            smoothed_signal = prop_dict['smoothed']
            peaks = prop_dict['peaks']
            proms = prop_dict['proms']
            heights = prop_dict['heights']
            leftIndex = prop_dict['leftIndex']
            rightIndex = prop_dict['rightIndex']

            fig, ax = plt.subplots()
            ax.plot(self.x_values, line_signal, color='tab:gray', label='raw signal')
            ax.plot(self.x_values, smoothed_signal, color='tab:cyan', label='smoothed signal')

            # plot all of the peak widths and amps in a loop
            for i in range(peaks.shape[0]):
                ax.hlines(heights[i],
                          leftIndex[i] * self.bin_length_pixels,
                          rightIndex[i] * self.bin_length_pixels,
                          color='tab:olive',
                          linestyle='-')
                ax.vlines(peaks[i] * self.bin_length_pixels,
                          smoothed_signal[peaks[i]]-proms[i],
                          smoothed_signal[peaks[i]],
                          color='tab:purple',
                          linestyle='-')
            # plot the first peak width and amp again so we can add it to the legend
            ax.hlines(heights[0],
                      leftIndex[0] * self.bin_length_pixels,
                      rightIndex[0] * self.bin_length_pixels,
                      color='tab:olive',
                      linestyle='-',
                      label='FWHM')
            ax.vlines(peaks[0] * self.bin_length_pixels,
                      smoothed_signal[peaks[0]]-proms[0],
                      smoothed_signal[peaks[0]],
                      color='tab:purple',
                      linestyle='-',
                      label='Peak amplitude')

            ax.legend(loc='upper right', fontsize='small', ncol=1)
            ax.set_xlabel('Distance (microns)')
            ax.set_ylabel('Signal (AU)')
            ax.set_title(f'{Ch_name} peak properties')
            plt.close(fig)

            return fig

        # empty dictionary to fill with figures, in the event that we make more than one
        self.ind_peak_figs = {}

        # fill the dictionary with plots for each channel
        if hasattr(self, 'ind_peak_widths'):
            its = self.num_channels*self.num_frames*self.num_lines
            with tqdm(total=its, miniters=its/100) as pbar:
                pbar.set_description('plotting indv peaks')
                for channel in range(self.num_channels):
                    for frame in range(self.num_frames):
                        for line in range(self.num_lines):
                            pbar.update(1)
                            peaks = self.ind_peak_props[f'Ch {channel} Frame {frame} Line {line}']['peaks']
                            #If only one peak
                            if type(peaks) == float:
                                #If NaN, then do not plot
                                if peaks == peaks:
                                    self.ind_peak_figs[f'Ch{channel + 1} Frame {frame + 1} Line {line + 1} LinePeak Props'] = return_figure(self.indv_line_values[channel, frame, line],
                                                                                                                                            self.ind_peak_props[
                                                                                                                                            f'Ch {channel} Frame {frame} Line {line}'],
                                                                                                                                            f'Ch{channel + 1} Frame {frame + 1} Line {line + 1}')
                            #If more than one peak
                            elif peaks[0] == peaks[0]:
                                self.ind_peak_figs[f'Ch{channel + 1} Frame {frame + 1} Line {line + 1} LinePeak Props'] = return_figure(self.indv_line_values[channel, frame, line],
                                                                                                                                        self.ind_peak_props[f'Ch {channel} Frame {frame} Line {line}'],
                                                                                                                                        f'Ch{channel + 1} Frame {frame + 1} Line {line + 1}')
                                
        return self.ind_peak_figs

    ###################################
    ####### MEAN PLOTTING #############
    ###################################

    def plot_mean_CCF(self):
        """
        Plots a summary of the period measurements for each channel and frame.

        Returns:
        matplotlib.figure.Figure: A matplotlib figure containing a plot of the mean cross-correlation curve, a histogram 
            of shift values, and a boxplot of shift values for each channel and frame.
        """
        def return_figure(arr: np.ndarray, shifts_or_periods: np.ndarray, channel_frame_combo: str):
            """
            Create a plot summarizing cross-correlation and shift measurements.

            Args:
            arr (np.ndarray): A 1D numpy array of cross-correlation values.
            shifts_or_periods (np.ndarray): A 1D numpy array of shift values.
            channel_frame_combo (str): A string describing the channel and frame.

            Returns:
            fig (matplotlib.figure.Figure): A matplotlib figure object.
            """
            plt.style.use('dark_background')
            fig, ax = plt.subplot_mosaic(mosaic='''
                                                  AA
                                                  BC
                                                  ''')
            arr_std = np.nanstd(arr, axis=0)
            x_axis = np.arange(-self.bin_num + 1, self.bin_num)
            ax['A'].plot(x_axis, arr, color='blue')
            ax['A'].fill_between(x_axis,
                                 arr - arr_std,
                                 arr + arr_std,
                                 color='blue',
                                 alpha=0.2)
            ax['A'].set_title(
                f'{channel_frame_combo} Mean Crosscorrelation Curve ± Standard Deviation')

            #to standardize to pixel size
            for i in range(len(shifts_or_periods)):
                shifts_or_periods[i] *= self.bin_length_pixels 
            
            ax['B'].hist(shifts_or_periods)
            shifts_or_periods = [
                val for val in shifts_or_periods if not np.isnan(val)]
            ax['B'].set_xlabel(f'Histogram of shift values (microns)')
            ax['B'].set_ylabel('Occurances')
            ax['C'].boxplot(shifts_or_periods)
            ax['C'].set_xlabel(f'Boxplot of shift values')
            ax['C'].set_ylabel(f'Measured shift (microns)')
            fig.subplots_adjust(hspace=0.25, wspace=0.5)
            plt.close(fig)

            return fig

        # empty dict to fill with figures, in the event that we make more than one
        self.mean_ccf_figs = {}

        # fill the dictionary with plots for each channel
        if self.num_channels > 1:
            its = self.num_frames
            with tqdm(total=its, miniters=its/100) as pbar:
                pbar.set_description('plotting mean ccfs')
                for combo_number, combo in enumerate(self.channel_combos):
                    for frame_num in range(self.num_frames):
                        pbar.update(1)
                        arr = self.mean_ccfs[combo_number][frame_num]
                        if not np.isnan(arr).any():
                            self.mean_ccf_figs[f'Ch{combo[0] + 1}-Ch{combo[1] + 1} Frame {frame_num + 1} Mean CCF'] = return_figure(arr,
                                                                                                                                    self.indv_shifts[combo_number, frame_num],
                                                                                                                                    f'Ch{combo[0] + 1}-Ch{combo[1] + 1} Frame {frame_num + 1}')

        return self.mean_ccf_figs

    def plot_mean_peak_props(self):
        """
        Create a plot of the mean peak properties for each channel and frame.

        Args:
        min_array (np.ndarray): A 1D numpy array of the minimum peak values.
        max_array (np.ndarray): A 1D numpy array of the maximum peak values.
        amp_array (np.ndarray): A 1D numpy array of the peak amplitudes.
        width_array (np.ndarray): A 1D numpy array of the peak widths.
        Ch_name (str): A string describing the channel and frame.

        Returns:
        fig (matplotlib.figure.Figure): A matplotlib figure object containing the mean peak properties plot.
        """
        def return_figure(min_array: np.ndarray, max_array: np.ndarray, amp_array: np.ndarray, width_array: np.ndarray, Ch_name: str):
            """
            Create a figure with four subplots displaying the peak properties for a given channel.

            Args:
            min_array (np.ndarray): Array of minimum peak values.
            max_array (np.ndarray): Array of maximum peak values.
            amp_array (np.ndarray): Array of peak amplitudes.
            width_array (np.ndarray): Array of peak widths.
            Ch_name (str): Name of the channel.

            Returns:
            matplotlib.figure.Figure: Figure object displaying histograms and boxplots of the peak properties.
            """
            plt.style.use('dark_background')
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
            # filter nans out of arrays
            min_array = [val for val in min_array if not np.isnan(val)]
            max_array = [val for val in max_array if not np.isnan(val)]
            amp_array = [val for val in amp_array if not np.isnan(val)]
            width_array = [val for val in width_array if not np.isnan(val)]

            plot_params = {'amp': (amp_array, 'tab:blue'),
                           'min': (min_array, 'tab:purple'),
                           'max': (max_array, 'tab:orange')}
            for labels, (arr, arr_color) in plot_params.items():
                ax1.hist(arr, color=arr_color, label=labels, alpha=0.75)
            boxes = ax2.boxplot(
                [val[0] for val in plot_params.values()], patch_artist=True)
            ax2.set_xticklabels(plot_params.keys())
            for box, box_color in zip(boxes['boxes'], [val[1] for val in plot_params.values()]):
                box.set_color(box_color)

            ax1.legend(loc='upper right', fontsize='small', ncol=1)
            ax1.set_xlabel(f'{Ch_name} histogram of peak values')
            ax1.set_ylabel('Occurances')

            ax2.set_xlabel(f'{Ch_name} boxplot of peak values')
            ax2.set_ylabel('Value (AU)')

            #to standardize to pixel size
            for i in range(len(width_array)):
                width_array[i] *= self.bin_length_pixels

            ax3.hist(width_array, color='dimgray', alpha=0.75)
            ax3.set_xlabel(f'{Ch_name} histogram of peak widths')
            ax3.set_ylabel('Occurances')
            bp = ax4.boxplot(width_array, vert=True, patch_artist=True)
            bp['boxes'][0].set_facecolor('dimgray')
            ax4.set_xlabel(f'{Ch_name} boxplot of peak widths')
            ax4.set_ylabel('Peak width (microns)')
            fig.subplots_adjust(hspace=0.6, wspace=0.6)
            plt.close(fig)

            return fig

        # empty dictionary to fill with figures, in the event that we make more than one
        self.mean_peak_figs = {}

        # fill the dictionary with plots for each channel
        if hasattr(self, 'mean_peak_widths'):
            its = self.num_channels*self.num_frames
            with tqdm(total=its, miniters=its/100) as pbar:
                pbar.set_description('plotting mean peaks')
                for channel in range(self.num_channels):
                    for frame in range(self.num_frames):
                        pbar.update(1)
                        self.mean_peak_figs[f'Ch {channel + 1} Frame {frame + 1} Peak Props'] = return_figure(self.ind_peak_mins[channel, frame],
                                                                                                              self.ind_peak_maxs[channel, frame],
                                                                                                              self.ind_peak_amps[channel, frame],
                                                                                                              self.ind_peak_widths[channel, frame],
                                                                                                              f'Ch {channel + 1} Frame {frame + 1}')

        return self.mean_peak_figs

    ###################################
    ###### FINAL PLOTTING #############
    ###################################

    def plot_reference_figure(self):
        """
        Creates and returns a reference figure for the image analysis.

        Returns:
        matplotlib.figure.Figure: The reference figure.
        """
        plt.style.use('dark_background')
        self.ref_fig, ax = plt.subplots()
        for coords in self.line_coords:
            ax.plot(coords[1], coords[0], 'w-', linewidth=0.5)
        ax.set_aspect('equal', adjustable='box')
        ref_frame = int(len(self.img) / 2)
        ax.imshow(self.img[ref_frame][0][0])
        self.ref_fig.subplots_adjust(hspace=0.5)
        return self.ref_fig

    def create_mean_linescan_plot_per_frame(self, frame):
        """
        Creates a linescan plot for a single frame and displays the Ch1 and Ch2 signals.

        Args:
        frame (int): The frame number to create the linescan plot for.

        Returns:
        matplotlib.figure.Figure: The linescan plot figure.
        """
        plt.style.use('dark_background')
        self.fig, (ax1) = plt.subplots(1, 1)
        plt.title(f'Frame {frame + 1}: {self.frame_rate*frame} seconds')
        ax1.set_xlabel(f'Distance from center (microns)')
        ax1.set_ylabel('Normalized Pixel Value')
        y_max = np.max(np.maximum(
            self.mean_line_values_per_frame[0], self.mean_line_values_per_frame[1]))
        ax1.set_ylim([0, y_max])
        signal1 = self.mean_line_values_per_frame[0][frame]
        signal2 = self.mean_line_values_per_frame[1][frame]

        ax1.plot(self.x_values, signal1, color='cyan', label=self.Ch1)
        ax1.plot(self.x_values, signal2, color='red', label=self.Ch2)
        plt.legend(loc='upper right', fontsize='small', ncol=3)
        plt.tight_layout()

    def create_mean_linescan_per_frame_movie(self):
        """
        Creates a linescan movie for all frames and saves it as an MP4 file in the specified image save path.

        Returns:
        None
        """
        for i in range(len(self.mean_line_values_per_frame[0])):
            fig = self.create_mean_linescan_plot_per_frame(i)
            plt.savefig(f'{self.im_save_path}/frame_{i:04d}.png')
            plt.close(fig)
        filename_wo_ext = self.filename.split('.')[0]
        with imageio.get_writer(f'{self.im_save_path}/{filename_wo_ext}.mp4', mode='I') as writer:
            for i in range(len(self.mean_line_values_per_frame[0])):
                image = imageio.imread(
                    f'{self.im_save_path}/frame_{i:04d}.png')
                writer.append_data(image)
        png_files = glob.glob(os.path.join(f'{self.im_save_path}', "*.png"))
        for file in png_files:
            os.remove(file)

    ###################################
    ## STORING MEASUREMENTS ###########
    ###################################

    def organize_measurements(self):
        """
        Organizes the measurements obtained from the image analysis.

        This method takes in the measurements obtained from the image analysis and organizes them into a 
        single pandas DataFrame containing summary statistics such as Mean, Median, Standard Deviation, and 
        Standard Error of the Mean (SEM). The measurements are grouped according to the type of measurement 
        (i.e., Shift or Peak), and then summarized for each channel or channel combination (depending on the
        type of measurement). The resulting DataFrame is saved as a class attribute.

        Returns:
        pandas DataFrame: A pandas DataFrame containing the organized measurements.
        """
        def add_stats(measurements: np.ndarray, measurement_name: str):
            """
            This function takes in an array of measurements and a measurement name and returns a list of 
            statistics for each measurement.

            If the measurement name is 'Shift', the function generates statistics for each combination 
            of channels and returns a list of statistics for each combination.

            If the measurement name is not 'Shift', the function generates statistics for each channel 
            and returns a list of statistics for each channel.

            The list of statistics includes the mean, median, standard deviation, standard error of 
            the mean, and the measurements themselves.

            Parameters:
            measurements (np.ndarray): An array of measurements to generate statistics for.
            measurement_name (str): A string indicating the type of measurement being analyzed.

            Returns:
            statified (list): A list of statistics for each measurement. The list contains the measurement name, mean,
                median, standard deviation, standard error of the mean, and the measurements themselves.
            """

            # shift measurements need special treatment to generate the correct measurements and names
            if measurement_name == 'Shift':
                statified = []

                # calculate the shift in pixels

                for combo_number, combo in enumerate(self.channel_combos):
                    # calculate the shift in pixels
                    measurements[combo_number] = measurements[combo_number] * -self.bin_length_pixels

                    meas_mean = np.nanmean(measurements[combo_number])
                    meas_median = np.nanmedian(measurements[combo_number])
                    meas_std = np.nanstd(measurements[combo_number])
                    meas_sem = meas_std / \
                        np.sqrt(len(measurements[combo_number]))
                    meas_list = list(measurements[combo_number])
                    meas_list.insert(0, meas_mean)
                    meas_list.insert(1, meas_median)
                    meas_list.insert(2, meas_std)
                    meas_list.insert(3, meas_sem)
                    meas_list.insert(
                        0, f'Ch{combo[0] + 1}-Ch{combo[1] + 1} {measurement_name}')
                    statified.append(meas_list)

            elif measurement_name == 'Peak Width':
                statified = []
                for channel in range(self.num_channels):
                    # calculate the width in pixels
                    measurements[channel] = measurements[channel] * self.bin_length_pixels

                    meas_mean = np.nanmean(measurements[channel])
                    meas_median = np.nanmedian(measurements[channel])
                    meas_std = np.nanstd(measurements[channel])
                    meas_sem = meas_std / np.sqrt(len(measurements[channel]))
                    meas_list = list(measurements[channel])
                    meas_list.insert(0, meas_mean)
                    meas_list.insert(1, meas_median)
                    meas_list.insert(2, meas_std)
                    meas_list.insert(3, meas_sem)
                    meas_list.insert(0, f'Ch {channel +1} {measurement_name}')
                    statified.append(meas_list)

            # peak measurements are just iterated by channel
            else:
                statified = []
                for channel in range(self.num_channels):
                    meas_mean = np.nanmean(measurements[channel])
                    meas_median = np.nanmedian(measurements[channel])
                    meas_std = np.nanstd(measurements[channel])
                    meas_sem = meas_std / np.sqrt(len(measurements[channel]))
                    meas_list = list(measurements[channel])
                    meas_list.insert(0, meas_mean)
                    meas_list.insert(1, meas_median)
                    meas_list.insert(2, meas_std)
                    meas_list.insert(3, meas_sem)
                    meas_list.insert(0, f'Ch {channel +1} {measurement_name}')
                    statified.append(meas_list)
            return(statified)

        # column names for the dataframe summarizing the box results
        col_names = ["Parameter", "Mean", "Median", "StdDev", "SEM"]
        col_names.extend([f'Frame{i + 1}' for i in range(self.num_frames)])
        # combine all the statified measurements into a single list
        statified_measurements = []

        if hasattr(self, 'mean_ccfs'):
            self.shifts_with_stats = add_stats(self.mean_shifts, 'Shift')
            for combo_number, combo in enumerate(self.channel_combos):
                statified_measurements.append(
                    self.shifts_with_stats[combo_number])

        if hasattr(self, 'mean_peak_widths'):
            self.peak_widths_with_stats = add_stats(
                self.mean_peak_widths, 'Peak Width')
            self.peak_maxs_with_stats = add_stats(
                self.mean_peak_maxs, 'Peak Max')
            self.peak_mins_with_stats = add_stats(
                self.mean_peak_mins, 'Peak Min')
            self.peak_amps_with_stats = add_stats(
                self.mean_peak_amps, 'Peak Amp')
            self.peak_relamp_with_stats = add_stats(
                self.mean_peak_rel_amps, 'Peak Rel Amp')
            for channel in range(self.num_channels):
                statified_measurements.append(
                    self.peak_widths_with_stats[channel])
                statified_measurements.append(
                    self.peak_maxs_with_stats[channel])
                statified_measurements.append(
                    self.peak_mins_with_stats[channel])
                statified_measurements.append(
                    self.peak_amps_with_stats[channel])
                statified_measurements.append(
                    self.peak_relamp_with_stats[channel])

        # and turn it into a dataframe
        self.im_measurements = pd.DataFrame(
            statified_measurements, columns=col_names)
        return self.im_measurements

    def summarize_image(self, file_name=None):
        """
        This function takes in a file name and group name, and summarizes measurements for each image. 
        The summarized measurements are stored in a dictionary called 'file_data_summary'. 
        The function also calculates the percentage of shifts and peaks that have no data and 
        stores these values in the dictionary. If shifts or peaks are present, the function stores the
        mean, median, standard deviation, and standard error of mean for each property.

        Parameters: 
        (string) the name of the file to be summarized

        Returns:
        (dictionary) a dictionary of summarized measurements for the image
        """
        # dictionary to store the summarized measurements for each image
        self.file_data_summary = {}

        if file_name:
            self.file_data_summary['File Name'] = file_name

        stats_location = ['Mean', 'Median', 'StdDev', 'SEM']

        if hasattr(self, 'shifts_with_stats'):
            pcnt_no_shift = [np.count_nonzero(np.isnan(self.mean_shifts[combo_number])) /
                             self.mean_shifts[combo_number].shape[0] * 100 for combo_number, combo in enumerate(self.channel_combos)]
            for combo_number, combo in enumerate(self.channel_combos):
                self.file_data_summary[f'Ch{combo[0] + 1}-Ch{combo[1] + 1} Pcnt No Shifts'] = pcnt_no_shift[combo_number]
                for ind, stat in enumerate(stats_location):
                    self.file_data_summary[f'Ch{combo[0] + 1}-Ch{combo[1] + 1} {stat} Shift'] = self.shifts_with_stats[combo_number][ind + 1]

        if hasattr(self, 'peak_widths_with_stats'):
            # using widths, but because these are all assigned together it applies to all peak properties
            pcnt_no_peaks = [np.count_nonzero(np.isnan(self.mean_peak_widths[channel])) /
                             self.mean_peak_widths[channel].shape[0] * 100 for channel in range(self.num_channels)]
            for channel in range(self.num_channels):
                self.file_data_summary[f'Ch {channel + 1} Pcnt No Peaks'] = pcnt_no_peaks[channel]
                for ind, stat in enumerate(stats_location):
                    self.file_data_summary[f'Ch {channel + 1} {stat} Peak Width'] = self.peak_widths_with_stats[channel][ind + 1]
                    self.file_data_summary[f'Ch {channel + 1} {stat} Peak Max'] = self.peak_maxs_with_stats[channel][ind + 1]
                    self.file_data_summary[f'Ch {channel + 1} {stat} Peak Min'] = self.peak_mins_with_stats[channel][ind + 1]
                    self.file_data_summary[f'Ch {channel + 1} {stat} Peak Amp'] = self.peak_amps_with_stats[channel][ind + 1]
                    self.file_data_summary[f'Ch {channel + 1} {stat} Peak Rel Amp'] = self.peak_relamp_with_stats[channel][ind + 1]

        return self.file_data_summary