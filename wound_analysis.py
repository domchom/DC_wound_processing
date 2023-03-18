import os
import sys
import timeit
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from wound_analysis_mods.napari_wound_analysis import EllipseCreation
from wound_analysis_mods.processor_wound_analysis import ImageProcessor
from wound_analysis_mods.gui_wound_analysis import BaseGUI

#set the behavior for two types of errors: divide-by-zero and invalid arithmetic operations 
np.seterr(divide='ignore', invalid='ignore') 

def main():
    '''** GUI Window and sanity checks'''
    # make GUI object and display the window
    gui = BaseGUI()
    gui.mainloop()
    # get GUI parameters
    num_lines = gui.num_lines
    line_length = gui.line_length
    bin_num = gui.bin_num
    folder_path = gui.folder_path
    frame_rate = gui.frame_rate
    Ch1 = gui.Ch1
    Ch2 = gui.Ch2
    plot_mean_CCFs = gui.plot_mean_CCFs
    plot_mean_peaks = gui.plot_mean_peaks
    plot_ind_CCFs = gui.plot_ind_CCFs
    plot_ind_peaks = gui.plot_ind_peaks
    plot_ref_fig = gui.plot_ref_fig
    plot_linescan_movie = gui.plot_linescan_movie
    pixel_size = gui.pixel_size
    
    # identify and report errors in GUI input
    errors = []
    if gui.bin_num < 11:
        errors.append("Bin number must be greater 11")

    if len(gui.folder_path) < 1:
        errors.append("You didn't enter a directory to analyze")
    
    if frame_rate == '':
        frame_rate = 1
    try:
        frame_rate = float(frame_rate)
    except ValueError:
        errors.append("Frame rate must be a number")

    if pixel_size == '':
        pixel_size = 1
    try:
        pixel_size = float(pixel_size)
    except ValueError:
        errors.append("Pixel size must be a number")

    if len(errors) >= 1:
        print("Error Log:")
        for count, error in enumerate(errors):
            print(count, ":", error)
        sys.exit("Please fix errors and try again.")

    if Ch1 == '':
        Ch1 = 'Ch1'

    if Ch1 == '':
        Ch1 = 'Ch2'

    # make dictionary of parameters for log file use
    log_params = {"Number of lines": num_lines,
                "Line length": line_length,
                "Frame Rate" : frame_rate,
                "Pixel size": pixel_size,
                "Base Directory": folder_path,
                "Number of bins per line": bin_num,
                "Plot Summary CCFs": plot_mean_CCFs,
                "Plot Summary Peaks": plot_mean_peaks,
                "Plot Individual CCFs": plot_ind_CCFs,
                "Plot Individual Peaks": plot_ind_peaks,
                "Files Processed": [],
                "Files Not Processed": [],
                'Plotting errors': []
                }
        
    ''' ** housekeeping functions ** '''
    def make_log(directory, logParams):
        '''
        Convert dictionary of parameters to a log file and save it in the directory
        '''
        now = datetime.datetime.now()
        logPath = os.path.join(
            directory, f"0_log-{now.strftime('%Y%m%d%H%M')}.txt")
        logFile = open(logPath, "w")
        logFile.write("\n" + now.strftime("%Y-%m-%d %H:%M") + "\n")
        for key, value in logParams.items():
            logFile.write('%s: %s\n' % (key, value))
        logFile.close()

    file_names = filelist = [fname for fname in os.listdir(
        folder_path) if fname.endswith('.tif') and not fname.startswith('.')]


    ''' ** Main Workflow ** '''
    # performance tracker
    start = timeit.default_timer()

    # create main save path
    now = datetime.datetime.now()
    os.chdir(folder_path)
    main_save_path = os.path.join(
        folder_path, f"!wound_processing-{now.strftime('%Y%m%d%H%M')}")

    # create directory if it doesn't exist
    if not os.path.exists(main_save_path):
        os.makedirs(main_save_path)

    # empty list to fill with summary data for each file
    summary_list = []
    # column headers to use with summary data during conversion to dataframe
    col_headers = []

    # creating the ellipses
    print('Create Ellipses')

    all_images_line_coords = EllipseCreation(
        folder_path=folder_path, num_lines=num_lines, line_length=line_length, bin_num=bin_num)
    all_images_line_coords = all_images_line_coords.create_ellipses()


    # processing movies
    with tqdm(total=len(file_names)) as pbar:
        pbar.set_description('Files processed:')
        for file_name in file_names:
            print('******'*10)
            print(f'Processing {file_name}...')

            # name without the extension
            name_wo_ext = file_name.rsplit(".", 1)[0]

            # create a subfolder within the main save path with the same name as the image file
            im_save_path = os.path.join(main_save_path, name_wo_ext)
            if not os.path.exists(im_save_path):
                os.makedirs(im_save_path)

            # Initialize the processor
            processor = ImageProcessor(filename=file_name, im_save_path=im_save_path,
                                    img=all_images_line_coords[file_name][0],
                                    line_coords=all_images_line_coords[file_name][1],
                                    line_length=line_length,
                                    Ch1=Ch1,
                                    Ch2=Ch2,
                                    frame_rate = frame_rate,
                                    pixel_size = pixel_size)

            # if file is not skipped, log it and continue
            log_params['Files Processed'].append(f'{file_name}')

            # calculate the population signal properties
            processor.calc_ind_peak_props()
            if processor.num_channels > 1:
                processor.calc_indv_CCFs()

            # calculate the mean signal properties per frame
            processor.calculate_mean_line_values_per_frame()

            # calculate the population signal properties
            processor.calc_mean_peak_props()
            if processor.num_channels > 1:
                processor.calc_mean_CCF()

            # plotting
            if plot_linescan_movie:
                processor.create_mean_linescan_per_frame_movie()

            if plot_ref_fig:
                processor.plot_reference_figure()
                plt.savefig(f'{im_save_path}/{name_wo_ext}_ref.png')

            if plot_ind_peaks:
                ind_peak_plots = processor.plot_ind_peak_props()
                ind_peak_path = os.path.join(
                    im_save_path, 'Individual_peak_plots')
                if not os.path.exists(ind_peak_path):
                    os.makedirs(ind_peak_path)
                for plot_name, plot in ind_peak_plots.items():
                    plot.savefig(f'{ind_peak_path}/{plot_name}.png')

            if plot_ind_CCFs:
                if processor.num_channels == 1:
                    log_params[
                        'Miscellaneous'] = f'CCF plots were not generated for {file_name} because the image only has one channel'
                ind_ccf_plots = processor.plot_ind_ccfs()
                ind_ccf_path = os.path.join(
                    im_save_path, 'Individual_CCF_plots')
                if not os.path.exists(ind_ccf_path):
                    os.makedirs(ind_ccf_path)
                for plot_name, plot in ind_ccf_plots.items():
                    plot.savefig(f'{ind_ccf_path}/{plot_name}.png')

            if plot_mean_CCFs:
                mean_ccf_plots = processor.plot_mean_CCF()
                mean_ccf_path = os.path.join(
                    im_save_path, 'Mean_CCF_plots')
                if not os.path.exists(mean_ccf_path):
                    os.makedirs(mean_ccf_path)
                for plot_name, plot in mean_ccf_plots.items():
                    plot.savefig(f'{mean_ccf_path}/{plot_name}.png')

            if plot_mean_peaks:
                mean_peak_plots = processor.plot_mean_peak_props()
                mean_peaks_path = os.path.join(
                    im_save_path, 'Mean_peak_plots')
                if not os.path.exists(mean_peaks_path):
                    os.makedirs(mean_peaks_path)
                for plot_name, plot in mean_peak_plots.items():
                    plot.savefig(f'{mean_peaks_path}/{plot_name}.png')

            # Summarize the data for current image as dataframe, and save as .csv
            im_measurements_df = processor.organize_measurements()
            im_measurements_df.to_csv(
                f'{im_save_path}/{name_wo_ext}_measurements.csv', index=False)

            # generate summary data for current image
            im_summary_dict = processor.summarize_image(
                file_name=file_name)

            # populate column headers list with keys from the measurements dictionary
            for key in im_summary_dict.keys():
                if key not in col_headers:
                    col_headers.append(key)

            # append summary data to the summary list
            summary_list.append(im_summary_dict)

            # useless progress bar to force completion of previous bars
            with tqdm(total=10, miniters=1) as dummy_pbar:
                dummy_pbar.set_description('cleanup:')
                for i in range(10):
                    dummy_pbar.update(1)

            pbar.update(1)

        # create dataframe from summary list
        summary_df = pd.DataFrame(summary_list, columns=col_headers)
        summary_df.to_csv(f'{main_save_path}/summary.csv', index=False)

        end = timeit.default_timer()
        log_params["Time Elapsed"] = f"{end - start:.2f} seconds"
        # log parameters and errors
        make_log(main_save_path, log_params)
        print('Done with Script!')

if __name__ == '__main__':
    main()