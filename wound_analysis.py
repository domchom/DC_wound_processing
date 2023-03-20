import os
import sys
import timeit
import napari
import datetime
import tifffile
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
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

    #open the images and create the ellipses. Save coordinates of the ellipses for each movie
    all_images = convert_images(folder_path)
    all_images_line_coords = create_ellipses(all_images,line_length,num_lines,bin_num)

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

def convert_images(folder_path):
    """
    Reads all TIFF images from the folder specified in the object's folder_path attribute and standardizes their dimensions
    by reshaping them into a 5D numpy array of shape (num_frames, num_slices, num_channels, height, width). 
    
    If the TIFF file has multiple slices, the images are max projected along the slice axis. The resulting images are stored 
    in the object's images attribute, which is a dictionary with filename paths as keys and the corresponding numpy arrays as values.

    Returns:
    A 5D numpy array of shape (num_frames, num_slices, num_channels, height, width)
    """
    # dict to store all image {filename_path: [np array as img]}
    images = {}

    for filename_path in os.listdir(folder_path):
        if filename_path.endswith('.tif'):

            image = tifffile.imread(filename_path)

            # standardize image dimensions
            with tifffile.TiffFile(filename_path) as tif_file:
                metadata = tif_file.imagej_metadata
            num_channels = metadata.get('channels', 1)
            num_slices = metadata.get('slices', 1)
            num_frames = metadata.get('frames', 1)
            image = image.reshape(num_frames,
                                    num_slices,
                                    num_channels, 
                                    image.shape[-1],  # columns
                                    image.shape[-2])  # rows

            # max project image stack if num_slices > 1
            if num_slices > 1:
                print(f'Max projecting image stack')
                image = np.max(image, axis=1)
                num_slices = 1
                image = image.reshape(num_frames,
                                                num_slices,
                                                num_channels,
                                                image.shape[-1],  # columns
                                                image.shape[-2])  # rows

            images[filename_path] = image

    return images

def create_ellipses(all_images,line_length,num_lines,bin_num):
    """
    For each image in the provided dictionary, opens a napari viewer and prompts the user to draw an ellipse
    on the image.
    The ellipse dimensions are then saved to the dictionary under the corresponding filename key, as a list 
    of line coordinates.
    If the key already has a value, the new line coordinates are appended to the existing list.

    Args:
    all_images: A dictionary containing filename keys and np array img values.

    Returns:
    A dictionary where each key is a filename and the corresponding value is a list of line coordinates for 
        the ellipse drawn on that image. [np array, line coords]
    """
    for filename_path in all_images:
        # the user will create the ellipse in napari for the given image
        last_shape = user_define_ellipse(filename_path)
        line_coords = calc_ellipse_lines(last_shape,line_length,num_lines,bin_num)
        if filename_path in all_images:
            value = all_images[filename_path]
            all_images[filename_path] = [value, line_coords]

    return all_images

def user_define_ellipse(filename_path):
    """
    Opens an image specified by the image path, displays it in a napari viewer, and allows the user to draw an ellipse.
    The function then saves the dimensions of the ellipse and closes the viewer.

    Args:
    filename_path (str): Path to the image file to be opened.

    Returns:
    list: A list of line coordinates, where each line is a list of x and y coordinates.
    """
    # asking the user to identify the ring of interest
    with napari.gui_qt():
        filename = filename_path.split('/')[-1]
        viewer = napari.Viewer(title=f'Create ellipse for {filename}. Close window to save ellipse')
        viewer.open(filename_path)

        #ellipse = np.array([[156, 165], [156, 368], [373, 368], [373, 165]])
        shapes_layer = viewer.add_shapes()

        @viewer.bind_key('s')
        def save_and_close(viewer):
            last_shape = viewer.layers['Shapes'].data[-1]
            viewer.window.close()

            return last_shape
        
        napari.run()
    
        return save_and_close(viewer) #return the ellipse coords
            
def calc_ellipse_lines(last_shape,line_length,num_lines,bin_num):
    if last_shape is not None:
        # Extract x coords and y coords of the ellipse as column vectors
        X = last_shape[:,1:]
        Y = last_shape[:,0:1]

        # Formulate and solve the least squares problem ||Ax - b ||^2
        A = np.hstack([X**2, X * Y, Y**2, X, Y])
        b = np.ones_like(X)
        x = np.linalg.lstsq(A, b)[0].squeeze()
        
        '''#plot the original datapoints
        plt.legend(loc='upper right', fontsize='small', ncol=3)
        plt.tight_layout()
        plt.scatter(X, Y, label='Data Points')'''

        #plot the fitted ellipse
        x_coord = np.linspace(0,511,num_lines)
        y_coord = np.linspace(0,511,num_lines)
        X_coord, Y_coord = np.meshgrid(x_coord, y_coord)
        Z_coord = x[0] * X_coord ** 2 + x[1] * X_coord * Y_coord + x[2] * Y_coord**2 + x[3] * X_coord + x[4] * Y_coord
        plt.contour(X_coord, Y_coord, Z_coord, levels=[1], colors=('r'), linewidths=2)                   

        plt.show()


        '''
        # Define the rectangle's coordinates
        x1, y1 = last_shape[0][0], last_shape[0][1]
        x2, y2 = last_shape[2][0], last_shape[2][1]

        # Calculate the center and ratio
        center = [(x1 + x2) / 2, (y1 + y2) / 2]
        ratio = abs(y2 - y1) / abs(x2 - x1)

        # Define the two ellipses
        small_ellipse_width = 2
        small_ellipse_height = small_ellipse_width * ratio
        large_ellipse_width = small_ellipse_width + (line_length * 2)
        large_ellipse_height = small_ellipse_height + (line_length * 2)

        small_ellipse = Ellipse(xy=(center[0], center[1]), width=small_ellipse_width, height=small_ellipse_height, angle=0)
        large_ellipse = Ellipse(xy=(center[0], center[1]), width=large_ellipse_width, height=large_ellipse_height, angle=0)



        theta = np.linspace(0, 2 * np.pi, num_lines)
        points = np.stack([large_ellipse.center[0] + large_ellipse.width/2*np.cos(theta),
                            large_ellipse.center[1] + large_ellipse.height/2*np.sin(theta)], axis=1)

        # Loop over each point on the large ellipse and calculate the shortest distance to the small ellipse and line segment
        line_coords = [[np.linspace(x0, x1, bin_num), np.linspace(y0, y1, bin_num)]
                            for i, point in enumerate(points)
                            if (distance := np.linalg.norm(small_ellipse.center - point) - small_ellipse_width / 2) >= 0
                            for theta in [np.arctan2(point[1] - large_ellipse.center[1], point[0] - large_ellipse.center[0])]
                            for x0, y0, x1, y1 in [[point[0], point[1],small_ellipse.center[0] + small_ellipse_width / 2 * np.cos(theta),
                                small_ellipse.center[1] + small_ellipse_height / 2 * np.sin(theta)]]]
        '''
    
    else:
        print('No ellipse has been drawn yet')

    line_coords = np.array(line_coords)

    return line_coords         

if __name__ == '__main__':
    main()