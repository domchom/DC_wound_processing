import sys
import tkinter as tk
from tkinter import ttk
from tkinter.filedialog import askdirectory

class BaseGUI(tk.Tk):
    def __init__(self):
        super().__init__()

        # configure root window
        self.title("Define your analysis parameters")
        self.geometry("680x310")
        
        #sets number of columns in the main window
        self.columnconfigure(0, weight = 1)
        self.columnconfigure(1, weight = 1)
        self.columnconfigure(2, weight = 1)

        # define variable types for the different widget field
        self.num_lines = tk.IntVar()
        self.num_lines.set(36)
        self.line_length = tk.IntVar()
        self.line_length.set(150)
        self.bin_num = tk.IntVar()
        self.bin_num.set(150)
        self.frame_rate = tk.StringVar()
        self.pixel_size = tk.StringVar()
        self.Ch1 = tk.StringVar()
        self.Ch2 = tk.StringVar()

        self.plot_ref_fig = tk.BooleanVar()
        self.plot_ref_fig.set(True)

        self.plot_linescan_movie = tk.BooleanVar()
        self.plot_linescan_movie.set(True)

        self.plot_mean_CCFs = tk.BooleanVar()
        self.plot_mean_CCFs.set(False)

        self.plot_mean_peaks = tk.BooleanVar()
        self.plot_mean_peaks.set(False)

        self.plot_ind_CCFs = tk.BooleanVar()
        self.plot_ind_CCFs.set(False)

        self.plot_ind_peaks = tk.BooleanVar()
        self.plot_ind_peaks.set(False)

        self.folder_path = tk.StringVar()

        # file path selection widget
        self.file_path_entry = ttk.Entry(self, textvariable = self.folder_path)
        self.file_path_entry.grid(row = 0, column = 0, padx = 10, sticky = 'E')
        self.file_path_button = ttk.Button(self, text = 'Select folder')

        # make a default path
        self.folder_path.set('/Users/domchom/Desktop/example_movies')
        self.file_path_button['command'] = self.get_folder_path
        self.file_path_button.grid(row = 0, column = 1, padx = 10, sticky = 'W')        

        # Number of lines selection widget
        self.num_line_entry = ttk.Entry(self, width = 3, textvariable = self.num_lines)
        self.num_line_entry.grid(row = 1, column = 0, padx = 10, sticky = 'E')
        self.num_line_label = ttk.Label(self, text = 'Number of Lines')
        self.num_line_label.grid(row = 1, column = 1, padx = 10, sticky = 'W')

        # Line length selection widget  
        self.line_length_entry = ttk.Entry(self, width = 3, textvariable = self.line_length)
        self.line_length_entry.grid(row = 2, column = 0, padx = 10, sticky = 'E')
        self.line_length_label = ttk.Label(self, text = 'Line Length (pixels)')
        self.line_length_label.grid(row = 2, column = 1, padx = 10, sticky = 'W')

        # Number of bins selection widget  
        self.bin_num_entry = ttk.Entry(self, width = 3, textvariable = self.bin_num)
        self.bin_num_entry.grid(row = 3, column = 0, padx = 10, sticky = 'E')
        # create ACF peak threshold label text
        self.bin_num_label = ttk.Label(self, text = 'Number of bins')
        self.bin_num_label.grid(row = 3, column = 1, padx = 10, sticky = 'W')

        # Ch1 entry widget
        self.Ch1_entry = ttk.Entry(self, textvariable = self.Ch1)
        self.Ch1_entry.grid(row = 4, column = 0, padx = 10, sticky = 'E')
        self.Ch1_label = ttk.Label(self, text = 'Channel 1')
        self.Ch1_label.grid(row = 4, column = 1, padx = 10, sticky = 'W')
        
        # Ch2 entry widget
        self.Ch2_entry = ttk.Entry(self, textvariable = self.Ch2)
        self.Ch2_entry.grid(row = 5, column = 0, padx = 10, sticky = 'E')
        self.Ch2_label = ttk.Label(self, text = 'Channel 2')
        self.Ch2_label.grid(row = 5, column = 1, padx = 10, sticky = 'W')

        # frame rate entry widget
        self.frame_rate_entry = ttk.Entry(self, textvariable = self.frame_rate)
        self.frame_rate_entry.grid(row = 6, column = 0, padx = 10, sticky = 'E')
        self.frame_rate_label = ttk.Label(self, text = 'Frame rate (frames per second)')
        self.frame_rate_label.grid(row = 6, column = 1, padx = 10, sticky = 'W')

        # frame rate entry widget
        self.pixel_size_entry = ttk.Entry(self, textvariable = self.pixel_size)
        self.pixel_size_entry.grid(row = 7, column = 0, padx = 10, sticky = 'E')
        self.pixel_size_label = ttk.Label(self, text = 'Pixel size (microns)')
        self.pixel_size_label.grid(row = 7, column = 1, padx = 10, sticky = 'W')

        # create checkbox for plotting summary CCFs
        self.plot_mean_CCFs_checkbox = ttk.Checkbutton(self, variable = self.plot_mean_CCFs)
        self.plot_mean_CCFs_checkbox.grid(row = 8, column = 0, padx = 10, sticky = 'E')
        self.plot_mean_CCFs_label = ttk.Label(self, text = 'Plot summary CCFs')
        self.plot_mean_CCFs_label.grid(row = 8, column = 1, padx = 10, sticky = 'W')

        # create checkbox for plotting summary peaks
        self.plot_mean_peaks_checkbox = ttk.Checkbutton(self, variable = self.plot_mean_peaks)
        self.plot_mean_peaks_checkbox.grid(row = 9, column = 0, padx = 10, sticky = 'E')
        self.plot_mean_peaks_label = ttk.Label(self, text = 'Plot summary peaks')
        self.plot_mean_peaks_label.grid(row = 9, column = 1, padx = 10, sticky = 'W')

        # create checkbox for plotting individual CCFs
        self.plot_ind_CCFs_checkbox = ttk.Checkbutton(self, variable = self.plot_ind_CCFs)
        self.plot_ind_CCFs_checkbox.grid(row = 8, column = 2, padx = 10, sticky = 'E')
        self.plot_ind_CCFs_label = ttk.Label(self, text = 'Plot individual CCFs')
        self.plot_ind_CCFs_label.grid(row = 8, column = 3, padx = 10, sticky = 'W')

        # create checkbox for plotting individual peaks
        self.plot_ind_peaks_checkbox = ttk.Checkbutton(self, variable = self.plot_ind_peaks)
        self.plot_ind_peaks_checkbox.grid(row = 9, column = 2, padx = 10, sticky = 'E')
        self.plot_ind_peaks_label = ttk.Label(self, text = 'Plot individual peaks')
        self.plot_ind_peaks_label.grid(row = 9, column = 3, padx = 10, sticky = 'W')
        
        # create checkbox for plotting reference figure
        self.plot_ref_fig_checkbox = ttk.Checkbutton(self, variable = self.plot_ref_fig)
        self.plot_ref_fig_checkbox.grid(row = 10, column = 0, padx = 10, sticky = 'E')
        self.plot_ref_fig_label = ttk.Label(self, text = 'Plot reference figure')
        self.plot_ref_fig_label.grid(row = 10, column = 1, padx = 10, sticky = 'W')

        # create checkbox for plotting linescan movies
        self.plot_linescan_movie_checkbox = ttk.Checkbutton(self, variable = self.plot_linescan_movie)
        self.plot_linescan_movie_checkbox.grid(row = 10, column = 2, padx = 10, sticky = 'E')
        self.plot_linescan_movie_label = ttk.Label(self, text = 'Plot Linescan movie')
        self.plot_linescan_movie_label.grid(row = 10, column = 3, padx = 10, sticky = 'W')

        # create start button
        self.start_button = ttk.Button(self, text = 'Start analysis')
        self.start_button['command'] = self.start_analysis
        self.start_button.grid(row = 11, column = 3, padx = 10, sticky = 'E')

        # create cancel button
        self.cancel_button = ttk.Button(self, text = 'Cancel')
        self.cancel_button['command'] = self.cancel_analysis
        self.cancel_button.grid(row = 11, column = 1, padx = 10, sticky = 'W')

    def get_folder_path(self):
        self.folder_path.set(askdirectory())

    def cancel_analysis(self):
        sys.exit('You have cancelled the analysis')
    
    def start_analysis(self):
        # get the values stored in the widget
        self.num_lines = self.num_lines.get()
        self.line_length = self.line_length.get()
        self.bin_num = self.bin_num.get()
        self.Ch1 = self.Ch1.get()
        self.Ch2 = self.Ch2.get()
        self.frame_rate = self.frame_rate.get()
        self.pixel_size = self.pixel_size.get()
        self.plot_mean_CCFs = self.plot_mean_CCFs.get()
        self.plot_mean_peaks = self.plot_mean_peaks.get()
        self.plot_ind_CCFs = self.plot_ind_CCFs.get()
        self.plot_ind_peaks = self.plot_ind_peaks.get()
        self.folder_path = self.folder_path.get()
        
        # destroy the widget
        self.destroy()