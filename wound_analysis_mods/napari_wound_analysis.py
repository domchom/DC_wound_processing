import numpy as np
from matplotlib.patches import Ellipse
import tifffile
import os
import napari

class EllipseCreation:

    def __init__(self, folder_path, num_lines, line_length, bin_num):
        self.folder_path = folder_path
        self.num_lines = num_lines
        self.line_length = line_length
        self.bin_num = bin_num

        self.all_images = self.convert_images()

    def convert_images(self):
        """
        Reads all TIFF images from the folder specified in the object's folder_path attribute and standardizes their dimensions
        by reshaping them into a 5D numpy array of shape (num_frames, num_slices, num_channels, height, width). 
        
        If the TIFF file has multiple slices, the images are max projected along the slice axis. The resulting images are stored 
        in the object's images attribute, which is a dictionary with filename paths as keys and the corresponding numpy arrays as values.

        Returns:
        A 5D numpy array of shape (num_frames, num_slices, num_channels, height, width)
        """
        # dict to store all image {filename_path: [np array as img]}
        self.images = {}

        for filename_path in os.listdir(self.folder_path):
            if filename_path.endswith('.tif'):

                self.image = tifffile.imread(filename_path)

                # standardize image dimensions
                with tifffile.TiffFile(filename_path) as tif_file:
                    metadata = tif_file.imagej_metadata
                self.num_channels = metadata.get('channels', 1)
                self.num_slices = metadata.get('slices', 1)
                self.num_frames = metadata.get('frames', 1)
                self.image = self.image.reshape(self.num_frames,
                                                self.num_slices,
                                                self.num_channels, 
                                                self.image.shape[-1],  # columns
                                                self.image.shape[-2])  # rows

                # max project image stack if num_slices > 1
                if self.num_slices > 1:
                    print(f'Max projecting image stack')
                    self.image = np.max(self.image, axis=1)
                    self.num_slices = 1
                    self.image = self.image.reshape(self.num_frames,
                                                    self.num_slices,
                                                    self.num_channels,
                                                    self.image.shape[-1],  # columns
                                                    self.image.shape[-2])  # rows

                self.images[filename_path] = self.image

        return self.images

    def create_ellipses(self):
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
        for filename in self.all_images:
            # the user will create the ellipse in napari for the given image
            line_coords = self.user_define_ellipse(filename)
            if filename in self.all_images:
                value = self.all_images[filename]
                self.all_images[filename] = [value, line_coords]

        return self.all_images

    def save_shape(self, viewer):
        last_shape = viewer.layers['Shapes'].data[-1]
        if last_shape is not None:
            # Define the rectangle's coordinates
            x1, y1 = last_shape[0][0], last_shape[0][1]
            x2, y2 = last_shape[2][0], last_shape[2][1]

            # Calculate the center and ratio
            center = [(x1 + x2) / 2, (y1 + y2) / 2]
            ratio = abs(y2 - y1) / abs(x2 - x1)

            # Define the two ellipses
            small_ellipse_width = 1
            small_ellipse_height = small_ellipse_width * ratio
            large_ellipse_width = self.line_length * 2 + small_ellipse_width
            large_ellipse_height = self.line_length * 2 + small_ellipse_height

            small_ellipse = Ellipse(xy=(center[0], center[1]), width=small_ellipse_width, height=small_ellipse_height, angle=0)
            large_ellipse = Ellipse(xy=(center[0], center[1]), width=large_ellipse_width, height=large_ellipse_height, angle=0)

            # Check if small ellipse is inside large ellipse
            if large_ellipse.contains_point(small_ellipse.center):

                # Create an array of points on the large ellipse
                theta = np.linspace(0, 2 * np.pi, self.num_lines)
                points = np.stack([large_ellipse.center[0] + large_ellipse.width/2*np.cos(theta),
                                   large_ellipse.center[1] + large_ellipse.height/2*np.sin(theta)], axis=1)

                # Loop over each point on the large ellipse and calculate the shortest distance to the small ellipse and line segment
                self.line_coords = [[np.linspace(x0, x1, self.bin_num), np.linspace(y0, y1, self.bin_num)]
                                    for i, point in enumerate(points)
                                    if (distance := np.linalg.norm(small_ellipse.center - point) - small_ellipse_width / 2) >= 0
                                    for theta in [np.arctan2(point[1] - large_ellipse.center[1], point[0] - large_ellipse.center[0])]
                                    for x0, y0, x1, y1 in [[point[0], point[1],small_ellipse.center[0] + small_ellipse_width / 2 * np.cos(theta),
                                        small_ellipse.center[1] + small_ellipse_height / 2 * np.sin(theta)]]]

        else:
            print('No ellipse has been drawn yet')

        self.line_coords = np.array(self.line_coords)

        return self.line_coords

    def user_define_ellipse(self, filename_path):
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
            ellipse_data = np.array(
                [[350, 190], [350, 310], [190, 310], [190, 190]])
            viewer = napari.Viewer()
            viewer.open(filename_path)

            ellipse_layer = viewer.add_shapes(
                data=[ellipse_data],
                shape_type='ellipse',
                edge_color='red',
                edge_width=2,
                face_color='transparent',
            )

            self.save_shape(viewer)

            napari.run()

        return self.save_shape(viewer)