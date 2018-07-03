import numpy as np


class MnistDbReader:
    def __init__(self, images_file_path, labels_file_path, max_nb_data=None):
        self.images_file = open(images_file_path, "r")
        self.labels_file = open(labels_file_path, "r")
        self.index = 0
        self.labels = []
        self.number_of_labels = 0
        self.images = []
        self.number_of_images = 0
        self.big_endian_uint32 = ">u4"
        self.big_endian_uint8 = ">u1"
        self.max_nb_data = max_nb_data

    def initialize(self):
        self._initialize_images_reader()
        self._initialize_labels_reader()

    def _initialize_images_reader(self):
        magic_number = np.fromfile(self.images_file, dtype=self.big_endian_uint32, count=1)[0]
        number_of_images_to_read = np.fromfile(self.images_file, dtype=self.big_endian_uint32, count=1)[0]
        rows_per_image = np.fromfile(self.images_file, dtype=self.big_endian_uint32, count=1)[0]
        columns_per_image = np.fromfile(self.images_file, dtype=self.big_endian_uint32, count=1)[0]
        if self.max_nb_data:
            number_of_images_to_read = self.max_nb_data
        self.number_of_images = number_of_images_to_read
        for i in range(number_of_images_to_read):
            image = np.fromfile(self.images_file, dtype=self.big_endian_uint8, count=columns_per_image*rows_per_image)
            self.images.append(image)
        print("Images file magic number: {}".format(magic_number))
        print("Number of images read: {}".format(number_of_images_to_read))
        print("Rows per image: {}".format(rows_per_image))
        print("Columns per image: {}".format(columns_per_image))
        self.images_file.close()

    def _initialize_labels_reader(self):
        magic_number = np.fromfile(self.labels_file, dtype=self.big_endian_uint32, count=1)[0]
        number_of_labels_to_read = np.fromfile(self.labels_file, dtype=self.big_endian_uint32, count=1)[0]
        if self.max_nb_data:
            number_of_labels_to_read = self.max_nb_data
        self.labels = np.fromfile(self.labels_file, dtype=self.big_endian_uint8, count=number_of_labels_to_read)
        self.number_of_labels = number_of_labels_to_read
        print("Labels file magic number: {}".format(magic_number))
        print("Number of labels read: {}".format(number_of_labels_to_read))
        self.labels_file.close()

    def _increment_index(self):
        self.index += 1

    def get_next_tuple(self):
        if (self.index >= self.number_of_images
                or self.index >= self.number_of_labels):
            return None, None
        else:
            label = self.labels[self.index]
            image = self.images[self.index]
            self._increment_index()
            return label, image
