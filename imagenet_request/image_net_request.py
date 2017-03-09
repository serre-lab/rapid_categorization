# Written by Ben and Michele Jan 24 2017

import xml.etree.ElementTree as ET
import os
import pickle
import random
import requests
from PIL import Image
import numpy as np
import math
from multiprocessing import Pool

class ImageNet:
    """
    A class which will open and parse the ImageNet XML file, download images,
    and balance the number of images per class.
    """
    def __init__(self, xml_path, output_path, class_list_file, tot_num_images,
                 url_list_path):
        """
        foo = ImageNet('./ImageNet_structure.xml', '/media/data/nsf_levels_michelecopy/test_1', './class_list_file_TEST.txt', 100, './fall11_urls.txt')
        xml_path = '/home/michele/emtcat/ImageNet_structure.xml'
        output_path = '/media/data/nsf_levels_michelecopy/<set_num>'
        class_list_file = '/home/michele/emtcat/class_list_file_TEST.txt'
        url_list_path = '/home/michele/emtcat/fall11urls.txt'

        Creates a new ImageNet.
        Creates folders within the output path for each of the classes in
        class_list.
        Takes in a list of class names to find Word Net IDs for in ImageNet.
        [What it needs to do]: Select and download a balanced set of images
        with a similarly balanced representation of synsets.
        :param xml_path: Path to the ImageNet XML file.
        :param output_path: Path to the output folder of balanced images.
                (i.e. ~/experiment_name/6)
        :param class_list_file: List of class names as a .txt file, with
                categories [ex. class=mammals, categories=cat,dog,human].
        :param tot_num_images: Total number of images for the set.
        :param url_list_path: Path to the WordNet ID to URL map.
        """

        self.output_path = output_path
        self.xml_path = xml_path
        self.class_list_file = class_list_file
        self.tot_num_images = tot_num_images
        self.url_list_path = url_list_path

        # Get root for xml file
        xml_root = ET.parse(xml_path).getroot()[1]
        print 'Using '+ xml_root.attrib['words']

        self.parsed_xml = xml_root

        # Open class_list_file and read out classes
        with open(class_list_file, 'r') as f:
            class_list = [line.split() for line in f]

        # Determine categories for each class.
        # For example, mammals and nonmammals are categories, while
        # cats and dogs are classes within mammals.
        self.categories = class_list[0]
        self.classes = class_list[1:]
        if len(self.classes) != len(self.categories):
            raise Exception("invalid format")

        # Read URL file into memory, with the form:
        # {
        #   <WordNet ID>: [
        #     [<Image ID>, <URL>],
        #     ...
        #   ],
        #   ...
        # }
        url_map = {}
        with open(self.url_list_path, 'r') as file:
            for line in file:
                wnid_and_url = line.split()
                if len(wnid_and_url) != 2:
                    continue

                wnid_and_image = wnid_and_url[0].split('_')
                if len(wnid_and_image) != 2:
                    continue

                if wnid_and_image[0] in url_map:
                    url_map[wnid_and_image[0]].append([wnid_and_image[1], wnid_and_url[1]])
                else:
                    url_map[wnid_and_image[0]] = [[wnid_and_image[1], wnid_and_url[1]]]

        # Create wnid map from classes in class_list, of the form:
        # {
        #   <category>: {
        #     <class>: [<WordNet ID>, ...]
        #     ...
        #   },
        #   ...
        # }
        self.wnid_map = {}
        # Create a wnid to image and URL map, of the form:
        # {
        #   <WordNet ID>: [[<Image ID>, <URL>], ...]
        #   ...
        # }
        self.wnid_url_map = {}

        for i, category in enumerate(self.categories):
            # Create structure of wnid_map, and find all the ids.
            self.wnid_map[category] = {}
            for class_entry in self.classes[i]:
                if (not ':' in class_entry) or (len(class_entry.split(':')) != 2):
                    raise Exception("invalid file format")
                class_name = class_entry.split(":")[0]
                wnid_of_class = class_entry.split(":")[1]
                # Breadth first search for all sub WordNet IDs related to class WordNet ID
                wnid_list_of_class = self.all_wnids_of_class(xml_root, wnid_of_class)

                self.wnid_map[category][class_name] = []
                for wnid in wnid_list_of_class:
                    wnid_number = wnid.attrib['wnid']
                    if not wnid_number in url_map:
                        continue

                    self.wnid_map[category][class_name].append(wnid_number)
                    self.wnid_url_map[wnid_number] = url_map[wnid_number]

        # Create file structure.
        self.image_set_path = os.path.join(self.output_path, 'image_set')
        self.raw_images_path = os.path.join(self.output_path, 'raw_images')
        self.serialized_wnid_map_path = os.path.join(self.output_path, 'wnid_map.p')
        self.serialized_wnid_url_map_path = \
            os.path.join(self.output_path, 'wnid_url_map.p')

        directories_to_create = [self.image_set_path, self.raw_images_path]
        for category in self.categories:
            category_directory = os.path.join(self.raw_images_path, category)
            directories_to_create += [category_directory]
            # Look into wnid_map[category] for class names associated with the
            # category, then make directories for those classes within the
            # category directory in raw_images directory.
            for class_name in self.wnid_map[category].keys():
                directories_to_create += [os.path.join(category_directory, class_name)]

        for directory in directories_to_create:
            try:
                os.mkdir(directory)
            except OSError:
                print 'Directory already exists: {0}'.format(directory)
                raw_input('Press anything to continue with the existing directory...')

    def all_wnids_of_class(self, xml_root, wnid_of_class):
        """
        Get WordNet IDs for categories in class_list
        Ex. category = mammals, class and wnid = cat:n02127808
        subwnids are below category wnids.
        ++Called by __init__++
        :param xml_root: Path to the ImageNet XML file.
        :param wnid_of_class: WordNet ID representing class of interest.
        :return: List of WordNet IDs that exist under class of interest.
        """
        element = None
        for synset in xml_root.iter():
            if synset.get("wnid") == wnid_of_class:
                # We found the element with the correct WordNet ID.
                element = synset
                break

        wnid_list_of_class = []
        element_queue = [element]

        # Do breadth-first traversal starting at element, appending each WordNet ID to the output list.
        while len(element_queue) > 0:
            current = element_queue.pop(0)
            wnid_list_of_class.append(current)
            element_queue += list(current)

            #if synset.findall("[@wnid=" + '\'' + wnid_of_class + '\'' + "]") != []:
            #    wnid_list_of_class = synset.findall("[@wnid=" + '\'' + wnid_of_class + '\'' + "]")
        return wnid_list_of_class

    def verify_image(self, image_filepath):
        """
        Removes image if it is not-openable, is smaller than 256x256,
        or has white corners.
        :return: Boolean TRUE if image is okay, FALSE if image is un-wanted.
        """
        is_good_image = True
        try:
            # Try opening image
            opened_image = Image.open(image_filepath)
            width, height = opened_image.size
            white_corners = 0
            image_matrix = np.asarray(opened_image.convert('L'))

            # Check for white corners
            for x in [0, width - 1]:
                for y in [0, height - 1]:
                    if image_matrix[y, x] > 237:
                        white_corners += 1

            # Check image size
            if white_corners == 4 or width < 256 or height < 256:
                is_good_image = False
        except:
            is_good_image = False

        return is_good_image


    def download_by_wnid(self, wnid, save_path, max_to_download):
        """
        Download images by WordNet ID to specified path.
        ++Called by fetch_images_of_class++
        :param wnid: WordNet ID to download corresponding images for.
        :param save_path: Path to save images to.
        :param max_to_download: Maximum number of images to download.
        :return: List of images that could not be download, and
        number of images successfully downloaded.
        """
        missed_images = []
        num_images_downloaded = 0
        for image_info in self.wnid_url_map[wnid]:
            image_number = image_info[0]
            image_url = image_info[1]
            save_filename = os.path.join(save_path, wnid + '_' + image_number)

            try:
                requests.packages.urllib3.disable_warnings()
                # verify=False skips verifying SSL certificate (man in the middle?)
                image_request = \
                    requests.get(image_url, stream=True, verify=False, timeout=(1, 10))

            # except (requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout,
            #         requests.exceptions.TooManyRedirects) as e:
            except:
                # print "Unable to download image at", image_url, "for reason:", e
                missed_images.append(image_number)
                continue

            if image_request.status_code == 200:
                image_successful_download = True
                with open(save_filename, 'wb') as f:
                    # If there is a "requests.exceptions.ConnectionError:
                    # HTTPConnectionPool(host='auto.people.com.cn', port=80): Read timed out."
                    # break and append the image to missed_images
                    try:
                        for image_chunk in image_request:
                            f.write(image_chunk)
                    except:
                        image_successful_download = False
                        missed_images.append(image_number)

                if image_successful_download and self.verify_image(save_filename):
                    num_images_downloaded += 1
                    if num_images_downloaded == max_to_download:
                        break
                    continue

                # print "Image failed to verify:", image_url
                os.remove(save_filename)
            missed_images.append(image_number)

        return missed_images, num_images_downloaded


    def fetch_images(self):
        """
        Fetch images of all categories.
        (Saves to disk - doesn't return anything functionally)
        """
        # Fetch and save images to respective category/class_name directory
        for category in self.wnid_map:
            num_classes = len(self.wnid_map[category])
            num_images_to_download = \
                math.ceil(float(self.tot_num_images)/(num_classes*len(self.categories)))

            for class_name in self.wnid_map[category]:
                print "Number of images to download & class name: ", num_images_to_download, class_name

                num_remaining = num_images_to_download
                save_path = os.path.join(self.raw_images_path, category, class_name)

                wnids_downloaded = []

                # Number of WordNet IDs associated with class
                num_wnids_for_class = len(self.wnid_map[category][class_name])
                print "Number of WordNet IDs for class & class name: ", num_wnids_for_class, class_name

                # Number of images to download per WordNet ID per class
                num_images_to_download_per_wnid = \
                    math.ceil(num_images_to_download / num_wnids_for_class)

                print "Number of images to download per WordNet ID & class name: ", \
                    num_images_to_download_per_wnid, class_name


                # Pick a random integer between 0 and len(self.wnid_map[category][class_name]

                while num_remaining > 0 and len(self.wnid_map[category][class_name]) > 0:
                    random_index = \
                        random.randint(0,len(self.wnid_map[category][class_name])-1)

                    # Randomly picked WordNet ID
                    wnid = self.wnid_map[category][class_name][random_index]

                    # Remove WordNet ID from list to pick from
                    del self.wnid_map[category][class_name][random_index]

                    # Download images with WordNet ID
                    missed_images, num_downloaded = \
                        self.download_by_wnid(wnid, save_path,
                                              min(num_images_to_download_per_wnid, num_remaining))
                    num_remaining -= num_downloaded

                    print "WordNet ID %d of %d"\
                        % (num_wnids_for_class-len(self.wnid_map[category][class_name]), num_wnids_for_class)
                    print "%d images could not be downloaded, %d images successfully downloaded. "\
                          % (len(missed_images), num_downloaded)

                    if num_remaining < 0:
                        raise Exception('Downloaded too many images.')

                    if num_downloaded > 0:
                        wnids_downloaded.append(wnid)

                    if len(missed_images) == 0:
                        continue

                    # missed_images is a list of image numbers.
                    self.wnid_url_map[wnid] = \
                        filter(
                            lambda image: not image[0] in missed_images,
                            self.wnid_url_map[wnid]
                        )

                if num_remaining > 0:
                    print "Unable to download some images, not enough available: ", num_remaining

                print "Total number of images downloaded for class & class name: ", \
                    num_images_to_download - num_remaining, class_name
                self.wnid_map[category][class_name] = wnids_downloaded

        print "Image fetching complete."

    # System for manual deletion, then rebalancing afterwards
    # (delete everything but necessary number of images for one set)
    def rebalance_img_set(self, desired_number_images_per_class):
        """
        After manual deletion of undesirable images from each of the folders in
        the set, will automatically and randomly delete images from each set so
        that there are an even number of images from each folder in the set.
        """
        for root, dirs, files in os.walk(self.raw_images_path):
            current_number_images_in_class = len(files)
            while current_number_images_in_class > desired_number_images_per_class:
                random_index = random.randrange(0, current_number_images_in_class)
                os.remove(os.path.join(root, files[random_index]))
                del files[random_index]
                current_number_images_in_class -= 1

        print "Image set rebalancing complete."

    def prepare_img_set(self):
        """
        This prepares the images for Amazon Mechanical Turk. It
        checks that images are in jpeg format, and crops them into a square.
        """
        for root, dirs, files in os.walk(self.raw_images_path):
            for file in files:
                # if file.endswith(".jpg"):
                file_path = os.path.join(root, file)
                image = Image.open(file_path)
                width, height = image.size
                # Crop image to square from center
                if not ((width == 256) and (height == 256)):
                    if width > height:
                        left = (width - height) / 2
                        top = 0
                        right = left + height
                        bottom = top + height
                    else:
                        left = 0
                        top = (height - width) / 2
                        right = left + width
                        bottom = top + width
                    image = image.crop((left, top, right, bottom))
                    # Resize to 256 x 256
                    image = image.resize((256, 256), Image.ANTIALIAS)
                os.remove(file_path)
                image.convert('RGB').save(file_path+'.jpg')

        print "Image preparation complete."

    def compile_and_norm_img_set(self):
        """
        Compiles image set from the individual class folders to create one set,
        normalize the pixel intensity of the images across the set and
        converts all images to grayscale.
        """
        images = {}
        image_mean_list = []
        for root, dirs, files in os.walk(self.raw_images_path):
            if len(files) > 0:
                for file in files:
                    if file.endswith(".jpg"):
                        file_path = os.path.join(root, file)
                        image = Image.open(file_path)
                        # Convert image to grayscale
                        image_array = np.asarray(image.convert('L'))
                        # Save image mean value
                        image_mean = np.mean(image_array)
                        image_mean_list.append(image_mean)
                        # Subtract mean from image
                        image_array = image_array - image_mean
                        images[file] = image_array
                image_set_global_mean = np.mean(image_mean_list)

        print "Compilation and normalization complete."

        # Add global mean to each image
        for file in images:
            image_array = images[file]

            image_array += image_set_global_mean
            image = Image.fromarray(image_array).convert('L')
            image.save(os.path.join(self.image_set_path, file))

    def pickle_imagenet(self):
        """
        Pickles the WordNet ID map, and the WordNet ID - URL map.
        """
        # Pickle wnid_map
        serialized_wnid_map_path = open(self.serialized_wnid_map_path, 'wb')
        pickle.dump(self.wnid_map, serialized_wnid_map_path)
        serialized_wnid_map_path.close()

        # Pickle wnid_url_map
        serialized_wnid_url_map_path = open(self.serialized_wnid_url_map_path, 'wb')
        pickle.dump(self.wnid_url_map, serialized_wnid_url_map_path)
        serialized_wnid_url_map_path.close()

        print "Pickling complete."


    # TODO: Create a function that can read all the images, make a list of what is
        # downloaded, and then exclude the already downloaded ones from being redownloaded
    # TODO: Parallelize the code




