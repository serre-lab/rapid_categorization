'''
Written by Michele 24 March 2017 to copy over distractor images from Jonah's experiment
but avoid copying over images with vehicle subset WNIDs
'''


import os
import shutil
import numpy as np
import random
from PIL import Image

def get_distractors(txtfile, source_dir, final_dir):
    trainingset_v_list = os.listdir(trainingset_v_dir)

    with open(vehicle_tree_txt, 'r') as f:
        vehicle_tree_list = list(f)
        vehicle_tree_list[:] = [x.strip().strip('-') for x in vehicle_tree_list]
        # print vehicle_tree_list

        num_skipped = 0
        for image in trainingset_v_list:
            # print str(image.split('_')[0])
            if str(image.split('_')[0]) in vehicle_tree_list:
                print 'Image: ', image, ' skipped.'
                num_skipped += 1
                continue
            else:
                shutil.copy(os.path.join(trainingset_v_dir, image), distractor_folder)

    print 'Copying complete. ', num_skipped, ' images skipped.'



# Load in vehicle WNIDs entire tree as list
vehicle_tree_txt = '/home/michele/python/rapid_categorization/imagenet_request/n04524313_whole_tree_vehicle.txt'

# Copy trainingset_v images to distractor folder but avoid vehicle WNIDs
trainingset_v_dir = '/media/data_cifs/nsf_levels/raw_ims/sets/trainingset_v'
distractor_folder = '/media/data/nsf_levels_michelecopy/jonah_distractors'


# Run get_distractors
# get_distractors(vehicle_tree_txt, trainingset_v_dir, distractor_folder)
################

# This function takes in a txt file in the format copied from ImageNet and turns it into a list.
def txt_to_list(txtfile):
    with open(txtfile, 'r') as f:
        tree_list = list(f)
        tree_list[:] = [x.strip().strip('-') for x in tree_list]
    f.close()
    return tree_list

def keep_percent_animal_structure(copy_from_dir, structure_txt, animal_txt,
                                  source_dir, percent_to_keep_structure, percent_to_keep_animal, total_img_num):

    max_num_structure_images = np.ceil(float(total_img_num)*percent_to_keep_structure)
    max_num_animal_images = np.ceil(float(total_img_num)*percent_to_keep_animal)

    # Copy over images to this new directory so deletions can begin
    print 'Copying files from ', copy_from_dir, ' to ', source_dir, '.'
    shutil.copytree(copy_from_dir, source_dir)
    print 'Copying complete.'

    distractor_list = os.listdir(source_dir)

    with open(animal_txt, 'r') as f:
        animal_tree_list = list(f)
        animal_tree_list[:] = [x.strip().strip('-') for x in animal_tree_list]
        # print vehicle_tree_list

        with open(structure_txt, 'r') as j:
            structure_tree_list = list(j)
            structure_tree_list[:] = [x.strip().strip('-') for x in structure_tree_list]

            animal_num = 0
            animal_removed = 0
            structure_num = 0
            structure_removed = 0
            distractor_num = 0
            distractor_removed = 0

            for iter in xrange(0, len(distractor_list)):
                random_index = random.randrange(0, len(distractor_list))
                curr_image = distractor_list[random_index]

                # animal
                if str(curr_image.split('_')[0]) in animal_tree_list and animal_num < max_num_animal_images:
                    animal_num += 1
                    distractor_num += 1
                    continue
                elif str(curr_image.split('_')[0]) in animal_tree_list and animal_num == max_num_animal_images:
                    os.remove(os.path.join(source_dir, curr_image))
                    animal_removed += 1
                    distractor_removed += 1

                # structure
                elif str(curr_image.split('_')[0]) in structure_tree_list and structure_num < max_num_structure_images:
                    structure_num += 1
                    distractor_num += 1
                    continue
                elif str(curr_image.split('_')[0]) in structure_tree_list and structure_num == max_num_structure_images:
                    os.remove(os.path.join(source_dir, curr_image))
                    structure_removed += 1
                    distractor_removed += 1

                # other distractors
                elif not str(curr_image.split('_')[0]) in animal_tree_list and \
                        not str(curr_image.split('_')[0]) in structure_tree_list and \
                                distractor_num < total_img_num:
                    distractor_num += 1
                    continue
                elif not str(curr_image.split('_')[0]) in animal_tree_list and \
                        not str(curr_image.split('_')[0]) in structure_tree_list and \
                                distractor_num == total_img_num:
                    os.remove(os.path.join(source_dir, curr_image))
                    distractor_removed += 1

                del distractor_list[random_index]

    print 'Removal complete. '
    print 'Num animal images kept: ', animal_num
    print 'Num structure images kept: ', structure_num
    print 'Num distractor images total: ', distractor_num
    print 'Num animal images removed: ', animal_removed
    print 'Num structure images removed: ', structure_removed
    print 'Num distractor images removed: ', distractor_removed



# Load in animal WNIDs entire tree as list
animal_tree_txt = '/home/michele/python/rapid_categorization/imagenet_request/whole_tree_animal.txt'
structure_tree_txt = '/home/michele/python/rapid_categorization/imagenet_request/structure_wnid_tree.txt'
# fifty_percent_animal_dir = '/media/data/nsf_levels_michelecopy/jonah_distractors_sets/individual_image_folders/50_percent_jonah_distractors'
#twentyfive_percent_animal_dir = '/media/data/nsf_levels_michelecopy/jonah_distractors_sets/individual_image_folders/25_percent_jonah_distractors'
root_dir = '/media/data/nsf_levels_michelecopy/'
twentyfive_percent_structure_animal_dir = os.path.join(root_dir,'structure_distractors_25_set')
balanced_dir = os.path.join(root_dir, 'jonah_distractors_sets/individual_image_folders/balanced_distractor_set')
copy_from_dir = os.path.join(root_dir, 'structure_distractors_pool_all_imgs')

# keep_percent_animal_structure(copy_from_dir, structure_tree_txt, animal_tree_txt, balanced_dir, 0.5, 0.2, 50000)
# keep_percent_animal(animal_tree_txt, twentyfive_percent_animal_dir, 0.25, 50000)
################


##### Get vehicle images from training set into new set

def get_vehicle_images(txtfile, source_image_directory, destination_image_directory):
    with open(txtfile, 'r') as f:
        all_training_images = list(f)
        for image in all_training_images:
            image_name = image.strip().split('\t')
            if image_name[1] == '1':
                shutil.copy(os.path.join(source_image_directory, image_name[0])+'.JPEG', destination_image_directory)
            elif image.strip().split('\t')[1] == '0':
                continue
    print 'Vehicle images copied.'


complete_list = '/media/data/nsf_levels_michelecopy/transportation_scenes_cleaned_1/training_sets_for_CNN/complete_training_list_images.txt'
source_dir = '/media/data/nsf_levels_michelecopy/transportation_scenes_cleaned_1/cleaned_images_for_CNN/vehicles'
dest_dir = '/media/data/nsf_levels_michelecopy/jonah_distractors_sets/individual_image_folders/vehicles'

# get_vehicle_images(complete_list, source_dir, dest_dir)
################




"""
Make training set.Gets images from individual folders and creates training set in another entire set dir.
Also makes complete training set txt file.
"""

def make_train_set(train_list_name, class_one_dir, class_zero_dir, dest_training_set_dir):
    with open(train_list_name, 'w') as f:
        ## Class one
        curr_class = class_one_dir
        print 'Working on class ' + curr_class
        class_label = '1'

        # list of all images in class one
        list_images_in_class = os.listdir(curr_class)
        for image in list_images_in_class:
            f.write(image.split('.')[0] + '\t' + class_label + '\n')
            shutil.copyfile(os.path.join(curr_class, image), os.path.join(dest_training_set_dir,image))

        ## Class zero
        curr_class = class_zero_dir
        print 'Working on class ' + curr_class
        class_label = '0'

        # list of all images in class zero
        list_images_in_class = os.listdir(curr_class)
        # shuffle list to make sure even sampling across all wnids
        random.shuffle(list_images_in_class)
        # copied_count = 0
        # while copied_count < 50000:
        for image in list_images_in_class:
            f.write(image.split('.')[0] + '\t' + class_label + '\n')
            # shutil.copy(os.path.join(curr_class, image), dest_training_set_dir)
            shutil.copyfile(os.path.join(curr_class, image), os.path.join(dest_training_set_dir,image.split('.')[0] + '.' + image.split('.')[1]))
                # copied_count +=1



    f.close()
    print 'Done selecting images for training set.'

indv_imgs_dir = '/media/data/nsf_levels_michelecopy/jonah_distractors_sets/individual_image_folders/'
entire_sets_dir = '/media/data/nsf_levels_michelecopy/jonah_distractors_sets/entire_sets/'

fifty_train_list_name = '/media/data/nsf_levels_michelecopy/jonah_distractors_sets/entire_sets/dist_veh_50_training_set_for_CNN/50_trainset_complete_list_CNN.txt'
twentyfive_train_list_name = '/media/data/nsf_levels_michelecopy/jonah_distractors_sets/entire_sets/dist_veh_25_training_set_for_CNN/25_trainset_complete_list_CNN.txt'
struct_25_train_list_name = os.path.join(entire_sets_dir, 'struct_dist_25_training_set_for_CNN/struct_25_trainset_complete_list_CNN.txt')

# Always true for my sets
class_one_dir = '/media/data/nsf_levels_michelecopy/jonah_distractors_sets/individual_image_folders/vehicles'

fifty_class_zero_dir = '/media/data/nsf_levels_michelecopy/jonah_distractors_sets/individual_image_folders/50_percent_jonah_distractors'
twentyfive_class_zero_dir = '/media/data/nsf_levels_michelecopy/jonah_distractors_sets/individual_image_folders/25_percent_jonah_distractors'
struct_25_class_zero_dir = os.path.join(indv_imgs_dir, 'structure_distractors_25_set')

fifty_dest_training_set_dir = '/media/data/nsf_levels_michelecopy/jonah_distractors_sets/entire_sets/dist_veh_50_training_set_for_CNN/50_percent_entire_set'
twentyfive_dest_training_set_dir = '/media/data/nsf_levels_michelecopy/jonah_distractors_sets/entire_sets/dist_veh_25_training_set_for_CNN/25_percent_entire_set'
struct_25_dest_training_set_dir = os.path.join(entire_sets_dir, 'struct_dist_25_training_set_for_CNN/struct_dist_25_entire_set')


# make_train_set(struct_25_train_list_name, class_one_dir, struct_25_class_zero_dir, struct_25_dest_training_set_dir)
################



# Make all images .JPEG (because some without labels)

def convert_images_to_JPEG(source_dir):
    image_list = os.listdir(source_dir)

    for train_image in image_list:
        if train_image.endswith('.JPEG'):
            continue
        else:
            image = Image.open(os.path.join(source_dir, train_image))
            train_image_JPEG = train_image + '.JPEG'
            image.save(os.path.join(source_dir, train_image_JPEG))


fifty_dest_training_set_dir = '/media/data/nsf_levels_michelecopy/jonah_distractors_sets/entire_sets/50_percent_entire_set'
twentyfive_dest_training_set_dir = '/media/data/nsf_levels_michelecopy/jonah_distractors_sets/entire_sets/25_percent_entire_set'


# convert_images_to_JPEG(fifty_dest_training_set_dir)
################

def verify_image(image_filepath):
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


def prepare_img_set(raw_images_path):
    """
    This prepares the images for Amazon Mechanical Turk. It
    checks that images are in jpeg format, and crops them into a square.
    """
    for root, dirs, files in os.walk(raw_images_path):
        for file in files:
            # if not file.endswith(".JPEG"):
            #     continue

            file_path = os.path.join(root, file)
            # print "Working on "+ file_path
            if verify_image(file_path):
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
                image.convert('RGB').save(file_path+'.JPEG')
                # print file_path+" saved."
            else:
                os.remove(file_path)
                print file_path+' removed.'

    print "Image preparation complete."


# prepare_img_set('/media/data/nsf_levels_michelecopy/jonah_distractors_crop')
################

# Function written on 10 Apr 2017 to copy over all sport synset images based on whole txtfile tree
# to /media/data/nsf_levels_michelecopy

def copy_wnids_list(txtfile, copy_from_dir, copy_to_dir):
    wnids_in_from_dir = os.listdir(copy_from_dir)
    wnid_list = txt_to_list(txtfile)
    missed_wnids = []
    for wnid in wnid_list:
        if wnid in wnids_in_from_dir:
            shutil.copytree(os.path.join(copy_from_dir,wnid), copy_to_dir)
        elif not wnid in wnids_in_from_dir:
            missed_wnids += [wnid]
            print 'WNID missed: ', wnid

    print 'Copying complete.'
    print '%d WNIDs missed' %(len(missed_wnids))

# copy_wnids_list('whole_tree_sport.txt', '/media/data_cifs/ImageNet/animal/', 'media/data/nsf_levels_michelecopy/sport/')
############################################

def balanced_sampling_via_wnids(total_num_images, wnid_list, wnid_dictionary, source_dir, dest_dir, exclusion_list):
    images_picked = 0
    while images_picked < total_num_images:
        # Pick one image per wnid per round (make it balanced)
        for wnid in wnid_list:
            current_list_images_choose_from = wnid_dictionary[wnid]
            if len(current_list_images_choose_from) == 0:
                continue
            random_index = random.randrange(0, len(current_list_images_choose_from))
            randomly_picked_image = current_list_images_choose_from[random_index]
            if randomly_picked_image.split('.')[0] in exclusion_list:
                continue
            # Copy images to training set (artifact_sport_distractor_dir)
            shutil.copyfile(os.path.join(source_dir, randomly_picked_image),
                            os.path.join(dest_dir, randomly_picked_image))
            images_picked += 1
            # print randomly_picked_image, ' copied.'
            del current_list_images_choose_from[random_index]





