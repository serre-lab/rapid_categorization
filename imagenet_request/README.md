Hello! Welcome to the README for the NSF Rapid Visual Categorization Experiment Dataset Creation Code. 

Originally written by Jonah Cader in 2016, the code was rewritten and adapted by Michele Winter in 2017.

Files:
- README_NSF.txt: Instructions on how to use this code.
- ImageNet_structure.xml: XML file containing synset structural information from ImageNet [Fall 2011 release].
- image_net_request.py: Script containing functions to create and manipulate the class 'ImageNet'.
- run_image_net_request.py: Script containing code that will run image_net_request.py.
- class_list_file_TEST.txt: Currently just a test file, but serves as a guide for writing other .txt files.
	This file is used to format the structure of the image dataset that will be made with image_net_request.py.

Sequence of actions necessary to get a dataset:

1. Make sure your class_list_file_TEST.txt (or the one you have made that has an equivalent structure) is correctly
	formatted and complete.

2. Edit run_image_net_request.py to reflect updated and correct file paths and number of desired images downloaded.
    *Note: The desired number of images downloaded should usually be double the number of images you actually want to
    use for experimentation. This will allow room for bad downloads, inappropriate downloads, and make balancing easier*

3. Use the script to create the ImageNet class, and execute the function 'fetch_images()'.

4. Manually delete downloaded images that have words in them, are too blurry, or otherwise don't match the requirements
	set by the dataset.

5. Use the script run_image_net_request.py to execute the function 'rebalance_image_set(<number_of_images_per_class>)',
	putting the number of desired images per class as an argument in the function. This will delete an extra images
	from the dataset randomly, until the desired number of images per class is reached.

6. Use the script run_image_net_request.py to execute the functions 'prepare_image_set()' and 'compile_and_norm_image_net()'.
	These functions prepare the image set for Amazon Mechanical Turk and then normalize the images and compile the
	downloaded images from the '~/raw_images' directory into the '~/image_set' directory.

7. Run 'pickle_imagenet()' to save the information of which images were used to create the set.

8. Now the dataset should be ready to use in experiments!
