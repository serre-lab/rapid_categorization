# Written by Michele 10 Feb 2017

from image_net_request import ImageNet

foo = ImageNet('/media/data/nsf_levels_michelecopy/imagenet_data/ImageNet_structure.xml',
               '/media/data/nsf_levels_michelecopy/cars_pedestrians',
               './cars_pedestrians.txt', 100 * 6, '/media/data/nsf_levels_michelecopy/imagenet_data/fall11_urls.txt')

# Check number of images we have references for
# print sum([len(foo.wnid_url_map[key]) for key in foo.wnid_url_map])

foo.fetch_images()

foo.rebalance_img_set(25*6)

foo.prepare_img_set()

foo.compile_and_norm_img_set()

foo.pickle_imagenet()

