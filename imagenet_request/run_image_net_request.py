# Written by Michele 10 Feb 2017

from image_net_request import ImageNet

foo = ImageNet('/media/data/nsf_levels_michelecopy/imagenet_data/ImageNet_structure.xml',
               '/media/data/nsf_levels_michelecopy/transportation_scenes_100K',
               './transportation_scenes.txt', 200000, '/media/data/nsf_levels_michelecopy/imagenet_data/fall11_urls.txt')

# Check number of images we have references for
# print sum([len(foo.wnid_url_map[key]) for key in foo.wnid_url_map])

foo.fetch_images()

foo.prepare_img_set()

# Wait for signal that images have been manually (human) sorted
sorted_signal = input("Type 'yes' if you are done manually sorting the fetched images.")
if sorted_signal == 'yes':

    foo.rebalance_img_set(50000)

    foo.compile_and_norm_img_set()

    foo.pickle_imagenet()

