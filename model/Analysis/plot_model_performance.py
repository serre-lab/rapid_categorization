"""
Written by Michele W on 20 March 2017, to graph the performance of the deep network without human correlation.
"""

from rapid_categorization.model import util
import numpy as np
import matplotlib.pyplot as plt
import os



def plot_model_performance(model, classifier_type, train_batches, set_index, set_name, filter_on, filter_category):

    # txt files
    # vehicles_list = '/home/michele/python/rapid_categorization/imagenet_request/n04524313_whole_tree_vehicle.txt'
    # animal_list = '/home/michele/python/rapid_categorization/imagenet_request/whole_tree_animal.txt'
    # structure_list = '/home/michele/python/rapid_categorization/imagenet_request/structure_wnid_tree.txt'
    # if filter_on == 'filter_on':
    #     if filter_category == 'vehicles':
    #         txt_file = vehicles_list
    #     elif filter_category == 'animals':
    #         txt_file = animal_list
    #     elif filter_category == 'structure':
    #         txt_file = structure_list
    #     elif filter_category == 'distractors':
    #         exclude_file_animal = animal_list
    #         exclude_file_structure = structure_list
    #         exclude_file_vehicles = vehicles_list
    #
    #     if not filter_category == 'distractors':
    #         with open(txt_file, 'r') as f:
    #             tree_list = list(f)
    #             tree_list[:] = [x.strip().strip('-') for x in tree_list]
    #     elif filter_category == 'distractors':
    #         with open(exclude_file_animal, 'r') as f:
    #             exc_animal_list = list(f)
    #             exc_animal_list[:] = [x.strip().strip('-') for x in exc_animal_list]
    #         with open(exclude_file_structure, 'r') as f:
    #             exc_structure_list = list(f)
    #             exc_structure_list[:] = [x.strip().strip('-') for x in exc_structure_list]
    #         with open(exclude_file_vehicles, 'r') as f:
    #             exc_vehicle_list = list(f)
    #             exc_vehicle_list[:] = [x.strip().strip('-') for x in exc_structure_list]

    # Hard code set_index to be 16
    layer_names = util.get_model_layers(model)
    # print layer_names
    print "layers len: ", len(layer_names)
    acc_list = []
    for layer in layer_names:
        inputfn = util.get_predictions_filename(model, layer, classifier_type, train_batches, set_index, set_name)
        modeldata = np.load(inputfn)

        # if not filter_category == 'distractors':
        #     zipped = [x for x in zip(modeldata['source_filenames'], modeldata['pred_labels'], modeldata['true_labels'])
        #               if x[0].split('_')[0] in tree_list]
        # elif filter_category == 'distractors':
        #     zipped = [x for x in zip(modeldata['source_filenames'], modeldata['pred_labels'], modeldata['true_labels'])
        #               if not x[0].split('_')[0] in exc_animal_list and not x[0].split('_')[0] in exc_structure_list
        #               and not x[0].split('_')[0] in exc_vehicle_list]

        # ternary operator
        # acc = reduce(lambda accum, data: accum + 1 if data[1] == data[2] else accum, zipped, 0)/ float(len(zipped))

        acc = float(sum(modeldata['pred_labels'] == modeldata['true_labels'])) / float(
            len(modeldata['pred_labels']))

        acc_list += [acc]

        # all incorrectly predicted images
        mislabeled = [x for x in zip(modeldata['source_filenames'], modeldata['pred_labels'], modeldata['true_labels'],
                                     modeldata['hyper_dist']) if x[1] != x[2]]

    print acc_list
    print "acc list len: ", len(acc_list)
    x = np.linspace(0, 1, len(layer_names))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # if filter_on == 'filter_on':
    #     ax.set_title('VGG16 Model accuracy on %s images' %(filter_category))
    ax.set_title('VGG16 Model Accuracy on %s Dataset setidx=%d' %(set_name,set_index))
    ax.set_xlabel('Relative Layer Depth')
    ax.set_ylabel('Accuracy (%)')
    # Plot accuracy points
    ax.plot(x, acc_list, 'o')
    # Plot polyfit
    p1_fit = np.polyfit(x, acc_list, 2)
    p1_fn = np.poly1d(p1_fit)
    xs = np.linspace(x[0], x[-1])
    print len(xs)
    ax.plot(xs, p1_fn(xs), 'b')
    plt.show()

    # show image collage and hyperplane distances of all incorrectly predicted images
    plt.figure()
    # Get list of paths to all incorrectly predicted images
    for i_zipped in xrange(len(mislabeled)):
        image_name = mislabeled[i_zipped][0]
        image_hyper_dist = mislabeled[i_zipped][3]
        image_path = os.path.join(TURK_images_root, image_name + '.png')
        ax = plt.subplot(10, 9, i_zipped + 1)
        plt.title(str(image_hyper_dist),, fontsize=6)


    nan_idx = 0

    for i_image in xrange(n_images):

        ## ----- cell 7 ----- ##
        curr_image_path = image_fold_path + '/' + image_list[i_image]
        print '## Running image ##', curr_image_path

        output_prob = output['prob'][0]  # the output probability vector for the first image in the batch

        # print 'predicted class is:', output_prob.argmax()

        if output_prob.argmax() == 0:
            print 'Condition met'

            predicted_prob = output_prob[output_prob.argmax()]

            ## ----- cell 9 ----- ##
            # load ImageNet labels
            labels = ['non-animal', 'animal']

            output_label = labels[output_prob.argmax()]
            # print 'output label:', output_label
            # print 'output probs: ', output_prob

            image_class_output_title = image_list[i_image] + '\n' + output_label + '\n' + str(predicted_prob)
            # print image_class_output_title


            im = ax.imshow(image)
            plt.title(image_class_output_title, fontsize=6)
            ax.axis('off')
            nan_idx += 1

    # plt.tight_layout()

    result_title = image_dir_root + image_folder + '_nan_filtered.png'

    plt.savefig(result_title)




# Layers that have been tested on but not trained on are 16, 17, 18, 19
set_name = 'artifact_vehicles_turk'
model = 'VGG16'
classifier_type = 'svm'
train_batch_num = 16
train_batches = range(0, train_batch_num)
set_index = 15

# for plotting incorrectly labeled images
TURK_images_root = '/media/data_cifs/nsf_levels/michele/entire_sets/artifact_testing_set_for_humans/artifact_testing_set/'

# category_txt adds a filter to see what accuracy corresponds to what image category


# plot_model_performance(model, classifier_type, train_batches, set_index, set_name)
# plot_model_performance(model, classifier_type, train_batches, set_index, set_name, 'filter_on', 'distractors')
plot_model_performance(model, classifier_type, train_batches, set_index, set_name, 'filter_off', 'nothing')