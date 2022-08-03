import os
# Training config
input_shape = (224, 224, 3)
weight_decay = 1e-4
label_smoothing = 0.1


#test_iterations = int(test_num / batch_size) + 1

# initial_learning_rate = 0.1

log_file = 'result/log/ResNet_50_mm.txt'


# Augmentation config
# From 'Bag of tricks for image classification with convolutional neural networks'
# Or https://github.com/dmlc/gluon-cv
short_side_scale = (256, 384)
aspect_ratio_scale = (0.8, 1.25)
hue_delta = (-36, 36)
saturation_scale = (0.6, 1.4)
brightness_scale = (0.6, 1.4)
pca_std = 0.1

mean = [103.939, 116.779, 123.68]
std = [58.393, 57.12, 57.375]
eigval = [55.46, 4.794, 1.148]
eigvec = [[-0.5836, -0.6948, 0.4203],
          [-0.5808, -0.0045, -0.8140],
          [-0.5675, 0.7192, 0.4009]]

# ResNet config

block_type = {18: 'basic block',
              34: 'basic block',
              50: 'bottlenect block',
              101: 'bottlenect block',
              152: 'bottlenect block'}

block_num = {18: (2, 2, 2, 2),
             34: (3, 4, 6, 3),
             50: (3, 4, 6, 3),
             101: (3, 4, 23, 3),
             152: (3, 4, 36, 3)}

filter_num = (64, 128, 256, 512)

save_steps = 10
"""
tf_config = {
        'cluster': {
            'worker' : ['192.168.1.21:30000', '192.168.1.22:30001'],
        },
        'task': {'type': 'worker'}
}
"""
tf_config = {
        'cluster': {
            'worker' : ['192.168.0.149:30000', '192.168.0.149:30001'],
        },
        'task': {'type': 'worker'}
}
