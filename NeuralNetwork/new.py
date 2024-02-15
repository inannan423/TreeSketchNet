import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np
from utils import natural_keys
from models.my_model import ModelTree  # 根据实际情况替换模型导入路径
from utils.write_logs import read_dictionary_from_file, save_dictionary_to_file
from utils.my_metrics import OneMinusRMSE
from dataset.parameters_elab import get_subdivision_keys, assign_correct_type_toParams
import ntpath
from dataset import PairImgParamClassification

local_dir = os.path.dirname(__file__)
test_set_dir = local_dir + "/test_set"
log_dir = local_dir + "/logs_archive/fit"

model_name = "efficientnet_multiple"

IMG_SHAPE = [224, 224, 3]

if os.path.exists(log_dir):
    list_model = [file for file in os.listdir(log_dir) if file.endswith(".h5")]
    if list_model:
        list_model.sort(key=natural_keys)
        model = tf.keras.models.load_model(os.path.join(log_dir, list_model[-1]), custom_objects={'OneMinusRMSE': OneMinusRMSE})

image_path = "NeuralNetwork\\test_set\\styles_sketch_images\\hand\\2_camera_right.png"
img = tf.io.read_file(image_path)
img = tf.io.decode_jpeg(img, channels=IMG_SHAPE[2])
img = tf.image.resize(img, [IMG_SHAPE[0], IMG_SHAPE[1]])
img = tf.cast(img, tf.float32)
img = img.numpy()

img = img / 255.0

# 添加批量维度
img = tf.expand_dims(img, axis=0)

result = model(img)

respective_dictionary = {}

subdivision_keys = get_subdivision_keys()

pair_img_param = PairImgParamClassification(IMG_SHAPE, subdivision_keys, model_name, test_set_dir, shuffle=False)

normalization_matrices = np.load(os.path.join(log_dir, 'normalization_dict.npy'), allow_pickle=True)
for index, key in enumerate(normalization_matrices.item().keys()):
    denorm = result[index].numpy() * normalization_matrices.item().get(key)
    param_dict = pair_img_param.reconstruct_params_dict(subdivision_keys[key], denorm)

    respective_dictionary.update(param_dict)

if 'keys_seed' not in normalization_matrices.item().keys():
    respective_dictionary['seed'] = 0

respective_dictionary = assign_correct_type_toParams(respective_dictionary)

print(respective_dictionary)