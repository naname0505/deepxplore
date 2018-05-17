import random
from collections import defaultdict

import numpy as np
from keras import backend as K
from keras.models import Model


# util function to convert a tensor into a valid image
# TFで扱えるようにgen_diffの最初でx_test/255でテンソルへ変換していた
# テンソルに変換したものを元の画像の画素値へ戻す関数
def deprocess_image(x):
    x *= 255
    # Given an interval, values outside the interval are clipped to the interval edges. For example, if an interval of [0, 1] is specified, values smaller than 0 become 0, and values larger than 1 become 1.
    # np.clip[min,max]:配列の中身をmin以下のものをminへ,max以上をmaxへ
    x = np.clip(x, 0, 255).astype('uint8')
    return x.reshape(x.shape[1], x.shape[2])  # original shape (1,img_rows, img_cols,1)


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)


def constraint_occl(gradients, start_point, rect_shape):
    new_grads = np.zeros_like(gradients)
    new_grads[:, start_point[0]:start_point[0] + rect_shape[0],
    start_point[1]:start_point[1] + rect_shape[1]] = gradients[:, start_point[0]:start_point[0] + rect_shape[0],
                                                     start_point[1]:start_point[1] + rect_shape[1]]
    return new_grads


def constraint_light(gradients):
    new_grads = np.ones_like(gradients)
    grad_mean = np.mean(gradients)
    return grad_mean * new_grads


def constraint_black(gradients, rect_shape=(6, 6)):
    start_point = (
        random.randint(0, gradients.shape[1] - rect_shape[0]), random.randint(0, gradients.shape[2] - rect_shape[1]))
    new_grads = np.zeros_like(gradients)
    patch = gradients[:, start_point[0]:start_point[0] + rect_shape[0], start_point[1]:start_point[1] + rect_shape[1]]
    if np.mean(patch) < 0:
        new_grads[:, start_point[0]:start_point[0] + rect_shape[0],
        start_point[1]:start_point[1] + rect_shape[1]] = -np.ones_like(patch)
    return new_grads


def init_coverage_tables(model1, model2, model3):
    #defaultdict from Collections Class
    model_layer_dict1 = defaultdict(bool)
    model_layer_dict2 = defaultdict(bool)
    model_layer_dict3 = defaultdict(bool)
    init_dict(model1, model_layer_dict1)
    init_dict(model2, model_layer_dict2)
    init_dict(model3, model_layer_dict3)
    return model_layer_dict1, model_layer_dict2, model_layer_dict3


def init_dict(model, model_layer_dict):
    for layer in model.layers:
        # "flatten" layerは平滑化を行う層
        # 詳しくは https://keras.io/ja/layers/core/ を参照
        if 'flatten' in layer.name or 'input' in layer.name:
            continue # "flatten"と"input"層は辞書に追加せずにスルー 
        for index in range(layer.output_shape[-1]):
            model_layer_dict[(layer.name, index)] = False

# 活性化していないニューロンの中からランダムに一つを選び,
# その層(layer_name)と何番目のニューロンか(index)の2つをreturn
def neuron_to_cover(model_layer_dict):
    not_covered = [(layer_name, index) for (layer_name, index), v in model_layer_dict.items() if not v]
    #print("##############")
    #print(not_covered)
    #print("##############")

    if not_covered:
        layer_name, index = random.choice(not_covered)
    else:
        layer_name, index = random.choice(model_layer_dict.keys())
    #print("$$$$$$$$$$$$$$"+str(index))
    return layer_name, index

# modelの[活性化したニューロン数][全ニューロン数][活性化したニューロン数/全ニューロン数]
def neuron_covered(model_layer_dict):
    #model_layer_dict.valuesはmodel内の層(neuronの数分)と,活性化の有無を示すTrue or Falseが記述
    covered_neurons = len([v for v in model_layer_dict.values() if v])
    # ↑model_layer_dict内の活性化の是非がTrueのものの数
    total_neurons = len(model_layer_dict)
    return covered_neurons, total_neurons, covered_neurons / float(total_neurons)


def update_coverage(input_data, model, model_layer_dict, threshold=0):
    layer_names = [layer.name for layer in model.layers if
                   'flatten' not in layer.name and 'input' not in layer.name]
    # ここからkeras_nural.py のshow_intermediateと似た内容
    # 中間層を出力とするmodelを定義し, 実際にデータを入力して中間層の出力を保持する
    # 中間層の値を出力層の値とするようなmodelを定義 
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=[model.get_layer(layer_name).output for layer_name in layer_names])
    intermediate_layer_outputs = intermediate_layer_model.predict(input_data)

    for i, intermediate_layer_output in enumerate(intermediate_layer_outputs):
        scaled = scale(intermediate_layer_output[0]) #確か[0]がテスト,[1]が学習.これによってDOの有無とかを変更している. 要API参照
        #print(scaled)
        for num_neuron in range(scaled.shape[-1]):
            if np.mean(scaled[..., num_neuron]) > threshold and not model_layer_dict[(layer_names[i], num_neuron)]: # 平均が閾値(0-1)を越えて且つdictのboolがFalseなら
                model_layer_dict[(layer_names[i], num_neuron)] = True #dictをTrueに


def full_coverage(model_layer_dict):
    if False in model_layer_dict.values():
        return False
    return True


def scale(intermediate_layer_output, rmax=1, rmin=0):
    #           中間層の出力から最小値を引いたもの
    # X_std =  -------------------------------------
    #            中間層の出力の最大値と最小値の差
    # 配列になっていて, それぞれの値は0-1の間(小数含む)

    X_std = (intermediate_layer_output - intermediate_layer_output.min()) / (intermediate_layer_output.max() - intermediate_layer_output.min())
    X_scaled = (X_std * (rmax - rmin)) + rmin # 今のところX_scaled = X_std
    return X_scaled


def fired(model, layer_name, index, input_data, threshold=0):
    intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    intermediate_layer_output = intermediate_layer_model.predict(input_data)[0]
    scaled = scale(intermediate_layer_output)
    if np.mean(scaled[..., index]) > threshold:
        return True
    return False


def diverged(predictions1, predictions2, predictions3, target):
    #     if predictions2 == predictions3 == target and predictions1 != target:
    if not predictions1 == predictions2 == predictions3:
        return True
    return False
