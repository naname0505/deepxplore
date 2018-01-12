'''
usage: python gen_diff.py -h
'''

from __future__ import print_function

import argparse

from keras.datasets import mnist
from keras.layers import Input
from scipy.misc import imsave

from Model1 import Model1
from Model2 import Model2
from Model3 import Model3
from configs import bcolors
from utils import *

# read the parameter
# argument parsing
parser = argparse.ArgumentParser(description='Main function for difference-inducing input generation in MNIST dataset')
# 画像に加えるノイズの設定.
#light           : 輝度変化
#occl(occlusion) : 透過ノイズを乗せてる 論文の例を参照するとイメージしやすい
#                  (矩形のサイズは調節可能)
#black           : 黒点(矩形のサイズは調節可能)
parser.add_argument('transformation', help="realistic transformation type", choices=['light', 'occl', 'blackout'])
parser.add_argument('weight_diff', help="weight hyperparm to control differential behavior", type=float)
parser.add_argument('weight_nc', help="weight hyperparm to control neuron coverage", type=float)
parser.add_argument('step', help="step size of gradient descent", type=float)
#seeds:3つのモデルのうち1つ以上が別のラベルと推測するような入力の生成数を設定するパラメーター
parser.add_argument('seeds', help="number of seeds of input", type=int)
parser.add_argument('grad_iterations', help="number of iterations of gradient descent", type=int)
parser.add_argument('threshold', help="threshold for determining neuron activated", type=float)
parser.add_argument('-t', '--target_model', help="target model that we want it predicts differently",
                    choices=[0, 1, 2], default=0, type=int)
parser.add_argument('-sp', '--start_point', help="occlusion upper left corner coordinate", default=(0, 0), type=tuple)
parser.add_argument('-occl_size', '--occlusion_size', help="occlusion size", default=(10, 10), type=tuple)

args = parser.parse_args()

# input image dimensions
img_rows, img_cols = 28, 28
# the data, shuffled and split between train and test sets
(_, _), (x_test, _) = mnist.load_data()

x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

x_test = x_test.astype('float32')
x_test /= 255

# define input tensor as a placeholder
# テンソルを確保する処理 kera.layers.Input
input_tensor = Input(shape=input_shape)

# load multiple models sharing same input tensor
model1 = Model1(input_tensor=input_tensor)
model2 = Model2(input_tensor=input_tensor)
model3 = Model3(input_tensor=input_tensor)

"""
print(model1.summary())
print(model2.summary())
print(model3.summary())
"""

# init coverage table
model_layer_dict1, model_layer_dict2, model_layer_dict3 = init_coverage_tables(model1, model2, model3)
#print(model_layer_dict1)

# ==============================================================================================
# start gen inputs
for _ in range(args.seeds):
    gen_img = np.expand_dims(random.choice(x_test), axis=0)
    orig_img = gen_img.copy()
    # first check if input already induces differences
    label1, label2, label3 = np.argmax(model1.predict(gen_img)[0]), \
                             np.argmax(model2.predict(gen_img)[0]), \
                             np.argmax(model3.predict(gen_img)[0])

    if not label1 == label2 == label3:
        print(bcolors.OKGREEN + 
                'input already causes different outputs: {}, {}, {}'.format(label1,label2,label3) + bcolors.ENDC)

        update_coverage(gen_img, model1, model_layer_dict1, args.threshold)
        update_coverage(gen_img, model2, model_layer_dict2, args.threshold)
        update_coverage(gen_img, model3, model_layer_dict3, args.threshold)

        print(bcolors.WARNING + 
                'covered neurons percentage %d neurons %.3f, %d neurons %.3f, %d neurons %.3f'
              % (len(model_layer_dict1), neuron_covered(model_layer_dict1)[2],
                 len(model_layer_dict2),
                 neuron_covered(model_layer_dict2)[2], len(model_layer_dict3),
                 neuron_covered(model_layer_dict3)[2]) + bcolors.ENDC)

        averaged_nc = (neuron_covered(model_layer_dict1)[0] + neuron_covered(model_layer_dict2)[0] +
                       neuron_covered(model_layer_dict3)[0]) / float(
            neuron_covered(model_layer_dict1)[1] + neuron_covered(model_layer_dict2)[1] +
            neuron_covered(model_layer_dict3)[1])

        print(bcolors.WARNING + 'averaged covered neurons %.3f' % averaged_nc + bcolors.ENDC)

        gen_img_deprocessed = deprocess_image(gen_img)

        # save the result to disk
        imsave('./generated_inputs/' + 'already_differ_' + str(label1) + '_' + str(
            label2) + '_' + str(label3) + '.png', gen_img_deprocessed)
        continue

    # 上記のif文で既に3つのモデルに対して出力ラベルが異なっていた場合の処理を記述
    # else(即ち,3つの出力ラベルの値が一致しているとき)時の処理がメインなので以下で処理
    # if all label agrees <= 3つのモデル全ての出力ラベルが一致しているということ
    orig_label = label1 # <= 別にlabel2でもlabel3でも構わない, 全て同じ結果なので
    layer_name1, index1 = neuron_to_cover(model_layer_dict1) # init_coverage_tablesによって作成
    layer_name2, index2 = neuron_to_cover(model_layer_dict2) # init_coverage_tablesによって作成
    layer_name3, index3 = neuron_to_cover(model_layer_dict3) # init_coverage_tablesによって作成

    # construct joint loss function
    # target_modelは3つのうちにどのモデルが他と違う出力ラベルようにさせるかを指定するパラメーター
    # 当たり前だけれどもMNISTなので0-9の10個の"before_softmax"が存在

    if   args.target_model == 0: # モデル0に対して実行します
        loss1 = -args.weight_diff * K.mean(model1.get_layer('before_softmax').output[..., orig_label])
        loss2 = K.mean(model2.get_layer('before_softmax').output[..., orig_label]) #任意の行のorig_label列目のみ
        loss3 = K.mean(model3.get_layer('before_softmax').output[..., orig_label]) #任意の行のorig_label列目のみ

    elif args.target_model == 1: # モデル1に対して実行します
        loss1 = K.mean(model1.get_layer('before_softmax').output[..., orig_label]) #任意の行のorig_label列目のみ
        loss2 = -args.weight_diff * K.mean(model2.get_layer('before_softmax').output[..., orig_label])
        loss3 = K.mean(model3.get_layer('before_softmax').output[..., orig_label]) #任意の行のorig_label列目のみ

    elif args.target_model == 2: # モデル2に対して実行します
        loss1 = K.mean(model1.get_layer('before_softmax').output[..., orig_label]) #任意の行のorig_label列目のみ
        loss2 = K.mean(model2.get_layer('before_softmax').output[..., orig_label]) #任意の行のorig_label列目のみ
        loss3 = -args.weight_diff * K.mean(model3.get_layer('before_softmax').output[..., orig_label])
    
    loss1_neuron = K.mean(model1.get_layer(layer_name1).output[..., index1])
    loss2_neuron = K.mean(model2.get_layer(layer_name2).output[..., index2])
    loss3_neuron = K.mean(model3.get_layer(layer_name3).output[..., index3])
    layer_output = (loss1 + loss2 + loss3) + args.weight_nc * (loss1_neuron + loss2_neuron + loss3_neuron)

    # for adversarial image generation
    final_loss = K.mean(layer_output)
    #print(final_loss)

    # we compute the gradient of the input picture wrt this loss
    grads = normalize(K.gradients(final_loss, input_tensor)[0])
    #print(grads)

    # this function returns the loss and grads given the input picture
    # K.backend.function : 
    """
    function(
     inputs,
     outputs,
    updates=None,
    **kwargs
    )
    Returns: return Function Class.
             format of Output values is Numpy arrays.
    """
    # input_tensor はこの時点ではただのプレースホルダー.
    # 中身は何も入っていない
    print(input_tensor)
    iterate = K.function([input_tensor], [loss1, loss2, loss3, loss1_neuron, loss2_neuron, loss3_neuron, grads])


    # we run gradient ascent for 20 steps
    for iters in range(args.grad_iterations):
        loss_value1, loss_value2, loss_value3, loss_neuron1, loss_neuron2, loss_neuron3, grads_value = iterate([gen_img])
        # grads_value = iterate[gen_img] はinput_tensorにgen_imgを入れたときのloss1-3, loss1-3_neuronの値
        if iters %10 == 0:
            print(grads_value)
        if   args.transformation == 'light':
            grads_value = constraint_light(grads_value)  # constraint the gradients value
        elif args.transformation == 'occl':
            grads_value = constraint_occl(grads_value, args.start_point,
                                          args.occlusion_size)  # constraint the gradients value
        elif args.transformation == 'blackout':
            grads_value = constraint_black(grads_value)  # constraint the gradients value

        gen_img += grads_value * args.step
        #print(gen_img)
        predictions1 = np.argmax(model1.predict(gen_img)[0])
        predictions2 = np.argmax(model2.predict(gen_img)[0])
        predictions3 = np.argmax(model3.predict(gen_img)[0])

        if not predictions1 == predictions2 == predictions3:
            update_coverage(gen_img, model1, model_layer_dict1, args.threshold)
            update_coverage(gen_img, model2, model_layer_dict2, args.threshold)
            update_coverage(gen_img, model3, model_layer_dict3, args.threshold)

            print(bcolors.WARNING + 'covered neurons percentage %d neurons %.3f, %d neurons %.3f, %d neurons %.3f'
                  % (len(model_layer_dict1), neuron_covered(model_layer_dict1)[2], len(model_layer_dict2),
                     neuron_covered(model_layer_dict2)[2], len(model_layer_dict3),
                     neuron_covered(model_layer_dict3)[2]) + bcolors.ENDC)
            averaged_nc = (neuron_covered(model_layer_dict1)[0] + neuron_covered(model_layer_dict2)[0] +
                           neuron_covered(model_layer_dict3)[0]) / float(
                neuron_covered(model_layer_dict1)[1] + neuron_covered(model_layer_dict2)[1] +
                neuron_covered(model_layer_dict3)[
                    1])
            print(bcolors.WARNING + 'averaged covered neurons %.3f' % averaged_nc + bcolors.ENDC)

            gen_img_deprocessed = deprocess_image(gen_img)
            orig_img_deprocessed = deprocess_image(orig_img)

            # save the result to disk
            imsave('./generated_inputs/' + args.transformation + '_' + str(predictions1) + '_' + str(
                predictions2) + '_' + str(predictions3) + '.png',
                   gen_img_deprocessed)
            imsave('./generated_inputs/' + args.transformation + '_' + str(predictions1) + '_' + str(
                predictions2) + '_' + str(predictions3) + '_orig.png',
                   orig_img_deprocessed)
            break

print("=== WARNING(about file name) ===")
print("(transformation name)_(1stPREDICTION)_(2ndPREDICTION)_(3rdPREDITION)")
