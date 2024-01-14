from transforms import normalize, normalize2
from tifffile import imread, imwrite
import numpy as np
import os
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr

def crop_image(inf3, top3, path_real, path_synth, transform_signal, w_start, w_end, h_start, h_end, output_dir, path_synth_mp=None):
    """
    get the patches in Figure 2/5

    inf3: a list of index of the inferior 3 channels
    top3: a list of index of the top 3 channels
    path_real: the path of the real image
    path_synth: the path of the synthetic image
    transform_signal: the list of transforms on the image
    w_start: the start coordinate of width
    w_end: the end coordinate of width
    h_start: the start coordinate of height
    h_end: the end coordinate of height
    output_dir: the output directory
    path_synth_mp: the path of the synthetic image in multi-panel experiment
    """

    im1 = imread(path_real).astype(np.float32) 
    transform_signal = [eval(t) for t in transform_signal]
    for t in transform_signal:
        im1 = t(im1)
    
    imwrite(os.path.join(output_dir, 'real_inf.tiff'), im1[inf3, w_start:w_end, h_start:h_end])
    imwrite(os.path.join(output_dir, 'real_top.tiff'), im1[top3, w_start:w_end, h_start:h_end])

    im2 = imread(path_synth).astype(np.float32)

    imwrite(os.path.join(output_dir, 'synth_inf.tiff'), im2[inf3, w_start:w_end, h_start:h_end])
    imwrite(os.path.join(output_dir, 'synth_top.tiff'), im2[top3, w_start:w_end, h_start:h_end])

    imwrite(os.path.join(output_dir, 'DIFF_inf.tiff'), np.abs(im1[inf3, w_start:w_end, h_start:h_end] - im2[inf3, w_start:w_end, h_start:h_end]))
    imwrite(os.path.join(output_dir, 'DIFF_top.tiff'), np.abs(im1[top3, w_start:w_end, h_start:h_end] - im2[top3, w_start:w_end, h_start:h_end]))

    if path_synth_mp != None:
        im3 = imread(path_synth_mp).astype(np.float32)
        
        imwrite(os.path.join(output_dir, 'MP_inf.tiff'), im3[inf3, w_start:w_end, h_start:h_end])
        imwrite(os.path.join(output_dir, 'MP_top.tiff'), im3[top3, w_start:w_end, h_start:h_end])

        imwrite(os.path.join(output_dir, 'DIFF_MP_inf.tiff'), np.abs(im1[inf3, w_start:w_end, h_start:h_end] - im3[inf3, w_start:w_end, h_start:h_end]))
        imwrite(os.path.join(output_dir, 'DIFF_MP_top.tiff'), np.abs(im1[top3, w_start:w_end, h_start:h_end] - im3[top3, w_start:w_end, h_start:h_end]))

def select_markers(idx1, idx2, path_real, path_synth, transform_signal):
    """
    get top3, inferior3 markers and their most similar observed markers

    idx1: a list of index of input channels in the final model
    idx2: a list of index of output channels in the final model
    path_real: the path of the real image
    path_synth: the path of the synthetic image
    transform_signal: the list of transforms on the image
    """
    im1 = imread(path_real).astype("f")
    transform_signal = [eval(t) for t in transform_signal]
    for t in transform_signal:
        im1 = t(im1)
    im2 = imread(path_synth)

    error = []
    for i in idx2:
        error.append(pearsonr(im1[i].flatten(), im2[i].flatten())[0])
    error = np.array(error)

    max_idx = error.argsort()[-3:][::-1]
    min_idx = error.argsort()[:3]

    idx2 = np.array(idx2)
    print("top3 markers: ", idx2[max_idx])
    print("inferior 3 markers: ", idx2[min_idx])

    print("the most similar input channel to the top3 markers: ")
    temp = []
    for i in idx1:
        temp.append(pearsonr(im1[idx2[max_idx][0]].flatten(), im1[i].flatten())[0])
    print(idx1[np.argmax(temp)])

    temp = []
    for i in idx1:
        temp.append(pearsonr(im1[idx2[max_idx][1]].flatten(), im1[i].flatten())[0])
    print(idx1[np.argmax(temp)])

    temp = []
    for i in idx1:
        temp.append(pearsonr(im1[idx2[max_idx][2]].flatten(), im1[i].flatten())[0])
    print(idx1[np.argmax(temp)])

    print("the most similar input channel to the inferior 3 markers: ")
    temp = []
    for i in idx1:
        temp.append(pearsonr(im1[idx2[min_idx][0]].flatten(), im1[i].flatten())[0])
    print(idx1[np.argmax(temp)])

    temp = []
    for i in idx1:
        temp.append(pearsonr(im1[idx2[min_idx][1]].flatten(), im1[i].flatten())[0])
    print(idx1[np.argmax(temp)])

    temp = []
    for i in idx1:
        temp.append(pearsonr(im1[idx2[min_idx][2]].flatten(), im1[i].flatten())[0])
    print(idx1[np.argmax(temp)])