# coding: utf-8
import os, numpy as np, re  # import modules
from tkinter import *
import tkinter as tk
from tkinter import messagebox, filedialog
import tkinter.filedialog as tkfd

import cv2
from natsort import natsorted
import glob
# from PIL import ImageGrab
import matplotlib.pyplot as plt, pydicom, nibabel as nib
from matplotlib import pyplot as plt
from PIL import Image, ImageTk

import itertools

import predict

# import loaditk, test , predict, functions#function modules

# 使用データ情報等の変数定義=========================
INPUT_IMAGE_SIZE = 320
DEPTH_SIZE = 64
De = 1
Di = 1
w1 = 1
counter = 0
ids = []
ids_red = []
input_sizeX, input_sizeY = 256, 256
# coordinates = np.zeros((int(images_list.shape[0]/2), input_sizeX, input_sizeY))
# coordinates_red = np.zeros((int(images_list.shape[0]/2), input_sizeX, input_sizeY))
value_slider = []
# doordinates = [0, 0]
save_name_initial = 'predict_initial'

# os.chdir("C:\\Users\\kenji ono\\Documents\\MATLAB\\GUI\\src")#ディレクトリを移動(絶対パス)
# os.chdir("C:\\Users\\Owner\\Documents\\GUI\\src")#ディレクトリを移動(絶対パス)
filename = 'S01*.nii.gz'
DIR_TESTS = os.path.join('..', 'TestData')
DIR_TESTS_ANS = os.path.join('..', 'TestData_Answer')
DIR_MODEL = os.path.join('..', 'Model')
weight = 'iFCN3c_S2-5_EP5_IT3_0-20'

# #データの読み込み
# (dep_list, inputs,file_names, deps,spacing_list,scan_size_list,images_list) = loaditk.load_itk(DIR_TESTS,filename,INPUT_IMAGE_SIZE,DEPTH_SIZE,scale=1)
# (dep_list_t, inputs_t,file_names_t, deps_t,spacing_list_t,scan_size_list_t,images_list_t)  = loaditk.load_itk(DIR_TESTS_ANS,filename,INPUT_IMAGE_SIZE,DEPTH_SIZE,scale=0)
# images_list = np.squeeze(images_list)
# images_list_t = np.squeeze(images_list_t)

# データの読み込み
case_number = 'Srr515_PV'
# initial_folder_path = "./../seko/refine/refine_test_initial/" + case_number + "/"
# label_folder_path = "./../seko/refine/refine_test_label/" + case_number + "/"
# origin_folder_path = "./../seko/refine/original_image/" + case_number + "/"
initial_folder_path = "./../GUI_RefineNet_npy/initial/" + case_number + "/"
label_folder_path = "./../GUI_RefineNet_npy/label/" + case_number + "/"
origin_folder_path = "./../GUI_RefineNet_npy/origin/" + case_number + "/"
if not os.path.exists(initial_folder_path): print("not exit")
initial_folder_list = natsorted(glob.glob(initial_folder_path + '*.npy'))
images_list = np.zeros((len(initial_folder_list), int(512 / 2), int(512 / 2)))
print(images_list.shape)
for i in range(len(initial_folder_list)):
    images_list[i, :, :] = np.load(
        initial_folder_path + "initial_pred" + case_number + str('{0:04d}'.format(i)) + ".npy")

if not os.path.exists(label_folder_path): print("not exit")
label_folder_list = natsorted(glob.glob(label_folder_path +'*.npy'))
images_label_list = np.zeros((len(label_folder_list), int(512/2), int(512/2)))
print("images_label_list.shape", images_label_list.shape)
for i in range(len(label_folder_list)):
    images_label_list[i, :, :] = np.load(label_folder_path + "initial_Ytest" + case_number + str('{0:04d}'.format(i)) + ".npy")

if not os.path.exists(origin_folder_path): print("not exit")
origin_folder_list = natsorted(glob.glob(origin_folder_path +'*.npy'))
images_origin_list = np.zeros((len(origin_folder_list), int(512/2), int(512/2)))
print("images_origin_list.shape", images_origin_list.shape)
for i in range(len(origin_folder_list)):
    images_origin_list[i, :, :] = np.load(origin_folder_path + "initial_Xtest" + case_number + str('{0:04d}'.format(i)) + ".npy")

# plt.imshow(images_list[5, :, :])
# plt.show()
# ids = np.zeros((int(images_list.shape[0]), 1))
# ids_red = np.zeros((int(images_list.shape[0]), 1))
coordinates = np.zeros((int(images_list.shape[0]), input_sizeX, input_sizeY))
coordinates_red = np.zeros((int(images_list.shape[0]), input_sizeX, input_sizeY))
ids_all = np.zeros((int(images_list.shape[0])))
print("coordinates.shape", coordinates.shape)
# images_list = natsorted(glob.glob(initial_folder_path +'*.npy'))
images_list_t = images_list
mages_label_list_t = images_label_list
#     initial_pred = np.zeros((len(initial_folder_list), int(512/2), int(512/2), 1))
#     print(initial_pred.shape)
# (dep_list, inputs,file_names, deps,spacing_list,scan_size_list,images_list) = loaditk.load_itk(DIR_TESTS,filename,INPUT_IMAGE_SIZE,DEPTH_SIZE,scale=1)
# (dep_list_t, inputs_t,file_names_t, deps_t,spacing_list_t,scan_size_list_t,images_list_t)  = loaditk.load_itk(DIR_TESTS_ANS,filename,INPUT_IMAGE_SIZE,DEPTH_SIZE,scale=0)
# images_list = np.squeeze(images_list)
# images_list_t = np.squeeze(images_list_t)

# #0点での初期予測
# if os.path.exists('../Outputs/predict_initial.npz'):
# 	predict_initial = np.load('../Outputs/predict_initial.npz')
# 	images_list_t = predict_initial['array_1']
# else:
# 	predict_initial = predict.predict(images_list,images_list_t,weight_load=weight,save_name = save_name_initial,De=De,Di=Di,w1=w1,coordinate3d_list=[0,0])
# 	images_list_t = predict_initial['array_1']

# インデックスマップの獲得．
# if os.path.exists('index_map.npz'):index_map = np.load('index_map.npz')
# else:
#     print("-----no-----")
# 	index_map = np.squeeze(np.array(functions.INDEX_MAP(DEPTH_SIZE,INPUT_IMAGE_SIZE,INPUT_IMAGE_SIZE)))
# 	np.savez_compressed('index_map',array_1=index_map)
print("-----no-----")


def delete_oval(event):
    canvas.delete(id)


def pushed():
    print("保存がクリックされました")
    print(ids)
    print(coordinates)
    global value_slider
    for i in range(value_slider):
        coordinate = cv2.flip(coordinates[i, :, :], 0)
        center = (int(input_sizeX / 2), int(input_sizeX / 2))
        # 回転角を指定
        angle = 270.0
        # スケールを指定
        scale = 1.0
        trans = cv2.getRotationMatrix2D(center, angle, scale)
        # アフィン変換
        coordinated = cv2.warpAffine(coordinate, trans, (input_sizeX, input_sizeY))
        print("----", np.max(coordinated))
        coordinated_nor = coordinated * 255
        coordinated_reshape = np.squeeze(coordinated_nor)
        print("----", np.max(coordinated_reshape))
        print(coordinated_reshape.shape)
        file = tkfd.asksaveasfilename(initialfile="FG" + str('{0:04d}'.format(i)) + ".png", title="保存場所を選択",
                                      filetypes=[("pngファイル", ".png")])
        print("-------", file)
        cv2.imwrite(file, coordinated_reshape)

        coordinate_red = cv2.flip(coordinates_red[i, :, :], 0)
        # アフィン変換
        coordinated_red = cv2.warpAffine(coordinate_red, trans, (input_sizeX, input_sizeY))
        print("----", np.max(coordinated_red))
        coordinated_red_nor = coordinated_red * 255
        coordinated_red_reshape = np.squeeze(coordinated_red_nor)
        print("----", np.max(coordinated_red_reshape))
        print(coordinated_red_reshape.shape)
        file = tkfd.asksaveasfilename(initialfile="BG" + str('{0:04d}'.format(i)) + ".png", title="保存場所を選択",
                                      filetypes=[("pngファイル", ".png")])
        print("-------", file)
        cv2.imwrite(file, coordinated_red_reshape)


def click(event):  # クリックされた場所に描画する
    slice_num = int(myval.get())  # slice numberをスライダーから獲得．
    global counter
    global ids
    counter = counter + 1
    size = 5  # 描写のサイズ
    x = event.x
    y = event.y
    id = canvas.create_oval(x - size, y - size, x + size, y + size,
                            fill="red", width=0, tag='oval')  # str(counter)

    place = 'num:' + str(counter) + ' ' + '(x,y,z)=(' + str(x) + ' ' + str(y) + ' ' + str(slice_num) + ' ):FG'
    print(place)
    lb.insert(tk.END, place)
    lb.pack()
    ids.append(id)

    global ent
    ent.delete(0, tk.END)
    ent.insert(tk.END, lb.size())
    return ids


def click_right(event):  # クリックされた場所に描画する
    slice_num = int(myval.get())  # slice numberをスライダーから獲得．
    global counter
    global ids
    counter = counter + 1
    size = 5  # 描写のサイズ
    x = event.x
    y = event.y
    id = canvas.create_oval(x - size, y - size, x + size, y + size,
                            fill="blue", width=0, tag='oval')  # str(counter)
    # print("clicked at", event.x, event.y,slice_num)
    place = 'num:' + str(counter) + ' ' + '(x,y,z)=(' + str(x) + ' ' + str(y) + ' ' + str(slice_num) + ' ):BG'
    print(place)
    lb.insert(tk.END, place)
    lb.pack()
    ids.append(id)
    # print('ids',ids)
    # canvas.tag_bind(id, '<1>', delete_oval)
    global ent
    ent.delete(0, tk.END)
    ent.insert(tk.END, lb.size())
    return ids


def motion(event):
    slice_num = int(myval1.get())
    txt.delete(0, tk.END)
    coordinate = 'x:' + str(event.x) + ' ' + 'y:' + str(event.y) + ' ' + 'z:' + str(slice_num)
    txt.insert(tk.END, coordinate)


def Return(event):
    pass


def open_file():
    root.withdraw()
    fTyp = [('niiファイルとnii.gzファイル', '*.nii;*.nii.gz')]
    # iDir='C:\\Users\\Owner\\Documents\\GUI\\TestData'
    iDir = 'C:\\Users\\Owner\\Documents\\GUI\\TestData'
    filename = filedialog.askopenfilename(filetypes=fTyp, initialdir=iDir)
    # (dep_list, inputs,file_names, deps,spacing_list,scan_size_list,images_list) = loaditk.load_itk(DIR_TESTS,filename,INPUT_IMAGE_SIZE,DEPTH_SIZE,scale=1)
    print("-----open_file-----")
    print(filename)


def denormalize_y(image):
    image = image * 255
    return image


def save_file():
    print(ids)
    print(coordinates)
    global value_slider
    # coordinate = cv2.rotate(coordinates, cv2.ROTATE_90_CLOCKWISE)
    # coordinate = coordinates.transpose(Image.ROTATE_90)
    #     global slice_num
    #     print(slice_num)
    coordinate = cv2.flip(coordinates[value_slider, :, :], 0)
    center = (int(input_sizeX / 2), int(input_sizeX / 2))
    # 回転角を指定
    angle = 270.0
    # スケールを指定
    scale = 1.0
    trans = cv2.getRotationMatrix2D(center, angle, scale)
    # アフィン変換
    coordinated = cv2.warpAffine(coordinate, trans, (input_sizeX, input_sizeY))
    print("----", np.max(coordinated))
    #     coordinated_nor = denormalize_y(coordinated)
    coordinated_nor = coordinated * 255
    coordinated_reshape = np.squeeze(coordinated_nor)
    print("----", np.max(coordinated_reshape))
    print(coordinated_reshape.shape)
    #     plt.imshow(coordinated)
    #     plt.show()
    # ImageGrab.grab().crop((x,y,x1,y1)).save("file path here")
    # savePath = filedialog.asksaveasfilename()
    #     savePath = filedialog.asksaveasfilename(initialfile="サンプル.png", title = "保存場所を選択",filetypes  = [("pngファイル", ".png")])
    file = tkfd.asksaveasfilename(initialfile="FG" + str('{0:04d}'.format(value_slider)) + ".png", title="保存場所を選択",
                                  filetypes=[("pngファイル", ".png")])
    print("-------", file)
    cv2.imwrite(file, coordinated_reshape)

    coordinate_red = cv2.flip(coordinates_red[value_slider, :, :], 0)
    # アフィン変換
    coordinated_red = cv2.warpAffine(coordinate_red, trans, (input_sizeX, input_sizeY))
    print("----", np.max(coordinated_red))
    coordinated_red_nor = coordinated_red * 255
    coordinated_red_reshape = np.squeeze(coordinated_red_nor)
    print("----", np.max(coordinated_red_reshape))
    print(coordinated_red_reshape.shape)
    file = tkfd.asksaveasfilename(initialfile="BG" + str('{0:04d}'.format(value_slider)) + ".png", title="保存場所を選択",
                                  filetypes=[("pngファイル", ".png")])
    print("-------", file)
    cv2.imwrite(file, coordinated_red_reshape)


#     cv2.imwrite(file+case_number + str('{0:04d}'.format(i)) + '.png', y_reshape)

#     if file:
#         with open(file, mode='w',encoding="utf-8") as f:
#             f.write(coordinated)
# canvas.postscript(file="outfile.ps")

# .save(savePath+".nii")

# pass
def close_disp():
    pass


def all_select():  # 全選択する
    lb.select_set(0, tk.END)
    canvas.itemconfig('oval', fill='blue')


def all_clear():  # 全クリアする
    lb.delete(0, tk.END)
    canvas.delete('oval')
    global ent
    ent.delete(0, tk.END)
    ent.insert(tk.END, lb.size())


print("-----no-----")
slice_num = []


def value_changed(*args):
    #     global value_slider
    #     print('value = %d' % myval.get())
    #     value_slider = (int(myval.get()))
    #     print("value_slider------------------", value_slider)
    #     #print("value", value)
    slice_num = int(myval.get())
    array = images_list[slice_num, :, :]
    array = array * 100
    # スライダーの値に応じたスライスを指定．
    global tkimg
    tkimg = ImageTk.PhotoImage(image=Image.fromarray(array))
    # キャンバスに画像を表示
    canvas.itemconfig(id, image=canvas.create_image(0, 0,
                                                    image=tkimg, anchor=tk.NW))
    # canvas.create_image(0, 0, image=tkimg, anchor=tk.NW)
    # canvas.create_oval(tag='oval')
    # canvas.update()
    slice_num_out = int(myval.get())
    array_out = images_list_t[slice_num_out, :, :]  # スライダーの値に応じたスライスを指定．
    array_out = array_out * 255
    global tkimg2
    tkimg2 = ImageTk.PhotoImage(image=Image.fromarray(array_out))
    # キャンバスに画像を表示
    out_cv.itemconfig(id, image=out_cv.create_image(0, 0,
                                                    image=tkimg2, anchor=tk.NW))


# 	myval.trace("w", value_changed)
# 	return value_slider

def value_changed_label(*args):
    slice_num = int(myval1.get())
    # 	images_subtra = images_label_list[slice_num,:,:]-images_list[slice_num,:,:]
    images_subtrab = images_list[slice_num, :, :] - images_label_list[slice_num, :, :]
    images_subtraf = images_label_list[slice_num, :, :] - images_list[slice_num, :, :]
    images_subtrab = np.where((images_subtrab) < 0, 0.0, images_subtrab)
    images_subtrab_RGB = np.zeros((1, 256, 256, 3))
    # 	images_subtrab_RGB[0, :, :, 3] = images_subtrab*255
    # 	images_subtrab_RGB[0, :, :, (1, 2)] = 0
    # 	images_subtrab_RGB[0, :, :, 0] = 255
    # 	images_subtrab_RGB = np.where((images_subtrab)>0, images_subtrab_RGB[:, :, :, 0]=1.0, images_subtrab_RGB[:, :, :, 0]=0.0)
    # # 	images_subtrab_RGB = cv2.cvtColor(images_subtrab,cv2.COLOR_GRAY2RGB)
    # 	plt.imshow(images_subtrab_RGB)
    # 	plt.show()
    images_subtraf = np.where((images_subtraf) < 0, 0.0, images_subtraf)
    # 	images_subtraf = np.where((images_subtraf)<0, 0.0, 255)
    # 	images_subtraf_RGB = cv2.cvtColor(images_subtraf,cv2.COLOR_GRAY2RGB)
    images_subtra = np.where((images_subtraf) < images_subtrab, images_subtrab * 255, (images_subtraf) * 100)
    # 	images_subtra = np.where((images_subtraf)<images_subtrab, cv2.cvtColor(images_subtrab,cv2.COLOR_GRAY2RGB), cv2.cvtColor(images_subtraf,cv2.COLOR_GRAY2RGB))
    array = images_subtra
    array = array
    # 	print(array)
    # 	print(array.shape)
    # 	array = images_subtrab_RGB[0, :, :, :]
    # 	print(array)
    # 	print(array.shape)
    # 	array = np.array(array)
    # 	print(array)
    # 	print(array.shape)
    # 	array = Image.fromarray(array) # RGBからPILフォーマットへ変換
    # 	print("np.max(array)", np.max(array))
    # array = images_subtrab_RGB
    # スライダーの値に応じたスライスを指定．

    global tkimg1
    tkimg1 = ImageTk.PhotoImage(image=Image.fromarray(array))
    # キャンバスに画像を表示
    canvas1.itemconfig(id, image=canvas1.create_image(0, 0,
                                                      image=tkimg1, anchor=tk.NW))

    slice_num = int(myval1.get())
    array1 = images_list[slice_num, :, :]
    array1 = array1 * 255
    array0 = images_origin_list[slice_num, :, :]
    array0 = array0 * 255
    array1 = cv2.addWeighted(array0, 1, array1, 0.2, 0)  # オーバーラップ表示（CT画像、初期画像）

    # スライダーの値に応じたスライスを指定．
    global tkimg3
    tkimg3 = ImageTk.PhotoImage(image=Image.fromarray(array1))
    # キャンバスに画像を表示
    canvas.itemconfig(id, image=canvas.create_image(0, 0,
                                                    image=tkimg3, anchor=tk.NW))

    slice_num_out = int(myval1.get())
    array_out = images_list_t[slice_num_out, :, :]  # スライダーの値に応じたスライスを指定．
    array_out = array_out * 255
    array_out = cv2.addWeighted(array0, 1, array_out, 0.2, 0)  # オーバーラップ表示（CT画像、初期画像）
    global tkimg2
    tkimg2 = ImageTk.PhotoImage(image=Image.fromarray(array_out))
    # キャンバスに画像を表示
    out_cv.itemconfig(id, image=out_cv.create_image(0, 0,
                                                    image=tkimg2, anchor=tk.NW))


def value_changed2(*args):
    global value_slider
    print('value = %d' % myval1.get())
    value_slider = (int(myval1.get()))
    print("value_slider------------------", value_slider)
    # print("value", value)


def value_changed_out(*args):
    slice_num_out = int(myval_out.get())
    array_out = images_list_t[slice_num_out, :, :]  # スライダーの値に応じたスライスを指定．
    array_out = array_out * 255
    global tkimg2
    tkimg2 = ImageTk.PhotoImage(image=Image.fromarray(array_out))
    # キャンバスに画像を表示
    out_cv.itemconfig(id, image=out_cv.create_image(0, 0,
                                                    image=tkimg2, anchor=tk.NW))


def deleteSelectedList():
    selectedIndex = tk.ACTIVE
    Index = lb.index(selectedIndex)
    selectedContents = lb.get(Index, last=None)  # 選択リストの内容
    # print('countents',selectedContents)
    out = 'num:'.join(selectedContents.split('num:')[1:])
    del_id = ids[int(out[:2]) - 1]  # 削除する描写のid
    del_tag = str(del_id)  # 削除する描写のidのtag
    # print('lb_index',Index,'del_tag',del_tag,'del_id',del_id)
    canvas.itemconfig(del_id, tag=del_tag)  # 削除のために，描写idのtagを変更
    lb.delete(selectedIndex)  # 選択リストを削除
    canvas.delete(del_tag)  # 選択リストに対応する描写を削除
    global ent
    ent.delete(0, tk.END)
    ent.insert(tk.END, lb.size())


print("-----no-----")


# 実行ボタン===============================
def exe_pushed():
    print("実行がクリックされました")
    #     coordinate3d_list = []
    #     coordinate3d_list_pos = []
    #     coordinate3d_list_neg = []
    #     lb_len = lb.size()
    #     for i in range(lb_len):
    #         lb_contents_i = lb.get(i,last=None)
    #         coordinate3d = ''.join(lb_contents_i.split('=(')[1:])
    #         coordinate3d = coordinate3d.split(' ')
    #         if 'FG' in lb_contents_i: coordinate3d_list_pos.append([coordinate3d[0],coordinate3d[1],coordinate3d[2]])
    #         else: coordinate3d_list_neg.append([coordinate3d[0],coordinate3d[1],coordinate3d[2]])
    #     print('選択された点群','pos:', coordinate3d_list_pos,'neg:',coordinate3d_list_neg)
    #     coordinate3d_list.append(coordinate3d_list_pos)
    #     coordinate3d_list.append(coordinate3d_list_neg)
    global coordinates
    print("--------------------------------------------------------------", np.max(coordinates))
    #     plt.imshow(coordinates[22, :, :])
    #     plt.show()
    #     coordinatess = coordinates
    #     coordinatess_red = coordinates_red
    #     center = (int(input_sizeX/2), int(input_sizeX/2))
    #     #回転角を指定
    #     angle = 270.0
    #     #スケールを指定
    #     scale = 1.0
    #     trans = cv2.getRotationMatrix2D(center, angle , scale)
    #     #アフィン変換
    #     coordinated = cv2.warpAffine(coordinatess, trans, (input_sizeX,input_sizeY))
    #     coordinated_red = cv2.warpAffine(coordinatess_red, trans, (input_sizeX,input_sizeY))
    #
    #     plt.imshow(coordinated[22, :, :])
    #     plt.show()

    #     print("----", np.max(coordinated))
    #     coordinated_nor = coordinated*255
    #     coordinated_reshape = np.squeeze(coordinated_nor)
    #     print("----", np.max(coordinated_reshape))
    #     print(coordinated_reshape.shape)

    #     R_pred = predict.predict(coordinated[:, :, :, np.newaxis], coordinated_red[:, :, :, np.newaxis])
    R_pred = predict.predict(coordinates[:, :, :, np.newaxis], coordinates_red[:, :, :, np.newaxis])
    print(R_pred.shape)
    #     R_pred = np.squeeze(R_pred )
    #     print(R_pred.shape)
    #     plt.imshow(R_pred[11, :, :])
    #     plt.show()

    global tkimg2
    global images_list_t
    global images_list
    # 予測関数開始
    # (predt,inputs_shape,inputs,best_OC,predt_OC_list,images_list_t,preds_list,file_names,DSC_list,new_seed_place) \
    #     test =predict.predict(images_list,images_list_t,weight_load=weight,save_name = '25ifcn_3c_s1',De=De,Di=Di,w1=w1,coordinate3d_list=coordinate3d_list)
    #     images_list_t = test
    slice_num_out = int(myval1.get())
    print(slice_num_out)
    array_out = R_pred[slice_num_out, :, :, 0]  # スライダーの値に応じたスライスを指定．
    print("-------------------444444444444444444-------------------------", array_out.shape)
    images_list_t = R_pred[:, :, :, 0]

    array_out = array_out * 255
    array0 = images_origin_list[slice_num_out, :, :]
    array0 = array0 * 255
    array_out = cv2.addWeighted(array0, 1, array_out, 0.2, 0)  # オーバーラップ表示（CT画像、初期画像）

    global tkimg2
    tkimg2 = ImageTk.PhotoImage(image=Image.fromarray(array_out))
    out_cv.itemconfig(id, image=out_cv.create_image(0, 0, image=tkimg2, anchor=tk.NW))


def coler_blue_pushed():
    print("色変更がクリックされました")
    global seed_color
    seed_color = "blue"
    canvas.config(cursor="arrow")


def coler_red_pushed():
    print("色変更がクリックされました")
    global seed_color
    seed_color = "red"
    canvas.config(cursor="arrow")


def paint_delete():
    print("消しゴムがクリックされました")
    global seed_color
    seed_color = "white"
    canvas.config(cursor="circle")

#
# def eraser_size_change1():
#     global eraser_size
#     eraser_size = 1
#
#
# def eraser_size_change2():
#     global eraser_size
#     eraser_size = 2
#
#
# def eraser_size_change3():
#     global eraser_size
#     eraser_size = 3


def eraser_size_change():
    global eraser_size
    eraser_size = iv1.get()
    print("eraser_size", eraser_size)


def slice_mover(event):
    print('wheel')
    print(event.delta)


def paint(event):
    global ids
    global ids_red
    global value_slider
    size = 2  # 描写のサイズ
    print("---paint(event)---")
    print("------------------------------------", value_slider)
    #     if eraser_on:
    #         paint_color = 'white'
    #     else:
    #         paint_color = 'black'
    paint_color = 'black'
    #  seed_coler = "red"

    old_x = None
    old_y = None

    if old_x and old_y:
        test_canvas.create_line(old_x, old_y, event.x, event.y, width=5.0, fill=paint_color, capstyle=tkinter.ROUND,
                                smooth=tkinter.TRUE, splinesteps=36)
        draw.line((old_x, old_y, event.x, event.y), fill=paint_color, width=5)
    old_x = event.x
    old_y = event.y
    x = event.x
    y = event.y
    #     id = canvas.create_oval(x-size, y-size, x+size, y+size, fill="red", width=0,tag='oval')#str(counter)
    if (seed_color == "blue"):
        #  ids = ids_all[value_slider]
        id = canvas.create_oval(x - size, y - size, x + size, y + size, fill=seed_color, width=0,
                                tag='oval_blue' + str(x - size) + str(y - size))  # str(counter)
        print("id--blue----", id)
        print(x - size, y - size, x + size, y + size)
        # ImageGrab.grab().crop((x-size, y-size, x+size, y+size)).save("file path here")
        if (x - size > 0 and y - size > 0):
            #         #xx = input_sizeX - (x-size)
            #         yy = input_sizeY - (y-size)
            coordinates[value_slider, x - size, y - size] = 1
        #             print(coordinates)
        #         print(ids[value_slider])
        #         np.append(ids[value_slider], (id))
        #         print(ids[value_slider])
        ids.append(id)
        #         np.append(ids[value_slider], (id))
        #         ids_all[value_slider] = ids
        print("ids-----", ids)
        # canvas.config(cursor="arrow")
    elif (seed_color == "red"):
        id = canvas.create_oval(x - size, y - size, x + size, y + size, fill=seed_color, width=0,
                                tag='oval_red' + str(x - size) + str(y - size))  # str(counter)
        print("id--red----", id)
        print(x - size, y - size, x + size, y + size)
        if (x - size > 0 and y - size > 0):
            coordinates_red[value_slider, x - size, y - size] = 1
        #             print(coordinates_red)
        ids_red.append(id)
        # canvas.config(cursor="arrow")
    elif (seed_color == "white"):
        print(seed_color)
        if (eraser_size == 1):
            l1 = [x - size - 3, x - size - 2, x - size - 1, x - size, x - size + 1, x - size + 2, x - size + 3]
            l2 = [y - size - 3, y - size - 2, y - size - 1, y - size, y - size + 1, y - size + 2, y - size + 3]
            eraser_s = 3
        elif (eraser_size == 2):
            l1 = [x - size - 4, x - size - 3, x - size - 2, x - size - 1, x - size, x - size + 1, x - size + 2,
                  x - size + 3, x - size + 4]
            l2 = [y - size - 4, y - size - 3, y - size - 2, y - size - 1, y - size, y - size + 1, y - size + 2,
                  y - size + 3, y - size + 4]
            eraser_s = 4
        elif (eraser_size == 3):
            l1 = [x - size - 5, x - size - 4, x - size - 3, x - size - 2, x - size - 1, x - size, x - size + 1,
                  x - size + 2, x - size + 3, x - size + 4, x - size + 5]
            l2 = [y - size - 5, y - size - 4, y - size - 3, y - size - 2, y - size - 1, y - size, y - size + 1,
                  y - size + 2, y - size + 3, y - size + 4, y - size + 5]
            eraser_s = 5
        print(str(x - size), str(y - size))
        for i, j in itertools.product(l1, l2):
            canvas.delete("oval_blue" + str(i) + str(j))   #canvas上のpaint削除
            canvas.delete("oval_red" + str(i) + str(j))
            coordinates[value_slider, i, j] = 0            #変数内のseed削除
            coordinates_red[value_slider, i, j] = 0
    #         canvas.delete("oval_blue"+str(x-size)+str(y-size))
    #         canvas.delete("oval_red"+str(x-size)+str(y-size))
    #     canvas.create_oval(x - size -eraser_s, y - size-eraser_s, x + size+eraser_s, y + size+eraser_s, fill=seed_color, width=0,
    #                        tag='oval1' + str(x - size) + str(y - size))  # str(counter)

        # canvas.config(cursor="icon")
        # canvas.config(cursor="circle")


    return ids, ids_red


global seed_color
global eraser_size
eraser_size = 1


# global count


#   slice_num = int(myval.get()) #slice numberをスライダーから獲得．
# 	global counter
# 	global ids
# 	counter = counter + 1
# 	size = 5 #描写のサイズ
# 	x = event.x
# 	y = event.y
# 	id = canvas.create_oval(x-size, y-size, x+size, y+size,
# 	fill="red", width=0,tag='oval')#str(counter)
# 	#print("clicked at", event.x, event.y,slice_num)
# 	place = 'num:'+str(counter) +' '+'(x,y,z)=('+str(x)+' '+str(y)+' '+str(slice_num)+ ' ):FG'
# 	print(place)
# 	lb.insert(tk.END, place)
# 	lb.pack()
# 	ids.append(id)
# 	#print('ids',ids)
# 	#canvas.tag_bind(id, '<1>', delete_oval)
# 	global ent
# 	ent.delete(0, tk.END)
# 	ent.insert(tk.END,lb.size())
# 	return ids

def white(event):
    global ids
    global ids_red
    global value_slider
    size = 2  # 描写のサイズ
    print("---paint(event)---")
    print("------------------------------------", value_slider)
    paint_color = 'black'

    old_x = None
    old_y = None

    if old_x and old_y:
        test_canvas.create_line(old_x, old_y, event.x, event.y, width=5.0, fill=paint_color, capstyle=tkinter.ROUND,
                                smooth=tkinter.TRUE, splinesteps=36)
        draw.line((old_x, old_y, event.x, event.y), fill=paint_color, width=5)
    old_x = event.x
    old_y = event.y
    x = event.x
    y = event.y

    if (seed_coler == "white"):
        id = canvas.create_oval(x - size, y - size, x + size, y + size, fill=seed_coler, width=0,
                                tag='oval')  # str(counter)
        print("id------", id)
        print(x - size, y - size, x + size, y + size)
        if (x - size > 0 and y - size > 0):
            coordinates[value_slider, x - size, y - size] = 1
        ids.append(id)
        ids_all[value_slider] = ids
        print("ids-----", ids)
    else:
        id = canvas.create_oval(x - size, y - size, x + size, y + size, fill=seed_coler, width=0,
                                tag='oval')  # str(counter)
        print(x - size, y - size, x + size, y + size)
        if (x - size > 0 and y - size > 0):
            coordinates_red[value_slider, x - size, y - size] = 1

        ids_red.append(id)
    return ids, ids_red


def reset(event):
    old_x, old_y = None, None


# ====================ルートの配置設定=======================
root = Tk()
root.title("インタラクティブセグメンテーションデモ")  # メインウィンドウのタイトルを変更
root.geometry("800x640")  # メインウィンドウサイズ
button = tk.Button(root, text="保存", command=pushed)  # ボタンを設置．
button.grid()
print("-----no-----")
# マウス座標表示ウィンドウ設置==================
lbl = tk.Label(text='マウス座標:')
lbl.place(x=120, y=10)
txt = tk.Entry(width=20)
txt.place(x=180, y=10)
lbl_ext = tk.Label(text='入力画像')  # 実行結果テキストを表示
lbl_ext.place(x=50, y=10)

# 選択点表示スクロールウィンドウ=========================
# frame=tk.Frame()
# frame.grid(row=0,sticky="we")
# frame.place(x=30, y=380)
# scroll=tk.Scrollbar(frame)
# scroll.pack(side=tk.RIGHT,fill="y")
# list_value=tk.StringVar()
# lb=tk.Listbox(frame,height=15,width=30,listvariable=list_value,selectmode="single",yscrollcommand=scroll.set)
# lb.configure(selectmode="single")
# frame_button=tk.Frame()#ボタンの作成フレーム２（下側：ボタンを格納する）
# frame_button.grid(row=1,sticky="we")
# frame_button.place(x=230, y=600)
# Button_allclear=tk.Button(frame_button,text="全クリア",command= all_clear)
# Button_allclear.grid(row=1,column=1)
# Button_delete = tk.Button(frame_button,text=u'選択項目削除', command=lambda: deleteSelectedList()) #関数に引数を渡す場合は、commandオプションとlambda式を使う
# Button_delete.grid(row=1,column=2)
# lbl_ext = tk.Label(text='選択した点の座標リスト')#実行結果テキストを表示
# lbl_ext.place(x=30, y=360)
# ent = tk.Entry(width=5)
# ent.place(x=230, y=360)
# lbl_ext = tk.Label(text='合計点数:')
# lbl_ext.place(x=170, y=360)


# 画像を表示するためのキャンバスの作成（黒で表示）==========================
canvas = tk.Canvas(bg="white", width=256, height=256)
canvas.place(x=50, y=30)  # 左上の座標を指定
# array = np.ones((int(180), int(180)))
# # array = images_list[int(images_list.shape[0]/2),:,:]
# array = array*100
# array = images_list[int(images_list.shape[0]/2),:,:]
array = images_list[int(images_list.shape[0] / 2), :, :]
array = array * 100
tkimg = ImageTk.PhotoImage(image=Image.fromarray(array))
canvas.create_image(0, 0, image=tkimg, anchor=tk.NW)  # キャンバスに画像を表示する。第一引数と第二引数は、x, yの座標
# size = 10
# canvas.create_rectangle(0-size, 0-size, 0+size, 0+size,fill='red') #出力を画像データに重ねて透過表示したい．

# 画像を表示するためのキャンバスの作成（黒で表示）==========================
canvas1 = tk.Canvas(bg="white", width=256, height=256)
canvas1.place(x=50, y=320)  # 左上の座標を指定
lbl_ext = tk.Label(text='補助')  # 実行結果テキストを表示
lbl_ext.place(x=50, y=300)
# array = np.ones((int(180), int(180)))
# # array = images_list[int(images_list.shape[0]/2),:,:]
# array = array*100
# array = images_list[int(images_list.shape[0]/2),:,:]

array1 = images_label_list[int(images_label_list.shape[0] / 2), :, :]
array1 = array1 * 100
# plt.imshow(array1)
# plt.show()
tkimg = ImageTk.PhotoImage(image=Image.fromarray(array1))
canvas1.create_image(0, 0, image=tkimg, anchor=tk.NW)  # キャンバスに画像を表示する。第一引数と第二引数は、x, yの座標

# color変更
button = tk.Button(root, text="Foreground Seed", fg='blue', command=coler_blue_pushed)
button.place(x=530, y=400, height=80, width=100)
button = tk.Button(root, text="Background Seed", fg='red', command=coler_red_pushed)
button.place(x=630, y=400, height=80, width=100)
# 削除・消しゴム
button = tk.Button(root, text="消しゴム", command=paint_delete)
button.place(x=530, y=500, height=80, width=100)
# button = tk.Button(root, text="サイズ1", command=eraser_size_change1)
# button.place(x=530, y=580, height=40, width=50)
# button = tk.Button(root, text="サイズ2", command=eraser_size_change2)
# button.place(x=580, y=580, height=40, width=50)
# button = tk.Button(root, text="サイズ3", command=eraser_size_change3)
# button.place(x=630, y=580, height=40, width=50)

# 実行群と出力キャンパス作成, predict_initialから，点追加毎の実行結果を更新表示
button = tk.Button(root, text="実行", command=exe_pushed)
button.place(x=630, y=500, height=80, width=100)
lbl_ext = tk.Label(text='実行結果')  # 実行結果テキストを表示
lbl_ext.place(x=450, y=10)
out_cv = tk.Canvas(root, bg="white", width=256, height=256)  # 出力画像を表示
out_cv.place(x=450, y=30)  # 左上の座標を指定
# tkimg2 =  ImageTk.PhotoImage(image=Image.fromarray(array))
out_cv.create_image(0, 0, image=tkimg, anchor=tk.NW)

# def value_changed(*args):
#     global value_slider
#     print('value = %d' % myval.get())
#     value_slider = (int(myval.get()))
#     print("value_slider------------------", value_slider)
#     #print("value", value)
#     return value_slider

# frame_slider = tk.Frame()
# myval = DoubleVar()# スケールの作成，スライダーでスライス番号を移動
# #myval1 = DoubleVar()# スケールの作成，スライダーでスライス番号を移動
# myval.trace("w", value_changed)
# myval.trace("w", value_changed2)
# #myval1.trace("w", value_changed_label)
# slider = tk.Scale(frame_slider, label='slice num(Input)', orient='h', from_=0, to=63,variable=myval)
# #slider1 = tk.Scale(frame_slider, label='slice num(Input)', orient='h', from_=0, to=63,variable=myval1)
# frame_slider.grid(sticky=(N,W,S,E))
# #frame_slider.grid(sticky1=(N,W,S,E))
# frame_slider.place(x=270, y=360)
# # slider.set(str(int(images_list.shape[0]/2)))
# slider.set(str(int(images_list.shape[0]/2)))
# # slider.set(str(int(images_label_list.shape[0]/2)))
# print("--------", str(int(images_list.shape[0]/2)))
# #print('value = %d' % myval.get())
# # slider.set(str(int(0000)))
# slider.pack()

frame_slider1 = tk.Frame()
myval1 = DoubleVar()  # スケールの作成，スライダーでスライス番号を移動
myval1.trace("w", value_changed_label)
myval1.trace("w", value_changed2)
# slider1 = tk.Scale(frame_slider1, label='slice num(Input)', orient='h', from_=0, to=63, variable=myval1)
Static1 = tk.Label(text=u'slice num', foreground='#ff0000', background='#ffaacc')
Static1.place(x=320, y=30)
# slider1 = tk.Scale(frame_slider1, label='slice num', orient="vertical", from_=0, to=int(images_label_list.shape[0]-1), variable=myval1, tickinterval=1)
slider1 = tk.Scale(frame_slider1, orient="vertical", from_=0, to=int(images_label_list.shape[0]-1), variable=myval1, tickinterval=1)
frame_slider1.grid(sticky=(N, W, S, E))
frame_slider1.place(x=310, y=60, width=70, height=325)
# slider.set(str(int(images_list.shape[0]/2)))
slider1.set(str(int(images_label_list.shape[0] / 2)))
# slider.set(str(int(images_label_list.shape[0]/2)))
print("--------", str(int(images_label_list.shape[0] / 2)))
# print('value = %d' % myval.get())
# slider.set(str(int(0000)))
slider1.pack()
#
# frame = tk.Label(root, text="6: RadioButton").grid(row=400, column=2, sticky="e")
frame = tk.Label(root, text="消しゴムサイズ")
frame_for_radio = tk.Frame(root)
frame_for_radio.grid(row=400, column=1, padx=300, pady=500)
iv1 = tk.IntVar()
iv1.set(1)
tk.Radiobutton(frame_for_radio, text="サイズ：１", value=1, variable=iv1, command=eraser_size_change).pack()
tk.Radiobutton(frame_for_radio, text="サイズ：２", value=2, variable=iv1, command=eraser_size_change).pack()
tk.Radiobutton(frame_for_radio, text="サイズ：３", value=3, variable=iv1, command=eraser_size_change).pack()
frame.place(x=350, y=500)
# Button=tk.Button(text="判定",command=eraser_size_change)
# Button.pack()

# メニューバー作成================================
men = tk.Menu(root)
root.config(menu=men)  # メニューバーを画面にセット
menu_file = tk.Menu(root)  # メニューに親メニュー（ファイル）を作成する
men.add_cascade(label='ファイル', menu=menu_file)
menu_file.add_command(label='開く(入力画像)', command=open_file)  # 親メニューに子メニュー（開く・閉じる）を追加する
menu_file.add_separator()
menu_file.add_command(label='保存', command=save_file)
menu_file.add_command(label='閉じる', command=close_disp)
# #メニューバー作成================================
# men_eraser = tk.Menu(root)
# root.config(menu=men_eraser) #メニューバーを画面にセット
# menu_file_eraser = tk.Menu(root) #メニューに親メニュー（ファイル）を作成する
# men.add_cascade(label='サイズ', menu=menu_file_eraser)
# menu_file.add_command(label='1', command=eraser_size_change1) #親メニューに子メニュー（開く・閉じる）を追加する
# menu_file.add_command(label='2', command=eraser_size_change2)
# menu_file.add_command(label='3', command=eraser_size_change3)
#
# frame=tk.LabelFrame(root,text="ラジオボタン",foreground="green")
# frame.pack()
# var=tk.IntVar()
# var.set(0)
# radio_0=tk.Radiobutton(frame,value=0,variable=var,text="ラジオボタン0")
# radio_0.pack()
# radio_1=tk.Radiobutton(frame,value=1,variable=var,text="ラジオボタン1")
# radio_1.pack()
# radio_2=tk.Radiobutton(frame,value=2,variable=var,text="ラジオボタン2")
# radio_2.pack()
# radio_3=tk.Radiobutton(frame,value=3,variable=var,text="ラジオボタン3")
# radio_3.pack()

print("-----no-----")
# イベントを設定する====================================
# canvas.bind("<Button-1>", click)
# canvas.bind("<Button-3>", click_right)
canvas.bind("<Motion>", motion)
canvas.bind("<Key-Return>", Return)
canvas.bind("<MouseWheel>", slice_mover)
canvas.bind('<B1-Motion>', paint)
canvas.bind('<ButtonRelease-1>', reset)

# 出力群===================
'''
frame_slider2 = tk.Frame()# スケールの作成，スライダーでスライス番号を移動
myval_out = DoubleVar()
myval_out.trace("w", value_changed_out)
slider_out = tk.Scale(frame_slider2, label='slice num(Output)', orient='h', 
from_=0, to=63,variable=myval_out)
frame_slider2.grid(sticky=(N,W,S,E))
frame_slider2.place(x=550, y=360)
slider_out.set(str(int(images_list_t.shape[0]/2))) 
slider_out.pack()
'''
print("-----mainloop-----")
# メインウィンドウを表示
root.mainloop()
