import os
import cv2
from .inference import detect_img_by_onnx


def getallpics(path, imgs):
    filelist = os.listdir(path)
    for file in filelist:
        # print(file,filecount)
        if file.lower().endswith('.jpg') or file.lower().endswith('.png') or file.lower().endswith(
                '.jpeg') or file.lower().endswith('.bmp'):
            imgs.append(os.path.join(path, file))
        elif os.path.isdir(os.path.join(path, file)):
            getallpics(os.path.join(path, file), imgs)
        else:
            pass

def cut_imgs(root):
    import shutil
    imgs = []
    getallpics(root,imgs)
    for img_pth in imgs:
        img = cv2.imread(img_pth)
        detect_img_by_onnx(img)
        for i,boxinfo in enumerate(predict_res["body_info"]):
            save_pth = img_pth.replace(root,root+"_cut").replace(".jpg","_%d.jpg"%i)
            dirs = "/".join(save_pth.split("/")[:-1])
            os.makedirs(dirs,exist_ok=True)
            # 根据框的位置将图片截出来
            box_pos = boxinfo["box"]
            box_img = img[box_pos[1]:box_pos[3], box_pos[0]:box_pos[2]]
            box_img = cv2.resize(box_img,(171,128))
            cv2.imwrite(save_pth,box_img)
            print(save_pth)
        predict_res["body_info"] = []
        predict_res["face_info"] = []
        # if i%5 == 0:
        #     new_img_pth = img_pth.replace("train","dev")
        #     dirs = "/".join(new_img_pth.split("/")[:-1])
        #     os.makedirs(dirs,exist_ok=True)
        #     shutil.move(img_pth,new_img_pth)
        #     print(img_pth,"-->",new_img_pth)