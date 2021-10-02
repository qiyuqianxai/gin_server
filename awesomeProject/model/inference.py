## cv系统 ##
import copy
from detector.yolov4.tool.torch_utils import *
from detector.yolov4.tool.utils import load_class_names
from detector.yolov4.models import Yolov4
import json
import onnxruntime
from recoger.posenet import posenet
from detector.face_dlib.face_reg_dlib import Face
from recoger.RepVGG.repvggA0 import repvggA0_onnx_inference
from tracker.deep_sort.deep_sort import DeepSort
import torch
import numpy as np
from recoger.C3D.network import C3D_model
import shutil
import cv2
import time
import threading

device_id = 0
torch.cuda.set_device(device_id)
# torch.backends.cudnn.benchmark = True
# 需要与web代码一致
# 预测的图片存放路径
predict_dir = "../static/predictions"
# 视频存放路径
videos_pth = "../videos"
# 信息交流的json的路径
message_json = "../message.json"
message = {}
# 权重路径
weight_pth = '_models'

# 当前image的预测结果
predict_res = {
    "image_name":"",
    "body_info":[],
    "face_info":[]
}
# 跳帧数
frame_skip = 1

class config():
    # base config
    save_pth = predict_dir
    os.makedirs(save_pth, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_cuda = True if device == "cuda" else False

    # yolov4 config
    n_classes = 2
    height = 416
    width = 416
    namesfile = r'detector/yolov4/data/custom.names'

    # weightfile = r"detector/yolov4/checkpoints/yolov4.pth"
    # detector = Yolov4(yolov4conv137weight=None, n_classes=n_classes, inference=True)
    # new_pre = {}
    # pretrained_dict = torch.load(weightfile)
    # for k, v in pretrained_dict.items():
    #     name = k.replace("module.","")
    #     print("name:",name)
    #     new_pre[name] = v
    # detector.load_state_dict(new_pre)
    # detector.to(device)


    option = onnxruntime.SessionOptions()
    option.log_severity_level = 3
    option.log_verbosity_level = 1
    session_yolov4 = onnxruntime.InferenceSession(os.path.join(weight_pth,"yolov4_-1_3_416_416_dynamic.onnx"), option)
    session_yolov4.set_providers(["CUDAExecutionProvider"],[{"device_id":device_id}])
    # session_yolov4 = onnxruntime.InferenceSession("yolov4_1_3_416_416_static.onnx",option, providers=[provider])
    # 框过滤阈值
    box_filter = 0.3

    # 行为分类repvgg
    onnx_path = os.path.join(weight_pth, 'repvgg_none_3_224_224_dynamic_4.onnx')
    action_cls1_session = onnxruntime.InferenceSession(onnx_path, option)
    action_cls1_session.set_providers(["CUDAExecutionProvider"],[{"device_id":device_id}])
    action_cls1_names = ['walk', 'stand', 'climb', 'wave']

    onnx_path = os.path.join(weight_pth, 'repvgg_none_3_224_224_dynamic_3.onnx')
    action_cls2_session = onnxruntime.InferenceSession(onnx_path, option)
    action_cls2_session.set_providers(["CUDAExecutionProvider"],[{"device_id":device_id}])
    action_cls2_names = ['phone', 'other', 'smoke']

    # posenet config
    posenet_weight = os.path.join(weight_pth,'mobilenet_v1_101.pth')
    posnet_model = posenet.load_model(101,posenet_weight)
    posnet_model.to(device)
    posnet_model.eval()
    # posnet_model.eval()
    # 关键点阈值
    key_point_filter = 0.3

    # 3D关键点识别模型
    # session_3D_pose = onnxruntime.InferenceSession("Resnet34_3inputs_448x448_20200609.onnx",option,providers=[provider])

    # 人脸及关键点检测
    expression_cls = ['nature','happy','amazing','angry']
    face_keypoints_detector = Face(os.path.join(weight_pth,'shape_predictor_68_face_landmarks.dat'))

    # 跟踪模型
    deepsort = DeepSort(model_path=os.path.join(weight_pth,"ckpt.t7"),use_cuda=use_cuda)

    # 视频行为识别
    # video_action_recoger = C3D_model.C3D(num_classes=101)
    # checkpoint = torch.load(os.path.join(weight_pth,'C3D-ucf101_epoch-20.pth.tar'), map_location=lambda storage, loc: storage)
    # video_action_recoger.load_state_dict(checkpoint['state_dict'])
    # video_action_recoger.to(device)
    # video_action_recoger.eval()
    onnx_path = "_models/C3D_none_3_16_112_112_dynamic_4.onnx"
    video_action_session = onnxruntime.InferenceSession(onnx_path, option)
    video_action_session.set_providers(["CUDAExecutionProvider"],[{"device_id":device_id}])
    vid_action_names = []
    with open('recoger/C3D/action_names.txt',"r",encoding="utf-8")as f:
        names = f.readlines()
        for name in names:
            name = name.replace("\n","").split(" ")[-1].strip()
            vid_action_names.append(name)

conf = config()
# 获取检测结果
# def detect_img(img):
#     model = conf.detector
#     sized = cv2.resize(img, (conf.width, conf.height))
#     sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
#     # for i in range(2):  # This 'for' loop is for speed check
#                         # Because the first iteration is usually longer
#     boxes = do_detect(model, sized, 0.4, 0.5, conf.use_cuda)
#     width = img.shape[1]
#     height = img.shape[0]
#     for box in boxes[0]:
#         box_info = {}
#         box_info["box"] = [int(box[0] * width) if int(box[0] * width) > 0 else 0 ,
#                            int(box[1] * height) if int(box[1] * height) > 0 else 0,
#                            int(box[2] * width) if int(box[2] * width) < width else width,
#                            int(box[3] * height) if int(box[3] * height) < height else height,
#                            float(box[5]),int(box[6])]
#         predict_res["body_info"].append(box_info)

# 获取检测结果

def detect_img_by_onnx(img):
    session = conf.session_yolov4
    IN_IMAGE_H = session.get_inputs()[0].shape[2]
    IN_IMAGE_W = session.get_inputs()[0].shape[3]

    # Input
    resized = cv2.resize(img, (IN_IMAGE_W, IN_IMAGE_H), interpolation=cv2.INTER_LINEAR)
    img_in = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    img_in = np.transpose(img_in, (2, 0, 1)).astype(np.float32)
    img_in = np.expand_dims(img_in, axis=0)
    img_in /= 255.0
    # print("Shape of the network input: ", img_in.shape)

    # Compute
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: img_in})

    boxes = post_processing(img_in, 0.5, 0.6, outputs)

    width = img.shape[1]
    height = img.shape[0]
    for box in boxes[0]:
        if float(box[5]) >= conf.box_filter:
            box_info = {}
            box_info["box"] = [int(box[0] * width) if int(box[0] * width) > 0 else 0 ,
                               int(box[1] * height) if int(box[1] * height) > 0 else 0,
                               int(box[2] * width) if int(box[2] * width) < width else width,
                               int(box[3] * height) if int(box[3] * height) < height else height,
                               float(box[5]),int(box[6])]
            if int(box[6]) == 1:
                predict_res["body_info"].append(box_info)
            else:
                predict_res["face_info"].append(box_info)

    # save_name = conf.save_name
    # class_names = load_class_names(conf.namesfile)
    # res_img = plot_boxes_cv2(img, boxes[0], save_name, class_names)
    # return res_img

# 获取posenet的关键点结果
@torch.no_grad()
def get_keypoints(img):
    # 关键点预测
    model = conf.posnet_model
    output_stride = model.output_stride

    for boxinfo in predict_res["body_info"]:
        # 根据框的位置将图片截出来
        box_pos = boxinfo["box"]
        box_img = img[box_pos[1]:box_pos[3],box_pos[0]:box_pos[2]]
        # cv2.imwrite("cut.jpg",box_img)
        try:
            input_image, draw_image, output_scale = posenet._process_input(
            box_img, scale_factor=1.0, output_stride=output_stride)
            # print(input_image.shape,draw_image.shape)
        except Exception as e:
            print(e)
            boxinfo["keypoints"] = []
            boxinfo["adjacent_keypoints"] = []
            continue


        input_image = torch.tensor(input_image).to(conf.device)
        heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = model(input_image)
        pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multiple_poses(
            heatmaps_result.squeeze(0),
            offsets_result.squeeze(0),
            displacement_fwd_result.squeeze(0),
            displacement_bwd_result.squeeze(0),
            output_stride=output_stride,
            max_pose_detections=16,
            min_pose_score=conf.key_point_filter)

        keypoint_coords *= output_scale

        cv_keypoints, adjacent_keypoints = posenet.draw_skel_and_kp(
                draw_image, pose_scores, keypoint_scores, keypoint_coords,
                min_pose_score=conf.key_point_filter, min_part_score=conf.key_point_filter)

        boxinfo["keypoints"] = cv_keypoints
        boxinfo["adjacent_keypoints"] = [pointline.tolist() for pointline in adjacent_keypoints]

# def get_3D_keypoints(img):
#     sess = conf.session_3D_pose
#     inputs = sess.get_inputs()
#     for boxinfo in predict_res["body_info"]:
#         # 根据框的位置将图片截出来
#         box_pos = boxinfo["box"]
#         img = img[box_pos[1]:box_pos[3],box_pos[0]:box_pos[2]]
#         # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img = cv2.resize(img, (448, 448))
#         img = img.astype(np.float32) / 255.0
#         img = img.transpose(2, 1, 0)
#         img = img[np.newaxis, ...]
#
#         pred_onx = sess.run(None, {
#             inputs[0].name: img,
#             inputs[1].name: img,
#             inputs[2].name: img
#         })
#
#         offset3D = np.squeeze(pred_onx[2])
#         heatMap3D = np.squeeze(pred_onx[3])
#         kps = np.zeros((24, 3), np.float32)
#         for j in range(0, 24):
#             # 找到第j个关节的28个特征图，并找到最大值的索引
#             joint_heat = heatMap3D[j * 28:(j + 1) * 28, ...]
#             if np.max(joint_heat) > 0.1:
#                 print(np.max(joint_heat))
#                 [x, y, z] = np.where(joint_heat == np.max(joint_heat))
#                 x = int(x[-1])
#                 y = int(y[-1])
#                 z = int(z[-1])
#                 # 通过heatmap的索引找到对应的offset图，并计算3D坐标的xyz值
#                 pos_x = offset3D[j * 28 + x, y, z] + x
#                 pos_y = offset3D[24 * 28 + j * 28 + x, y, z] + y
#                 pos_z = offset3D[24 * 28 * 2 + j * 28 + x, y, z] + z
#
#                 kps[j, 0] = pos_x
#                 kps[j, 1] = pos_y
#                 kps[j, 2] = pos_z
#             else:
#                 try:
#                     kps[j, 0] = kps[j - 1, 0]
#                     kps[j, 0] = kps[j - 1, 0]
#                     kps[j, 2] = kps[j - 1, 2]
#                 except:
#                     pass
#             # print("%f,%f,%f;" % (pos_x, pos_y, pos_z))
#         print("3D关键点",kps)

# 根据dlib获取结果
def get_face_detect_result(img):
    face_box = [boxinfo["box"] for boxinfo in predict_res['face_info']]
    res = conf.face_keypoints_detector.get_face(img,face_box)
    # todo:write result into predict_res
    predict_res['face_info'] = res

# 获取行为识别结果
def get_action_recog_result(img):
    action_cls1_session = conf.action_cls1_session
    action_cls2_session = conf.action_cls2_session
    for boxinfo in predict_res["body_info"]:
        # 根据框的位置将图片截出来
        box_pos = boxinfo["box"]
        box_img_1 = img[box_pos[1]:box_pos[3],box_pos[0]:box_pos[2]]
        box_img_2 = img[box_pos[1]:box_pos[1] + (box_pos[3]-box_pos[1])//2,box_pos[0]:box_pos[2]]
        action_cls_1 = repvggA0_onnx_inference(action_cls1_session,box_img_1, conf.action_cls1_names)
        action_cls_2 = repvggA0_onnx_inference(action_cls2_session,box_img_2, conf.action_cls2_names)
        boxinfo['img_action_cls'] = [action_cls_1,action_cls_2]
    # total_time_str = str(datetime.timedelta(seconds=int(total_time)))

cur_track_info = {}
# 获取跟踪结果
def get_tracker_result(img,save):
    def center_crop(frame):
        frame = frame[8:120, 30:142, :]
        return np.array(frame).astype(np.uint8)
    # do detection
    bbox = [boxinfo["box"] for boxinfo in predict_res["body_info"]]
    if bbox:
        detections = np.array(bbox)
        bbox_xyxy, cls_conf = detections[:,:4],detections[:,4]
        bbox_xywh = bbox_xyxy.copy()
        bbox_xywh[:, 2] = bbox_xyxy[:, 2] - bbox_xyxy[:, 0]
        bbox_xywh[:, 3] = bbox_xyxy[:, 3] - bbox_xyxy[:, 1]
        bbox_xywh[:, 0] = bbox_xywh[:, 0] + bbox_xywh[:, 2]/2
        bbox_xywh[:, 1] = bbox_xywh[:, 1] + bbox_xywh[:, 3]/2
        # select person class

        # bbox dilation just in case bbox too small, delete this line if using a better pedestrian detector
        # bbox_xywh[:, 3:] *= 1.2

        # do tracking
        im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        outputs = conf.deepsort.update(bbox_xywh, cls_conf, im)

        predict_res["body_info"] = []
        new_id = []
        for box in outputs:
            box_info = {}
            box_info["box"] = box
            predict_res["body_info"].append(box_info)

            # todo 保存同id的图片到内存中,为视频行为提供数据
            if save:
                id = box[-1]
                try:
                    cut_img = img[box[1]:box[3],box[0]:box[2]]
                    tmp_ = center_crop(cv2.resize(cut_img, (171, 128)))
                    tmp = tmp_ - np.array([[[90.0, 98.0, 102.0]]])
                except Exception as e:
                    # print(e,box)
                    continue
                if id not in cur_track_info:
                    cur_track_info[id] = []
                elif len(cur_track_info[id]) == 16:
                    cur_track_info[id].pop(0)
                cur_track_info[id].append(tmp)
                new_id.append(id)
        # 清除无用的id
        if save:
            all_keys = list(cur_track_info.keys())
            for key in all_keys:
                if key not in new_id:
                    cur_track_info.pop(key)

@torch.no_grad()
def get_video_action_recog_res():
    model = conf.video_action_recoger
    for id in cur_track_info:
        if len(cur_track_info[id]) == 16:
            inputs = np.array(cur_track_info[id]).astype(np.float32)
            inputs = np.expand_dims(inputs, axis=0)
            inputs = np.transpose(inputs, (0, 4, 1, 2, 3))
            inputs = torch.from_numpy(inputs)
            inputs = torch.autograd.Variable(inputs, requires_grad=False).to(conf.device)
            outputs = model(inputs)
            probs = torch.nn.Softmax(dim=1)(outputs)
            label = torch.max(probs, 1)[1].detach().cpu().numpy()[0]
            action_name = conf.vid_action_names[label]
            for boxinfo in predict_res["body_info"]:
                box = boxinfo["box"]
                if id == box[-1]:
                    boxinfo["action_cls"] = action_name
                    break

def get_video_action_recog_res_by_onnx():
    session = conf.video_action_session
    batch_inputs = []
    box_ids = []
    for id in cur_track_info:
        if len(cur_track_info[id]) == 16:
            inputs = np.array(cur_track_info[id]).astype(np.float32)
            inputs = np.expand_dims(inputs, axis=0)
            inputs = np.transpose(inputs, (0, 4, 1, 2, 3))
            batch_inputs.append(inputs)
            box_ids.append(id)

    # 送一个batch进行推理
    if batch_inputs != []:
        batch_inputs = np.array(batch_inputs)
        batch_inputs = np.vstack(batch_inputs)
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: batch_inputs})
        class_logits = outputs[0]
        # score = np.max(class_logits,axis=-1)
        results = np.argmax(class_logits, axis=-1)
        # print(result,score)
        for res,id in zip(results,box_ids):
            action_name = conf.vid_action_names[res]
            for boxinfo in predict_res["body_info"]:
                box = boxinfo["box"]
                if id == box[-1]:
                    boxinfo["action_cls"] = action_name
                    break

# 根据模型推理结果绘制图片
def draw_result_imgs(img,message):
    predict_res_copy = copy.deepcopy(predict_res)
    color = {
        "normal":(255,0,0),
        "warn":(0,140,255),
        "danger":(0,0,255)
    }
    img_name = predict_res_copy["image_name"]
    json_save_pth = os.path.join(conf.save_pth,img_name.replace(".jpg",".json"))
    # dirs = os.path.sep.join(json_save_pth.split(os.path.sep)[:-1])
    # os.makedirs(dirs,exist_ok=True)
    img_save_pth = os.path.join(conf.save_pth,img_name)
    class_names = load_class_names(conf.namesfile)
    res_img = np.copy(img)
    img_h,img_w,_ = res_img.shape

    # 绘制人体信息
    if message["req_body_detect"]:
        for boxinfo in predict_res_copy["body_info"]:
            # 绘制人体框
            box = boxinfo["box"]
            x1 = box[0]
            y1 = box[1]
            x2 = box[2]
            y2 = box[3]

            # 绘制行为识别结果
            action_type = "normal"
            action_class_1 = boxinfo['action_cls'] if 'action_cls' in boxinfo else ""
            action_class_2 = boxinfo['img_action_cls'] if 'img_action_cls' in boxinfo else ""
            action_class = action_class_1 + "|" + action_class_2[0] + "|" + action_class_2[1]
            if action_class and message["req_action_reg"]:
                # todo 分类行为类别
                res_img = cv2.rectangle(res_img, (x1, y2),
                                        (x2, int(y2-0.035*(y2-y1))),color[action_type], -2)
                res_img = cv2.putText(res_img, action_class, (x1, y2-2),
                                      cv2.FONT_HERSHEY_SIMPLEX, (y2-y1)/700, (255, 255, 255), 1)

            # 根据行为绘制人体框的颜色
            res_img = cv2.rectangle(res_img, (x1, y1),
                                    (x2, int(y1-0.035*(y2-y1))), color[action_type], -2)
            if message["person_track"]:
                identiy_id = box[-1]
                res_img = cv2.putText(res_img, str(identiy_id),(x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, (y2 - y1) / 700, (255, 255, 255), 1)
                pass
            else:
                res_img = cv2.putText(res_img, "person",
                                  (x1, y1-2),cv2.FONT_HERSHEY_SIMPLEX, (y2-y1)/700,(255, 255, 255), 1)
            # 人体框默认正常
            res_img = cv2.rectangle(res_img, (x1, y1), (x2, y2), color[action_type], 1)

            # 绘制关键点
            if message["req_body_keypoints"]:
                cv_keypoints = boxinfo["keypoints"]
                adjacent_keypoints = [np.array(points).astype(np.int32) for points in boxinfo["adjacent_keypoints"]]
                boxinfo["keypoints"] = [[points[0]+x1, points[1]+y1, points[2]] for points in cv_keypoints]
                cv_keypoints = [cv2.KeyPoint(points[0]+x1, points[1]+y1, (y2-y1)/500) for points in cv_keypoints]

                for points in adjacent_keypoints:
                    points[0][0] += x1
                    points[0][1] += y1
                    points[1][0] += x1
                    points[1][1] += y1
                if cv_keypoints:
                    res_img = cv2.drawKeypoints(
                        res_img, cv_keypoints, outImage=np.array([]), color=(255, 255, 0))
                if adjacent_keypoints:
                    # print(adjacent_keypoints)
                    res_img = cv2.polylines(res_img, adjacent_keypoints, isClosed=False, color=(255, 255, 0))

    # 绘制人脸及人脸关键点
    if message["req_face_detect"]:
        for face_info in predict_res_copy["face_info"]:
            face_coordinate = face_info['face_coordinate']
            expression = face_info['expression']
            attention_score = face_info['attention_score']
            x1,y1,x2,y2 = face_coordinate
            face_key_points = [cv2.KeyPoint(points[0], points[1], (y2 - y1) / 500) for points in
                               face_info["key_points"]]
            # 绘制人脸框
            res_img = cv2.rectangle(res_img, (x1, y1), (x2, y2), (0, 255, 0), 1)
            # 绘制人脸关键点
            if message["req_face_keypoints"]:
                res_img = cv2.drawKeypoints(
                    res_img, face_key_points, outImage=np.array([]), color=(255, 255, 255))
            # 绘制表情
            if message['req_expression_reg']:
                res_img = cv2.putText(res_img, 'expression:'+expression, (x1, int(y1+(y2-y1)*0.05)),
                                      cv2.FONT_HERSHEY_SIMPLEX, (y2-y1)/250, (0,165,255), 1)
            if message["req_attention_reg"]:
                res_img = cv2.putText(res_img, 'attention:' + str(round(attention_score,2)), (x1, int(y1+(y2-y1)*0.05*3)),
                                      cv2.FONT_HERSHEY_SIMPLEX, (y2-y1)/250, (0,165,255), 1)
            pass

    print(img_save_pth)
    cv2.imwrite(img_save_pth, res_img)

# 获取数据分析的结果
def get_data_analysis_res():
    predict_res_copy = copy.deepcopy(predict_res)
    img_name = predict_res_copy["image_name"]
    data_save_pth = os.path.join(conf.save_pth, img_name.replace('.jpg', '.json'))
    if os.path.exists(data_save_pth):
        return

    action_data_analysis = {cls: 0 for cls in conf.vid_action_names + conf.action_cls1_names + conf.action_cls2_names}
    expresssion_data_analysis = {cls:0 for cls in conf.expression_cls}
    attention_score_analysis = {"认真":0,"一般":0,"不认真":0}
    person_num = 0

    for bodyinfo in predict_res_copy["body_info"]:
        if "action_cls" in bodyinfo and bodyinfo["action_cls"] != "":
            if bodyinfo["action_cls"] in action_data_analysis:
                action_data_analysis[bodyinfo["action_cls"]] += 1
        if "img_action_cls" in bodyinfo and bodyinfo["img_action_cls"] != []:
            if bodyinfo["img_action_cls"][0] in action_data_analysis:
                action_data_analysis[bodyinfo["img_action_cls"][0]] += 1
            if bodyinfo["img_action_cls"][1] in action_data_analysis:
                action_data_analysis[bodyinfo["img_action_cls"][1]] += 1
        person_num += 1

    for faceinfo in predict_res_copy["face_info"]:
        expresssion_data_analysis[faceinfo["expression"]] += 1

        if int(faceinfo["attention_score"]) > 80:
            attention_score_analysis["认真"] += 1
        elif int(faceinfo["attention_score"]) > 60:
            attention_score_analysis["一般"] += 1
        else:
            attention_score_analysis["不认真"] += 1

    data_analysis = {"action_analysis":[],"expression_analysis":[],"attention_analysis":[],"person_num":person_num}
    # print(action_data_analysis)
    # print(expresssion_data_analysis)
    for k,v in action_data_analysis.items():
        data_analysis["action_analysis"].append([k,str(v)])
    for k,v in expresssion_data_analysis.items():
        data_analysis["expression_analysis"].append([k,str(v)])
    for k,v in attention_score_analysis.items():
        data_analysis["attention_analysis"].append([k,str(v)])

    with open(data_save_pth,"w",encoding="utf-8")as f:
        f.write(json.dumps(data_analysis,indent=4,ensure_ascii=False))

# 推理视频
cur_video_pth = ""

def detect_capture():
    global cur_video_pth,predict_res
    if cur_video_pth:
        if cur_video_pth == "capture":
            cap = cv2.VideoCapture(0)
            if os.path.exists(os.path.join(conf.save_pth,cur_video_pth.split(os.path.sep)[-1])):
                shutil.rmtree(os.path.join(conf.save_pth,cur_video_pth.split(os.path.sep)[-1]))
        else:
            cap = cv2.VideoCapture(cur_video_pth)
    else:
        print(cur_video_pth,"未注入value")
        return
    os.makedirs(os.path.join(conf.save_pth, cur_video_pth.split(os.path.sep)[-1]),exist_ok=True)
    old_video_pth = cur_video_pth
    if cap.isOpened():
        count = 0
        while cap.grab():
            if not message["Play_video"]:
                cap.release()
                cv2.destroyAllWindows()
                return
            if old_video_pth != cur_video_pth:
                cap.release()
                cv2.destroyAllWindows()
                return

            _, img = cap.retrieve()
            predict_res["image_name"] = os.path.join(cur_video_pth.split(os.path.sep)[-1],str(count)+".jpg")
            print(count)
            predict_res_json = os.path.join(conf.save_pth,predict_res["image_name"].replace(".jpg","_predict.json"))
            if not os.path.exists(predict_res_json):
                predict_res["body_info"] = []
                predict_res["face_info"] = []
                # 检测获取人体框
                t1 = time.time()
                detect_img_by_onnx(img)
                print("req_detect inference time:",time.time()-t1)

                # 跟踪
                t1 = time.time()
                get_tracker_result(img, count % 2 == 0)
                print("person_track inference time:", time.time() - t1)

                # 行为识别
                # 使用跟踪就用sense识别，否则用repvgg
                t1 = time.time()
                # todo C3D识别
                get_video_action_recog_res_by_onnx()
                # repvgg 识别
                get_action_recog_result(img)
                print("req_action_reg inference time:",time.time()-t1)

                # 根据框信息获取关键点
                t1 = time.time()
                get_keypoints(img)
                print("req_keypoints inference time:", time.time() - t1)

                # 获取人脸检测的结果
                t1 = time.time()
                get_face_detect_result(img)
                print("req_face_keypoints inference time:", time.time() - t1)

                with open(predict_res_json,"w",encoding="utf-8")as f:
                    f.write(json.dumps(predict_res,indent=4,ensure_ascii=False))

            with open(predict_res_json,"r",encoding="utf-8")as f:
                predict_res = json.load(f)
            # 绘制结果
            t1 = time.time()
            draw_result_imgs(img,message)
            # 数据分析
            get_data_analysis_res()
            print("drawing and data_analysis time:", time.time() - t1)
            count += 1
    cap.release()
    cv2.destroyAllWindows()

# 测试图片
def test_img(img_pth):
    predict_res["image_name"] = img_pth
    img = cv2.imread(img_pth)
    # 检测获取框
    t1 = time.time()
    detect_img_by_onnx(img)
    print("detect time:",time.time()-t1)
    # detect_img(img)
    # print(predict_res["body_info"])

    # 跟踪
    t2 = time.time()
    get_tracker_result(img)
    print("track time:",time.time()-t2)

    # 根据框信息获取关键点
    t3 = time.time()
    get_keypoints(img)
    print("keypoints time:",time.time()-t3)
    # print(predict_res["body_info"])

    # 获取3D节点
    # get_3D_keypoints(img)
    # 行为识别
    t4 = time.time()
    get_action_recog_result(img)
    print("action recog time:",time.time()-t4)

    # 获取face检测结果
    t5 = time.time()
    get_face_detect_result(img)
    print("face recog time:",time.time()-t5)

    # 绘制图片
    t6 = time.time()
    draw_result_imgs(img,{})
    print("drawing time:",time.time()-t6)

    # 获取分析数据
    t7 = time.time()
    get_data_analysis_res()
    print("data analysis time:",time.time()-t7)

    print("total time:",time.time()-t1)

# 监控message变化
def check_message():
    global message,cur_video_pth
    while True:
        if os.path.exists(message_json):
            with open(message_json,"r",encoding="utf-8")as f:
                message = json.load(f)
                cur_video_pth = os.path.join(videos_pth, message["video_name"])
        print("check。。。")
        time.sleep(1)


if __name__ == "__main__":

    ############  boot ############
    t1 = threading.Thread(target=check_message,args=())
    t1.start()
    print("t1 start")
    time.sleep(1)
    t2 = threading.Thread(target=detect_capture,args=())
    t2.start()
    print("t2 start")
    while True:
        if not t2.isAlive():
            t2 = threading.Thread(target=detect_capture, args=())
            t2.start()
            print("t2 restart")
        time.sleep(0.5)
    ############# test img #############
    # message = {
    #     "current_id": 493,
    #     "play_video": True,
    #     "video_name": "capture",
    #     "req_face_detect": True,
    #     "req_body_detect": True,
    #     "req_keypoints": True,
    #     "req_expression_reg": True,
    #     "req_action_reg": True,
    #     "person_track":True
    # }
    # img_name = "hezaho.jpg"
    # test_img(img_name)

    ############# test video #############
    # detect_capture("/BOBO/cv_proj/videos/baidu.mp4")
    # cut_imgs("/BOBO/datasets/video_data_imgs")






