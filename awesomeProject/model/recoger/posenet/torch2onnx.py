import onnxruntime
import torch
import time
import cv2
import numpy as np
import torchvision.transforms as T
import datetime
import posenet


def transform_to_onnx(weight_file):
    model = posenet.load_model(101,'/BOBO/cv_proj/model/_models/mobilenet_v1_101.pth')
    # pretrained_dict = torch.load(weight_file, map_location=torch.device('cuda'))
    # model.load_state_dict(pretrained_dict)

    input_names = ["input"]
    output_names = ['heatmaps_result', 'offsets_result',  'displacement_fwd_result', 'displacement_bwd_result']


    x = torch.randn((1, 3, 753, 353), requires_grad=True)
    onnx_file_name = "mobilenet_v1_101_none_3_dynamic.onnx"
    dynamic_axes = {"input": {0: "batch_size",1:'height',2:'width'},
                    "heatmaps_result": {0: "batch_size"},
                    "offsets_result": {0: "batch_size"},
                    "displacement_fwd_result": {0: "batch_size"},
                    "displacement_bwd_result": {0: "batch_size"}}
    # Export the model
    print('Export the onnx model ...')
    torch.onnx.export(model,
                      x,
                      onnx_file_name,
                      export_params=True,
                      opset_version=11,
                      do_constant_folding=True,
                      input_names=input_names, output_names=output_names,
                      dynamic_axes=dynamic_axes)

    print('Onnx model exporting done')
    return onnx_file_name

transform = T.Compose([
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
])

def test_onnx(onnx_path,img_pth):
    print("test onnx...")

    session = onnxruntime.InferenceSession(onnx_path)
    # session = onnx.load(onnx_path)
    print("The model expects input shape: ", session.get_inputs()[0].shape)
    IN_IMAGE_H = session.get_inputs()[0].shape[2]
    IN_IMAGE_W = session.get_inputs()[0].shape[3]

    # Input
    img = cv2.imread(img_pth)
    # resized = cv2.resize(img, (IN_IMAGE_W, IN_IMAGE_H), interpolation=cv2.INTER_LINEAR)
    img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_in = np.transpose(img_in, (2, 0, 1)).astype(np.float32)
    img_in = np.expand_dims(img_in, axis=0)
    img_in = np.concatenate((img_in, img_in), axis=0)
    img_in /= 255.0
    # 使用模型进行预测
    input_name = session.get_inputs()[0].name

    start_time = time.time()

    heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = session.run(None, {input_name: img_in})
    print(heatmaps_result,offsets_result,displacement_fwd_result,displacement_bwd_result)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("inference time:",total_time_str)



if __name__ == '__main__':
    print("Converting to onnx and running demo ...")
    # onnx_file_name = transform_to_onnx('_models/mobilenet_v1_101.pth')
    onnx_file_name = 'mobilenet_v1_101_none_3_dynamic.onnx'
    test_onnx(onnx_file_name,img_pth='/BOBO/cv_proj/model/hezaho.jpg')
