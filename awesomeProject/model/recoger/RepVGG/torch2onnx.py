import onnxruntime
import torch
from PIL import Image
import time
from nets import repvgg,repvgga0
import cv2
import numpy as np
import torchvision.transforms as T
import datetime


def transform_to_onnx(weight_file, n_classes, IN_IMAGE_H, IN_IMAGE_W):
    model = repvgga0(num_class=n_classes)
    pretrained_dict = torch.load(weight_file, map_location=torch.device('cuda'))
    model.load_state_dict(pretrained_dict)

    input_names = ["input"]
    output_names = ['class_name']


    x = torch.randn((1, 3, IN_IMAGE_H, IN_IMAGE_W), requires_grad=True)
    onnx_file_name = "repvgg_none_3_{}_{}_dynamic_{}.onnx".format(IN_IMAGE_H, IN_IMAGE_W, n_classes)
    dynamic_axes = {"input": {0: "batch_size"}, "class_name": {0: "batch_size"}}
    # Export the model
    print('Export the onnx model ...')
    torch.onnx.export(model,
                      x,
                      onnx_file_name,
                      export_params=True,
                      opset_version=11,
                      do_constant_folding=True,
                      input_names=input_names,
                      output_names=output_names,
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
    resized = cv2.resize(img, (IN_IMAGE_W, IN_IMAGE_H), interpolation=cv2.INTER_LINEAR)
    img_in = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    img_in = np.transpose(img_in, (2, 0, 1)).astype(np.float32)
    img_in = np.expand_dims(img_in, axis=0)
    img_in = np.concatenate((img_in, img_in), axis=0)
    img_in /= 255.0
    # 使用模型进行预测
    input_name = session.get_inputs()[0].name

    start_time = time.time()

    outputs = session.run(None, {input_name: img_in})
    class_logits = outputs[0]
    result = np.argmax(class_logits,axis=-1)
    print(result)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("inference time:",total_time_str)



if __name__ == '__main__':
    print("Converting to onnx and running demo ...")
    onnx_file_name = transform_to_onnx('task_2_action_reg_0.70_.pth',3,224,224)
    # onnx_file_name = 'repvgg_none_3_224_224_dynamic.onnx'
    test_onnx(onnx_file_name,img_pth='/BOBO/RepVGG/data/phone/1.jpg')
