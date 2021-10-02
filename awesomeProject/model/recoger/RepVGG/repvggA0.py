import cv2
import numpy as np


def repvggA0_onnx_inference(session,img,class_names):

    # session = onnxruntime.InferenceSession(onnx_path)
    # session = onnx.load(onnx_path)
    # print("The model expects input shape: ", session.get_inputs()[0].shape)
    IN_IMAGE_H = session.get_inputs()[0].shape[2]
    IN_IMAGE_W = session.get_inputs()[0].shape[3]

    # Input
    resized = cv2.resize(img, (IN_IMAGE_W, IN_IMAGE_H), interpolation=cv2.INTER_LINEAR)
    img_in = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    img_in = np.transpose(img_in, (2, 0, 1)).astype(np.float32)
    img_in = np.expand_dims(img_in, axis=0)
    # img_in = np.concatenate((img_in, img_in), axis=0)
    img_in /= 255.0
    # 使用模型进行预测
    input_name = session.get_inputs()[0].name

    outputs = session.run(None, {input_name: img_in})
    class_logits = outputs[0]
    result = np.argmax(class_logits,axis=-1).item()
    # print(result)
    result = class_names[result]
    # print("action:",result)

    return result
