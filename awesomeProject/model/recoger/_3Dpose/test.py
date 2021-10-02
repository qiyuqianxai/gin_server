# coding: utf-8
import numpy as np
import onnxruntime as rt
import cv2
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sess = rt.InferenceSession("Resnet34_3inputs_448x448_20200609.onnx")
fig, ax = plt.subplots()
fig.set_tight_layout(True)
inputs = sess.get_inputs()

cap = cv2.VideoCapture(1)

def update():
    plt.cla()
    ret, img = cap.read()
    count = 0
    while ret:
        # if not count%200 == 0:
        #     count += 1
        #     ret, img = cap.read()
        #     continue
        # img = cv2.imread("5.jpg")
        cv2.imshow("test", img)
        cv2.waitKey(1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (448, 448))
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 1, 0)
        img = img[np.newaxis, ...]

        pred_onx = sess.run(None, {
            inputs[0].name: img,
            inputs[1].name: img,
            inputs[2].name: img
        })

        offset3D = np.squeeze(pred_onx[2])
        heatMap3D = np.squeeze(pred_onx[3])
        '''
        print(offset3D.shape)
        print(heatMap3D.shape)
        print(offset3D.shape[0] / heatMap3D.shape[0])
        '''
        kps = np.zeros((24, 3), np.float32)
        for j in range(0, 24):
            # 找到第j个关节的28个特征图，并找到最大值的索引
            joint_heat = heatMap3D[j * 28:(j + 1) * 28, ...]
            if np.max(joint_heat)>0.1:
                print(np.max(joint_heat))
                [x, y, z] = np.where(joint_heat == np.max(joint_heat))
                x = int(x[-1])
                y = int(y[-1])
                z = int(z[-1])
                # 通过heatmap的索引找到对应的offset图，并计算3D坐标的xyz值
                pos_x = offset3D[j * 28 + x, y, z] + x
                pos_y = offset3D[24 * 28 + j * 28 + x, y, z] + y
                pos_z = offset3D[24 * 28 * 2 + j * 28 + x, y, z] + z

                kps[j, 0] = pos_x
                kps[j, 1] = pos_y
                kps[j, 2] = pos_z
            else:
                try:
                    kps[j, 0] = kps[j-1, 0]
                    kps[j, 0] = kps[j-1, 0]
                    kps[j, 2] = kps[j-1, 2]
                except:
                    pass
            #print("%f,%f,%f;" % (pos_x, pos_y, pos_z))
        print(kps)
        fig, ax = plt.subplots()
        fig.set_tight_layout(True)
        ax = fig.gca(projection='3d')
        ax.set_xlim(0, 30)
        ax.set_ylim(-30, 0)
        ax.set_zlim(-30, 0)
        ax.scatter3D(kps[:, 0], -kps[:, 1], -kps[:, 2], 'red')
        print(len(kps))
        parent = np.array([15, 1, 2, 3, 3, 15, 6, 7, 8, 8, 12, 15, 14, 15, 24, 24, 16, 17, 18, 24, 20, 21, 22, 0]) - 1
        for i in range(len(kps)):
            if (parent[i] != -1):
                ax.plot3D(kps[[i, parent[i]], 0], -kps[[i, parent[i]], 1], -kps[[i, parent[i]], 2], 'gray')

        ax.xaxis.set_tick_params(labelsize=10)
        ax.yaxis.set_tick_params(labelsize=10)
        ax.zaxis.set_tick_params(labelsize=10)
        ax.view_init(elev=10., azim=180)
        ax.figure.savefig("result%d.png"%count)
        count += 1
        plt.show()
        ret, img = cap.read()
    # return ax
if __name__ == '__main__':
    # FuncAnimation 会在每一帧都调用“update” 函数。
    # anim = FuncAnimation(fig, update, interval=200)
    update()
    # plt.show()







