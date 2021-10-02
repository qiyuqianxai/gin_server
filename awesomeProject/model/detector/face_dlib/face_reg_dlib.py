import cv2
import dlib
import time
import numpy as np
import math
# 获取人脸关键点  获取基本的人脸信息

class BE2:
    def __init__(self, R1):
        """
        创建对象时，需要输入两个隶属度矩阵
        :param R1: 表情的隶属度矩阵
        """
        self.R1 = R1

    @staticmethod
    def normalized(data):
        """
        保证data中的数据相加为1
        :param data: 输入的一维数组
        :return:
        """
        sum = 0
        for i in range(len(data)):
            sum += data[i]
        if sum == 1:
            return np.array(data)
        return np.array(data) / sum

    @staticmethod
    def min_max_operator(W, R):
        '''
        主因素突出型：M(Λ, V)
        利用最值算子合成矩阵
        :param W:评判因素权向量
        :param R:模糊关系矩阵
        :return:
        '''
        B = np.zeros((1, np.shape(R)[1]))
        for column in range(0, np.shape(R)[1]):
            _list = []
            for row in range(0, np.shape(R)[0]):
                _list.append(min(W[row], R[row, column]))
            B[0, column] = max(_list)
        return B

    @staticmethod
    def min_add_operator(W, R):
        '''
        主因素突出型：M(Λ, +)
        先取小，再求和
        :param W:评判因素权向量
        :param R:模糊关系矩阵
        :return:
        '''
        B = np.zeros((1, np.shape(R)[1]))
        for column in range(0, np.shape(R)[1]):
            _list = []
            for row in range(0, np.shape(R)[0]):
                _list.append(min(W[row], R[row, column]))
            B[0, column] = np.sum(_list)
        return B

    @staticmethod
    def mul_max_operator(W, R):
        '''
        加权平均型：M(*, +)
        利用乘法最大值算子合成矩阵
        :param W:评判因素权向量
        :param R:模糊关系矩阵
        :return:
        '''
        B = np.zeros((1, np.shape(R)[1]))
        for column in range(0, np.shape(R)[1]):
            list = []
            for row in range(0, np.shape(R)[0]):
                list.append(W[row] * R[row, column])
            B[0, column] = max(list)
        return B

    @staticmethod
    def mul_add_operator(W, R):
        '''
        加权平均型：M(*, +)
        先乘再求和
        :param W:评判因素权向量 A = (a1,a2 ,L,an )
        :param R:模糊关系矩阵
        :return:
        '''
        return np.matmul(W, R)

    def get(self, W, R):
        """
        :return: 获得var最大的
        """
        s = [self.normalized(self.min_max_operator(W, R).reshape(np.shape(R)[1])),
             self.normalized(self.min_add_operator(W, R).reshape(np.shape(R)[1])),
             self.normalized(self.mul_max_operator(W, R).reshape(np.shape(R)[1])),
             self.normalized(self.mul_add_operator(W, R).reshape(np.shape(R)[1]))]
        vars = []

        for i in range(len(s)):
            vars.append(s[i].var())

        i = np.argmax(vars)
        return s[i]

    def run(self, W1):
        """

        :param W1: 表情的概率
        :return:
        """
        R = self.get(W1, self.R1),

        return np.dot(R, [1, 0.9, 0.8, 0.6])

class Face():
    def __init__(self,predictor_pth="shape_predictor_68_face_landmarks.dat"):
        # self.dector = dlib.get_frontal_face_detector()

        self.predictor = dlib.shape_predictor(predictor_pth)
        self.POINTS_NUM_LANDMARK = 68
        expression_membership = np.array([
            [0.7, 0.3, 0.1, 0],  # nature
            [0.1, 0.1, 0.3, 0.5],  # happy
            [0.8, 0.1, 0.1, 0],  # angry
            [0.1, 0.1, 0.4, 0.4]  # amazing
        ])

        # 输入模糊矩阵构建 模糊总和分析 对象
        self.be2 = BE2(expression_membership)


    def get_attention_score(self, dot, emo, len):
        """
          获取专注力分数
          参数1  dot 脸部朝向角度
          参数2  emo脸部此时对应的表情
          参数3  是否存在人脸

        """
        # 获取对应表情的标签
        if  len != 0 :
            index = 0
            if emo['emotion'] == 'nature':
                index = 0
            elif emo['emotion'] == 'happy':
                index = 1
            elif emo['emotion'] == 'amazing':
                index = 3
            elif emo['emotion'] == 'angry':
                index = 2

            angle = abs(dot)
            angle = min([angle, 90])
            if angle <= 20:
                angle_score = 1 - angle / 100
            elif angle <= 40:
                angle_score = 1 - 0.2 - (angle - 20) / 80
            elif angle <= 60:
                angle_score = 1 - 0.2 - 0.25 - (angle - 40) / 60
            else:
                angle_score = 1 - 0.2 - 0.25 - 0.3 - (angle - 60) / 40

            expression_weight = np.zeros([4])
            expression_weight[index] = 1
            # 表情和转头角度对专注度判断的重要性
            weight = [0.5, 0.5]

            expression_score = self.be2.run(expression_weight)
            return np.dot(weight, [expression_score, angle_score]), index
        else :
            return np.dot([0, 0],[0 , 0]),0

        # 从dlib的检测结果抽取姿态估计需要的点坐标

    def get_image_points_from_landmark_shape(self, landmark_shape):
        if landmark_shape.num_parts != self.POINTS_NUM_LANDMARK:
            return False, None

        # 2D image points. If you change the image, you need to change vector
        image_points = np.array([
            (landmark_shape.part(30).x, landmark_shape.part(30).y),  # Nose tip
            (landmark_shape.part(8).x, landmark_shape.part(8).y),  # Chin
            (landmark_shape.part(36).x, landmark_shape.part(36).y),  # Left eye left corner
            (landmark_shape.part(45).x, landmark_shape.part(45).y),  # Right eye right corne
            (landmark_shape.part(48).x, landmark_shape.part(48).y),  # Left Mouth corner
            (landmark_shape.part(54).x, landmark_shape.part(54).y)  # Right mouth corner
        ], dtype="double")

        return True, image_points

    # 用dlib检测关键点，返回姿态估计需要的几个点坐标

        # 获取旋转向量和平移向量

    def get_pose_estimation(self, img_size, image_points):
        model_points = np.array([
            (0.0, 0.0, 0.0),  # Nose tip
            (0.0, -330.0, -65.0),  # Chin
            (-225.0, 170.0, -135.0),  # Left eye left corner
            (225.0, 170.0, -135.0),  # Right eye right corne
            (-150.0, -150.0, -125.0),  # Left Mouth corner
            (150.0, -150.0, -125.0)  # Right mouth corner

        ])

        focal_length = img_size[1]
        center = (img_size[1] / 2, img_size[0] / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )

        dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                      dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

        return success, rotation_vector

        # 从旋转向量转换为欧拉角

    def get_euler_angle(self, rotation_vector):
        theta = cv2.norm(rotation_vector, cv2.NORM_L2)
        w = math.cos(theta / 2)
        x = math.sin(theta / 2) * rotation_vector[0][0] / theta
        y = math.sin(theta / 2) * rotation_vector[1][0] / theta
        z = math.sin(theta / 2) * rotation_vector[2][0] / theta
        ysqr = y * y
        t3 = 2.0 * (w * z + x * y)
        t4 = 1.0 - 2.0 * (ysqr + z * z)
        roll = math.atan2(t3, t4)
        # 单位转换：将弧度转换为度
        Z = int((roll / math.pi) * 180)

        return Z
    def get_face(self,o_img, res):
        """
           获取面部数据
           返回人脸数据列表
           列表针对每个人脸组成一个字典
           字典格式
           key_points 68个关键点, face_coordinate 脸部框型坐标, expression 人脸表情, attention_score 专注力分数及其评判
        """

        img = cv2.cvtColor(o_img, cv2.COLOR_RGB2GRAY)

        self.s = {}
        # self.res = self.dector(img)
        self.res = res
        # 眉毛直线拟合数据缓冲
        line_brow_x = []
        line_brow_y = []
        # 68点坐标
        ret = []
        # self.res是对应人脸的数目

        dp = [[0 for k in range(2)] for j in range(68)]
        for id,lm in enumerate(self.res):
            lm = dlib.rectangle(lm[0],lm[1],lm[2],lm[3])
            # ans记录人脸矩形
            ans = []
            ans.append(lm.left())
            ans.append(lm.top())
            ans.append(lm.right())
            ans.append(lm.bottom())

            self.face_width = lm.right() - lm.left()
            self.face_higth = lm.top() - lm.bottom()
            # 计算人脸长度


            # 使用预测器得到68点数据的坐标
            shape = self.predictor(img, lm)

            # 分析点的位置关系来作为表情识别的依据
            mouth_width = (shape.part(54).x - shape.part(48).x) / self.face_width  # 嘴巴咧开程度
            mouth_higth = (shape.part(66).y - shape.part(62).y) / self.face_width  # 嘴巴张开程度

            # 通过两个眉毛上的10个特征点，分析挑眉程度和皱眉程度
            brow_sum = 0  # 高度之和
            frown_sum = 0  # 两边眉毛距离之和
            for j in range(17, 21):
                brow_sum += (shape.part(j).y - lm.top()) + (shape.part(j + 5).y - lm.top())
                frown_sum += shape.part(j + 5).x - shape.part(j).x
                line_brow_x.append(shape.part(j).x)
                line_brow_y.append(shape.part(j).y)

            tempx = np.array(line_brow_x)
            tempy = np.array(line_brow_y)

            # np.ployfit(x,a,n)拟合点集a得到n级多项式，其中x为横轴长度
            z1 = np.polyfit(tempx, tempy, 1)  # 拟合成一次直线
            self.brow_k = -round(z1[0], 3)  # 拟合出曲线的斜率和实际眉毛的倾斜方向是相反的

            eye_sum = (shape.part(41).y - shape.part(37).y + shape.part(40).y - shape.part(38).y +
                       shape.part(47).y - shape.part(43).y + shape.part(46).y - shape.part(44).y)
            eye_hight = (eye_sum / 4) / self.face_width


            if round(mouth_higth >= 0.03):
                if eye_hight >= 0.056:
                    self.s = dict(id = 0, emotion = 'amazing')
                else:
                    self.s = dict(id = 0, emotion = 'happy')

            # 没有张嘴，可能是正常和生气
            else:
                if self.brow_k <= -0.5:
                    self.s = dict(id = 0, emotion = 'angry')
                else:
                    self.s = dict(id = 0,emotion = 'nature')
            # 记录68点的坐标
            for k in range(68):
                dp[k][0] = shape.part(k).x
                dp[k][1] = shape.part(k).y
            marks = np.array(dp)
            '''
                :type
                输入一张图像，输出人脸欧拉角 roll
            '''
            size = img.shape
            self.re = None

            ok_face_kp, image_points = self.get_image_points_from_landmark_shape(shape)
            if not ok_face_kp:
                self.re = None
            ok_3d_face, rotation_vector= self.get_pose_estimation(
                size,
                image_points)
            if not ok_3d_face:
                self.re = None
            else :
                self.re = self.get_euler_angle(rotation_vector)
            score = 0
            if self.re != None:
                score = self.get_attention_score((abs(self.re * 2)), self.s, len)
            cnt = dict(key_points=marks.tolist(), face_coordinate=ans, expression=self.s['emotion'], attention_score=score[0])
            ret.append(cnt)
        return ret

def main():
    # 初始化函数
    CTime = 0
    PTime = 0
    cap = cv2.VideoCapture(0)
    # path_image = "a.jpg"
    op = Face()
    # cap.set(3, 480)
    # 开启初始的摄像头图像为内置
    while True:
        success, img = cap.read()
        # img = cv2.imread(path_image)
        # 获取脸部数据
        ret = op.get_face(img)
        print(ret)
        # with open("test.txt", "w") as f:
        #     f.write(str(ret))  # 这句话自带文件关闭功能，不需要再写f.close()
        if cv2.waitKey(10) == 27:
            break
        CTime = time.time()
        fps = 1 / (CTime - PTime)
        PTime = CTime
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        # 范围 字体 比例 颜色 速度
        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    # main()
    op = Face()
    img = cv2.imread("/BOBO/cv_proj/model/hezaho.jpg")
    t1= time.time()
    ret = op.get_face(img)
    t2=time.time()
    print(t2-t1)
    print(len(ret))
