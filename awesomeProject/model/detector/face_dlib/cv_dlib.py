import cv2
import dlib
import time
import numpy as np
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
    def __init__(self):
        self. dector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("detector/face_dlib/shape_predictor_68_face_landmarks.dat")
        expression_membership = np.array([
            [0.7, 0.3, 0.1, 0],  # nature
            [0.1, 0.1, 0.3, 0.5],  # happy
            [0.8, 0.1, 0.1, 0],  # angry
            [0.1, 0.1, 0.4, 0.4]  # amazing
        ])

        # 输入模糊矩阵构建 模糊总和分析 对象
        self.be2 = BE2(expression_membership)
    def get_face(self,img):
        """
           获取面部数据
           返回值两个
           一个用于返回表情 一个用于返回68点的坐标
           最后一个来确定是否存在人脸图像
        """
        self.s = {}
        res = self.dector(img)
        print(len(res))
        # 眉毛直线拟合数据缓冲
        line_brow_x = []
        line_brow_y = []
        # 68点坐标
        dp = [[0 for k in range(2)]for j in range(68)]
        face_coordinate = []
        if (len(res) != 0):
            for i in range(0,len(res)):
                for id,lm in enumerate(res):
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
                            self.s = dict(emotion = 'amazing')
                        else:
                            self.s = dict(emotion = 'happy')

                    # 没有张嘴，可能是正常和生气
                    else:
                        if self.brow_k <= -0.5:
                            self.s = dict(emotion = 'angry')
                        else:
                            self.s = dict(emotion = 'nature')
                    for i in range(68):
                        # self.s = ""
                        # self.s += "id: "+ str(i) +","
                        # self.s += "x: " + str(shape.part(i).x) + ","
                        # self.s += "y: " + str(shape.part(i).y)
                        dp[i][0] = shape.part(i).x
                        dp[i][1] = shape.part(i).y

            marks = np.array(dp)
            return self.s,marks,len(res)
        else:
            return self.s,np.array(dp),len(res)

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


def main():
    # 初始化函数
    CTime = 0
    PTime = 0
    cap = cv2.VideoCapture(0)
    op = Face()
    # cap.set(3, 480)
    # 开启初始的摄像头图像为内置
    while True:
        success, img = cap.read()
        # 获取脸部数据
        emo,vis, len = op.get_face(img)
        # emo字典 返回当前表情   vis返回二维数组 68点的坐标 len代表是否存在人脸
        print(emo)
        print(vis)
        #  每个角度的专注力分数
        for i in range(68):
           score = op.get_attention_score(i,emo,len)
           print(score)

        CTime = time.time()
        fps = 1 / (CTime - PTime)
        PTime = CTime
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        # 范围 字体 比例 颜色 速度
        cv2.imshow("Image", img)
        cv2.waitKey(1)



if __name__ == "__main__":
    op = Face()
    img = cv2.imread("/BOBO/cv_proj/model/hezaho.jpg")
    emo,vis, _ = op.get_face(img)
    print(len(vis))
