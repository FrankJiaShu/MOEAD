# -*- coding: UTF-8 -*-
# main MOEA/D algorithm.
# 2020-10-24 FrankJiaShu

import numpy as np
import random
import evolution.MOEAD.tools as tools


# 参数定义
rate = 0.5  # 变异概率
iteration = 20  # 一次迭代次数
dimensions = 2  # 目标函数维度
decisionFun = 'Tchebycheff'  # 判别方法


# 定义个体类
# x个维度->一维list f->目标函数一维list
class Individual:
    def __init__(self, x: list):
        self.x = x
        f1 = float(x[0] / 10000)
        h = float(1 + x[1] / 1000)
        f2 = h * (1 - (f1 / h)**2 - (f1 / h) * np.sin(8 * np.pi * f1))
        # f列表表示多目标函数由f1和f2组成
        self.f = [f1, f2]


# 初始化
# input 数量为N->int
# return 种群P->一维list 权重向量lambda->二维list [第一维代表种群中的个体list 第二维代表个体的权重list]
def Initial(N):
    P = []
    lambd = []
    for i in range(N):
        P.append(Individual([random.random()*10000, random.random()*10000]))
        lambd.append([float(i) / N, 1.0-float(i) / N])
    return P, lambd


# 根据权重向量λ计算T个邻居存入B
# return B->二维list [第一维代表种群中的个体list 第二维代表个体的T个邻居list]
def Neighbor(lambd, T):
    B = []
    for i in range(len(lambd)):
        temp = []
        for j in range(len(lambd)):
            # 计算二维欧氏距离
            distance = np.sqrt((lambd[i][0] - lambd[j][0])**2 + (lambd[i][1] - lambd[j][1])**2)
            temp.append(distance)
        res = np.argsort(temp)  # 下标排序
        B.append(res[:T])  # 取前T个近的邻居加入B
    return B


# 获取种群的最小f值作为参考点-min
# input 种群P->一维list
# return z->一维list
def BestValue(P):
    z = [P[0].f[i] for i in range(len(P[0].f))]  # 初始化
    for i in range(1, len(P)):
        for j in range(len(P[i].f)):
            if P[i].f[j] < z[j]:
                z[j] = P[i].f[j]
    return z


# 计算x是否支配y，min表示最小化目标函数
# input x, y->individual
# return 是否支配->bool
def Dominate(x, y, minimum=True):
    if minimum:
        for i in range(len(x.f)):
            if x.f[i] > y.f[i]:
                return False
        return True
    else:
        for i in range(len(x.f)):
            if x.f[i] < y.f[i]:
                return False
        return True


# 主函数运行
def MOEAD(N, T):
    # step1 --------------------初始化处理----------------------------
    p, lambd = Initial(N)
    B = Neighbor(lambd, T)
    z = BestValue(p)
    EP = []
    # step2 -------------------迭代更新处理---------------------------
    index = 0
    while index < iteration:
        index += 1
        print('Round', index, '| non-dominated solutions: ', len(EP))
        for i in range(N):
            # 随机从T个中选两个邻居GA操作 j+k->y1+y2
            j = random.randint(0, T - 1)
            k = random.randint(0, T - 1)
            y1, y2 = tools.GeneticOperation(p[B[i][j]], p[B[i][k]], rate)
            if Dominate(y1, y2):
                y = y1
            else:
                y = y2
            # 更新z-best参考点-min
            for j in range(len(z)):
                if y.f[j] < z[j]:
                    z[j] = y.f[j]
            # 更新种群 参数选择-切比雪夫判别法找min
            for j in range(len(B[i])):
                Ta = tools.Tchebycheff(p[B[i][j]], z, lambd[B[i][j]])
                Tb = tools.Tchebycheff(y, z, lambd[B[i][j]])
                if Tb < Ta:
                    p[B[i][j]] = y
            # 更新存储的非支配集
            if EP is None:
                EP.append(y)
            else:
                dominateY = False  # 是否有支配Y的解
                remove = []  # 被Y支配的解
                for j in range(len(EP)):
                    if Dominate(y, EP[j]):
                        remove.append(EP[j])
                    elif Dominate(EP[j], y):
                        dominateY = True
                        break
                # 没有支配Y的 将Y加入EP 移除被Y支配的解
                if not dominateY:
                    EP.append(y)
                    for j in range(len(remove)):
                        EP.remove(remove[j])
    # step3 -------------------可视化处理-----------------------------
    tools.plot(EP)


# main函数运行
if __name__ == '__main__':
    MOEAD(1500, 10)
