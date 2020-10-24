# -*- coding: UTF-8 -*-
# tools of MOEA/D algorithm.
# 2020-10-24 FrankJiaShu

import numpy as np
import math
import random
import matplotlib.pyplot as plt
from evolution.MOEAD.moead import Individual


# 切比雪夫方法操作-找最大的值返回
# input x->一维list, lambda->二维list, z->一维list
# return gte->一维list
def Tchebycheff(x, z, lambd):
    Gte = []
    for i in range(len(x.f)):
        Gte.append(np.abs(x.f[i]-z[i]) * lambd[i])
    return np.max(Gte)  # 取距离*权重最远的点


# GA遗传算法操作
# input a, b->parents, rate->父母变异概率默认0.5
# return y1, y2->mutation & crossover individuals
def GeneticOperation(a, b, rate=0.5):
    offspring = crossover(a, b)
    r = random.random()
    if r > rate:
        return mutation(a), offspring  # 父亲变异+后代
    else:
        return mutation(b), offspring  # 母亲变异+后代


# GA-整型转二进制字符串
def integerToString(m, n):
    cadenas = [bin(m), bin(n)]
    cadenas[0] = cadenas[0].replace("0b", "")
    cadenas[1] = cadenas[1].replace("0b", "")
    if len(cadenas[0]) > len(cadenas[1]):
        cadenas[1] = ("0" * (len(cadenas[0]) - len(cadenas[1]))) + cadenas[1]
    else:
        cadenas[0] = ("0" * (len(cadenas[1]) - len(cadenas[0]))) + cadenas[0]
    return cadenas[0], cadenas[1]


# GA-二进制字符串转整型
def stringToInteger(string):
    decimal = 0
    for i, v in enumerate(string):
        if v == '1':
            decimal += math.pow(2, len(string) - 1 - i)
    return decimal


# GA-交叉操作
# input a, b->parents
# return offspring->Individual
def crossover(a, b):
    offspring = []
    for i in range(len(a.x)):
        str1, str2 = integerToString(int(a.x[i]), int(b.x[i]))
        R = random.randint(0, len(str1) - 1)  # 随机取交叉位
        new_x = stringToInteger(str1[:R] + str2[R:])
        if new_x > 10000:
            new_x = 10000
        offspring.append(new_x)
    return Individual(offspring)


# GA-变异操作
# input c->individual(变异前)
# return c(变异后)
def mutation(c):
    r = random.randint(0, len(c.x) - 1)  # 随机取变异维度
    str1, str2 = integerToString(int(c.x[r]), int(c.x[r]))
    R = random.randint(0, len(str1) - 1)  # 随机取变异位
    if str1[R] == '0':
        str1 = str1[:R]+'1'+str1[R+1:]
    else:
        str1 = str1[:R]+'0'+str1[R+1:]

    new_x = stringToInteger(str1)
    if new_x > 10000:
        c.x[r] = 10000
    else:
        c.x[r] = new_x
    return Individual(c.x)


# 画图构造EP非支配集
def plot(EP):
    x = []
    y = []
    for i in range(len(EP)):
        x.append(EP[i].f[0])
        y.append(EP[i].f[1])
    plt.plot(x, y, '*')
    plt.xlabel('f1 direction')
    plt.ylabel('f2 direction')
    plt.show()
