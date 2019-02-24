import numpy as np 
import scipy as scipy
from scipy.optimize import leastsq
import matplotlib.pyplot as plt

#目标函数
def real_func(x):
    return np.sin(2*np.pi*x)

#多项式
def fit_func(p,x):
    f = np.poly1d(p) #np.poly1d([1,2,3])生成1x^2+2x+3
    return f(x) #f(x)是求x = num时的输出值

#残差
def residuals_func(p,x,y):
    ret = fit_func(p,x) - y
    return ret

#十个点
x = np.linspace(0,1,10) #linspace产生等差数列
x_points = np.linspace(0,1,100)
#加上正态分布噪音的目标函数的值
y_ = real_func(x)
y = [np.random.normal(0,0.1) + y1 for y1 in y_]
# normal正态分布

def fitting(M=0):
    # n为多项式的系数
    # 随机初始化多项式参数
    #返回一组服从“0~1”均匀分布的随机样本值，随机样本值取值范围
    #是[0,1),不包括1
    p_init = np.random.rand(M+1)
    #最小二乘法,第一个放入残差函数，第二个放入拟合方程的初始系数
    #将其余参数打包到args中
    p_lsq = leastsq(residuals_func,p_init,args=(x,y))
    print("Fitting Parameters", p_lsq[0])

    #可视化
    plt.plot(x_points, real_func(x_points), label="real")
    plt.plot(x_points, fit_func(p_lsq[0], x_points), label='fitted curve')
    plt.plot(x,y,'bo',label = 'noise')
    #显示图例，显示标签的摆放位置
    plt.legend()
    plt.show()
    return p_lsq
#M=0
p_lsq_0 = fitting(M=0)

#M=1
p_lsq_1 = fitting(M=1)

#M=3
p_lsq_3 = fitting(M=3)

#M=9
p_lsq_9 = fitting(M=9)
##############################################################
#正则化
#结果显示过拟合，引入正则项，降低过拟合，回归问题中，正则化可以是：
#L1：regularization *abs(p)
#L2: 0.5 * regularization * np.square(p)

regularization = 0.0001

def residuals_func_regularization(p,x,y):
    ret = fit_func(p,x) - y
    ret = np.append(ret, np.sqrt(0.5 * regularization* np.square(p)))
    return ret 
#最小二乘法，加正则项
p_init = np.random.rand(9+1)
p_lsq_regularization = leastsq(
    residuals_func_regularization,p_init,args=(x,y))

plt.plot(x_points, real_func(x_points), label='real')
plt.plot(x_points, fit_func(p_lsq_9[0], x_points), label='fitted curve')
plt.plot(x_points, fit_func(p_lsq_regularization[0], x_points), label='regularization')
plt.plot(x, y, 'bo', label='noise')
plt.legend()
plt.show()
