import glob
import itertools
import math
import os.path
import pickle
import time
from bisect import bisect_left
import inspect
from lib2to3.btm_matcher import type_repr

import numpy
import sympy
from sympy import *
import numpy as np
import pandas as pd
from obspy.taup import TauPyModel
from obspy.taup import taup_create
from multiprocessing.pool import Pool
from scipy.sparse.linalg import lsqr
import matplotlib.pyplot as plt
from numpy.linalg import norm
from dt_pkp import PKP
import random
from grid import Grid, spherical_to_cartesian
from math import radians, sin, cos, sqrt
from scipy.spatial import KDTree
import pygmt


def f(nn):
    return float(nn)


def find_zone(di, nu):
    if type(di) == dict:
        di = list(di.keys())
    left = bisect_left(di, nu)
    return (left, left + 1) if nu - di[left] >= 0 else (left - 1, left)


def find_nearest_neighbors(grid_points, query_point, k=10):
    """
    使用KDTree查找最近的k个邻居

    参数:
    grid_points: 网格点数组，形状为(n, 3)
    query_point: 查询点，形状为(3,)的数组
    k: 要查找的最近邻数量

    返回:
    distances: 最近邻的距离数组
    indices: 最近邻在grid_points中的索引数组
    """
    # 构建KDTree
    tree = KDTree(grid_points)

    # 查询最近的k个点
    distances, indices = tree.query(query_point, k=k)

    return distances, indices


def find_nearest_index(array, value):
    """
    在NumPy数组中找到最接近给定值的元素的索引

    参数:
    array: NumPy数组
    value: 要查找的目标值

    返回:
    最接近值的索引
    """
    # 计算数组中每个元素与目标值的绝对差
    absolute_diff = np.abs(array - value)

    # 找到最小差值的索引
    index = np.argmin(absolute_diff)

    return index


def calculate_c_velocity_spherical(A, B, C, v1, v2):
    """
    在球面坐标中通过三点位置推算C点速度

    参数:
    A, B, C: 三点的球面坐标，格式为(经度,纬度,深度)
    v1, v2: A点和B点的速度

    返回:
    v3: C点的估算速度
    """
    # 将球面坐标转换为笛卡尔坐标
    # A_cart = spherical_to_cartesian(A[0], A[1], A[2])
    # B_cart = spherical_to_cartesian(B[0], B[1], B[2])
    # C_cart = spherical_to_cartesian(C[0], C[1], C[2])
    A_cart = (A[0], A[1], A[2])
    B_cart = (B[0], B[1], B[2])
    C_cart = (C[0], C[1], C[2])
    # print("A_cart = ", A_cart)
    # print("B_cart = ", B_cart)
    # print("C_cart = ", C_cart)

    # 转换为numpy数组
    A_cart = np.array(A_cart)
    B_cart = np.array(B_cart)
    C_cart = np.array(C_cart)

    # 计算AB和AC向量
    AB = B_cart - A_cart
    AC = C_cart - A_cart

    # 计算AB的长度
    AB_length = np.linalg.norm(AB)
    # print("AB_length = ", AB_length)


    # 如果AB长度为零，则A和B是同一点，无法插值
    if AB_length == 0:
        raise ValueError(A,B,C, v1, v2,"点A和点B不能重合")

    # 计算AC在AB方向上的单位向量投影
    AB_unit = AB / AB_length
    AC_projection = np.dot(AC, AB_unit)

    # 根据线性插值计算C点速度
    v3 = v1 + (v2 - v1) * (AC_projection / AB_length)

    return v3


class Iv:
    def __init__(self, fi, lpm, dam, phase='PKIKP', height=6, coa=0.95, absolute=True, sys_inf=None, step=None,
                 up_depth=300, down_depth=200, la_int=10, lo_int=10, grid_inter=10):
        self.grid_6_latlon = None
        self.grid_6_cartesian = None
        self.lo_int = lo_int
        self.la_int = la_int
        self.down_depth = down_depth
        self.up_depth = up_depth
        self.sys_v = None
        self.sys_inf = sys_inf
        self.absolute = absolute
        self.coa = coa
        self.he = height
        self.misfit = 1
        self.cycle = None
        self.prior_model = None
        self.main_phase = phase
        self.assign = None
        self.file = fi
        self.lpm = lpm
        self.dam = dam
        # self.depth = depth
        self.earth_ra = 6371
        self.CMB_dep = 2891
        self.ICB_ak135 = 5153.50
        self.ICB_PREM = 5149.50
        self.final_fit = 0.1
        self.max_dt = 4
        self.rang = None
        # self.step = step
        # self.global_net = int((360 / self.lo_int)*(180 / self.la_int + 1))
        self.outer_core_layer = (2, 200)
        self.inner_core_layer = (3, 200)
        # self.prior_model = prior_model

        grid = Grid()
        self.grid_6_cartesian, self.grid_6_latlon, self.grid_ne = grid.boby(grid_inter,
                                                                            outer_core_layer=self.outer_core_layer,
                                                                            inner_core_layer=self.inner_core_layer,
                                                                            plot=False)
        self.rang = np.arange(0, len(self.grid_6_cartesian), 1)
        one_layer = int(len(self.rang) / (self.outer_core_layer[0] + self.inner_core_layer[0]))
        self.grid_outer_core = self.grid_6_cartesian[0: one_layer * self.outer_core_layer[0]: 1]
        self.grid_inner_core = self.grid_6_cartesian[one_layer * self.outer_core_layer[0]: len(self.rang): 1]
        self.v = self.fir()  # 初始速度模型
        self.ass(self.v)
        # alpha = self.lps(self.lpm, self.grid_ne)
        # for i,index in enumerate(alpha[0]):
        #     print(i, index)

        # self.model(self.v)

    def fir(self):
        outer_core_layer = self.outer_core_layer
        inner_core_layer = self.inner_core_layer
        velo = [0 for x in self.rang]
        # print(self.rang)
        # for depths in self.rang:
        # layer = (360/self.lo_int)*(180/self.la_int+1)
        # intial_depth = self.ICB_PREM-self.up_depth
        # outer core
        one_layer = int(len(self.rang) / (outer_core_layer[0] + inner_core_layer[0]))
        for i in range(outer_core_layer[0]):
            k = (self.earth_ra + outer_core_layer[1]*(outer_core_layer[0]-1-i) -self.ICB_PREM) / self.earth_ra
            a, b, c, d = 11.0487, -4.0362, 4.8023, -13.5732
            for ii in range(one_layer):
                velo[ii + i * one_layer] = a + b * k + c * (k ** 2) + d * (k ** 3)
        # print(i)
        # print(ii+int((i-1)*self.global_net), len(self.rang))
        # inner core
        for i in range(inner_core_layer[0]):
            k = (self.earth_ra - self.ICB_PREM - i * inner_core_layer[1]) / self.earth_ra
            a, b, c, d = 11.2622, 0.0, -6.364, 0.0
            for ii in range(one_layer):
                # print(ii + int((self.up_depth/self.step + 1 + i - 1)) * self.global_net)
                velo[ii + (i + outer_core_layer[0]) * one_layer] = a + b * k + c * (k ** 2) + d * (k ** 3)
        # print(self.rang[i]+2891, velo[i])
        return Matrix(velo)

    def vx(self):
        vxx = [0 for x in self.rang]
        # for depths in self.rang:
        for i in range(0, len(self.rang), 1):
            xx = 'v' + str(i)
            vxx[i] = sympy.symbols(xx)
        return pd.Series(vxx)

    def ve_mar(self, name, m, real=False):
        v__, h = m
        if name == 'own':
            v_m = [0 for x in self.rang]
            for i in range(0, len(self.rang), 1):
                depths = self.rang[i]
                if real:
                    k = (self.earth_ra - depths - self.CMB_dep) / self.earth_ra
                    a, b, c, d = 11.0487, -4.0362, 4.8023, -13.5732
                    v = a + b * k + c * (k ** 2) + d * (k ** 3)
                    v_m[i] = v + v__ / h * depths - v__ if depths <= h else v
                else:
                    v_m[i] = v__ / h * depths - v__ if depths <= h else 0.0
            return np.array(v_m, dtype=np.float64)
        return None

    def ass(self, vk):
        V = self.vx()
        ass1 = {}
        # for depths in self.rang:
        for i in range(0, len(self.rang), 1):
            ass1[V[i]] = vk[i]
        self.assign = ass1
        return ass1

    def model(self, v_m, out='model'):  # 一维速度模型和距离间隔
        if os.path.exists('npz/model.npz'):
            os.unlink('npz/model.npz')
        nd = out + '.nd'
        depth_m = self.CMB_dep + self.depth
        with open('npz/prem.nd', 'r') as f2:
            model_prem_nd = np.loadtxt(f2)
            st = np.where(model_prem_nd[:, 0] == self.CMB_dep)[0][0]
            en = np.where(model_prem_nd[:, 0] > depth_m)[0][1]
            model_up = model_prem_nd[0:st + 1:1]
            model_down = model_prem_nd[en:len(model_prem_nd):1]
            for i in range(0, len(self.rang), 1):
                depth_line = self.CMB_dep + self.rang[i]
                dl = (6371 - depth_line) / 6371
                ρ = 12.5815 - 1.2638 * dl - 3.6426 * (dl ** 2) - 5.5281 * (dl ** 3)
                model_line = np.array([depth_line, v_m[i], 0.00000, ρ, 57822.0, 0.0])
                model_up = np.append(model_up, [model_line], axis=0)
            model_whole = np.append(model_up, model_down, axis=0)
            if os.path.exists(nd):
                os.unlink(nd)
        with open(nd, 'w') as file_np:
            for line in range(0, len(model_whole)):
                if line == 4:
                    file_np.writelines('mantle \n')
                elif line == st + 1:
                    file_np.writelines('outer-core \n')
                elif line == np.where(model_whole[:, 0] == 5149.50)[0][1]:
                    file_np.writelines('inner-core \n')
                np.savetxt(file_np, model_whole[line], fmt='%f', newline='  ', header='', footer='\n',
                           comments='')
        taup_create.build_taup_model(nd, output_folder=os.getcwd() + '/npz')
        if os.path.exists(nd):
            os.unlink(nd)
        return

    def velocity(self, lon, lat, depth, grid, k=10):
        # print(grid)
        grid,vx_outer = (self.grid_outer_core, 0) if grid == 'outer' else (self.grid_inner_core, len(self.grid_outer_core))
        V = self.vx()
        C = np.array(spherical_to_cartesian(lon, lat, depth))  # 点A的坐标(a,b,c)
        # . 使用KDTree查找最近的10个点
        k = 10
        distances, indices = find_nearest_neighbors(grid, C, k=k)
        tree_point = []
        for i in indices:
            tree_point.append(i)
        all_pairs = list(itertools.combinations(tree_point, 2))
        vc = 0

        for i, pair in enumerate(all_pairs):
            vc += calculate_c_velocity_spherical(grid[pair[0]], grid[pair[1]], C,
                                                 V[pair[0]+vx_outer], V[pair[1]+vx_outer])
        #     print(C,grid[pair[0]], grid[pair[1]])
        # print(vc/len(all_pairs))
        return vc/len(all_pairs)

    def check(self, key):
        if type(key) == numpy.float64 and key >= 1.0:
            return1 = numpy.float64(1.0)
        elif (type(key) == sympy.core.add.Add or type(key) == sympy.core.mul.Mul) and key.subs(self.assign) >= 1.0:
            return1 = numpy.float64(1.0)
        else:
            return1 = key
        return return1

    def formula(self, path_sub, grid_out_inn='outer', ray_direction='up'):
        dt_seg = 0
        # grid_use = grid_outer_core if grid_out_inn == 'outer-core' else grid_inner_core
        for ix in range(len(path_sub) - 1):
            pa = path_sub[ix]
            pb = path_sub[ix + 1]
            # if pa[3] - self.CMB_dep >= self.depth:
            #     dt_seg += pb[1] - pa[1]
            #     continue
            # ra = self.earth_ra - pa[3]
            # rb = self.earth_ra - pb[3]
            # T = math.pi / 180
            # cosθ = math.cos(pb[4] * T) * math.cos(pa[4] * T) * math.cos(pb[5] * T - pa[5] * T) + math.sin(
            #     pa[4] * T) * math.sin(pb[4] * T)
            # sinθ = math.sqrt(1 - cosθ * cosθ)
            # sina = self.check(ray_par * Va / ra)
            # sinb = self.check(ray_par * Vb / rb)
            # cosa = sympy.sqrt(1 - sina * sina)
            # cosb = sympy.sqrt(1 - sinb * sinb)
            # k2 = ra * sina / (sina * cosθ + sinθ * cosa)
            # k1 = rb * sinb / (sinb * cosθ - sinθ * cosb)
            # down = sinb * cosa * cosθ - sinb * sina * sinθ - sina * cosθ * cosb - sinθ * cosb * cosa
            # up = (rb - k2) * (sina * cosθ + sinθ * cosa) + (ra - k1) * (-sinθ * cosb + sinb * cosθ)
            # dl = (up / down).subs(self.assign)
            # print(dl)
            Va = self.velocity(pa[5], pa[4], pa[3], grid=grid_out_inn)
            Vb = self.velocity(pb[5], pb[4], pb[3], grid=grid_out_inn)
            v_ = (Va + Vb) / 2
            k = 0
            for i in self.vx():
                k = k + sympy.diff(v_, i) * i
            v_ = v_.subs(self.assign)
            time_dl = np.abs(pb['time'] - pa['time'])
            dt_seg += -(time_dl / v_) * k
            # print(dt_seg)
            # print((dl / (v_ * v_)).subs(self.assign))

        return dt_seg

    def path(self, phase, ray, model='prem', dif=False):
        model_pre = TauPyModel(model=model)
        if phase == 'PKdiffP':
            arrivals1 = model_pre.get_ray_paths_geo(ray['dp'], ray['ea'], ray['eo'], ray['sa'], ray['so']
                                                    , ['PKP'], True)

            phase = 'PKdiffP' if len(arrivals1) == 1 else 'PKP'
        arrivals1 = model_pre.get_ray_paths_geo(ray['dp'], ray['ea'], ray['eo'], ray['sa'], ray['so']
                                                , [phase], True)


        # if self.sys_inf is not None:
        #     if len(arrivals1) == 0:
        #         return None


        if len(arrivals1) == 0:
            return 0, 0
        path = arrivals1[0].path
        depth_start_outer_core = self.ICB_PREM - (self.outer_core_layer[0] - 1) * self.outer_core_layer[1]
        depth_end_inner_core = self.ICB_PREM + (self.inner_core_layer[0] - 1) * self.inner_core_layer[1]
        # depth_start_outer_core = self.ICB_PREM - 200
        # depth_end_inner_core = self.ICB_PREM + 400

        qua = 0
        tim = path[-1]['time']
        if self.sys_inf and not diff:
            return qua, tim
        if phase == 'PKIKP':
            depth_max = np.argmax(path['depth'])
            ray_down = path[0:depth_max + 1:1]
            ray_up = path[depth_max:-1:1]

            start1 = find_nearest_index(ray_down['depth'], depth_start_outer_core)
            end1 = np.where(ray_down['depth'] == self.ICB_PREM)[0][0]
            path1 = path[start1:end1 + 1:1]

            qua += self.formula(path1, 'outer')

            end2 = find_nearest_index(ray_down['depth'], depth_end_inner_core)
            path2 = path[end1:end2 + 1:1]
            qua += self.formula(path2, 'inner')

            start3 = find_nearest_index(ray_up['depth'], depth_end_inner_core)
            end3 = np.where(ray_up['depth'] == self.ICB_PREM)[0][0]
            path3 = ray_up[start3:end3 + 1:1]
            qua += self.formula(path3, 'inner')

            end4 = find_nearest_index(ray_up['depth'], depth_start_outer_core)
            path4 = ray_up[end3:end4 + 1:1]
            qua += self.formula(path4, 'outer')

        if phase == 'PKdiffP':

            end1 = np.where(path['depth'] == self.ICB_PREM)[0][0]
            start2 = np.where(path['depth'] == self.ICB_PREM)[0][-1]
            ray_down = path[0:end1 + 1:1]
            ray_up = path[start2: - 1:1]

            start1 = find_nearest_index(ray_down['depth'], depth_start_outer_core)
            path1 = path[start1:end1 + 1:1]
            qua += self.formula(path1, 'outer')

            path2 = path[end1:start2 + 1:1]
            qua += self.formula(path2, 'outer')

            end2 = find_nearest_index(ray_up['depth'], depth_start_outer_core)
            path3 = ray_up[0:end2 + 1:1]
            qua += self.formula(path3, 'outer')


        if phase == 'PKP':
            depth_max = np.argmax(path['depth'])
            if np.max(path['depth']) <= depth_start_outer_core:
                return 0, tim
            else:
                ray_down = path[0:depth_max + 1:1]
                ray_up = path[depth_max:-1:1]
                start1 = find_nearest_index(ray_down['depth'], depth_start_outer_core)

                end2 = find_nearest_index(ray_up['depth'], depth_start_outer_core)

                path1 = ray_down[start1:-1:1]
                qua += self.formula(path1, 'outer')
                path2 = ray_up[0:end2 + 1:1]
                qua += self.formula(path2, 'outer')
        return qua, tim

    def process(self, ray):

        model = 'prem'
        phases = ('PKIKP', "PKdiffP")
        k1, time_act1 = self.path(phases[1], ray, model, dif=True)
        if  time_act1 == 0:
            return None
        k2, time_act0 = self.path(phases[0], ray, model, dif=True)
        qua = k1 - k2

        h = self.he
        a1 = self.diff(qua, weight=h)
        # for i in range(len(a1[0])):
        #     print(i, a1[0][i])
        a2 = self.diff(k1, weight=1)
        a3 = self.diff(k2, weight=1)
        if self.sys_inf:
            model_sys = 'npz/model.npz'
            k1, time_sys_diff = self.path(phases[1], ray, model_sys, dif=False)
            k2, time_sys_I = self.path(phases[0], ray, model_sys, dif=False)
            print(time_sys_diff,time_sys_I, time_act1, time_act0)
            print(time_sys_diff - time_sys_I, time_act1 - time_act0)
            t_obs = - time_act1 + time_act0 + (time_sys_diff - time_sys_I)
            b1 = h * t_obs  # 构建常数项数据data 相对值

            b2 = 1 * (- time_act1 + time_sys_diff)
            b3 = 1 * (- time_act0 + time_sys_I)
        else:
            t_obs = - time_act1 + time_act0 + ((ray['pkpc_r'] - ray['mc_c']) - (ray['pkIkp_r'] - ray['mc_I']))
            print(ray['pkpc_r'] - ray['pkIkp_r'], time_act1 - time_act0, ray['pkpc_p'] - ray['pkIkp_p'])
            b1 = h * t_obs  # 构建常数项数据data 相对值
            b2 = 1 * (- time_act1 + ray['pkpc_r'] - ray['mc_c'])
            b3 = 1 * (- time_act0 + ray['pkIkp_r'] - ray['mc_I'])


        # self.absolute = True
        if abs(t_obs) >= 4:
            print((ray['pkpc_r'], ray['mc_c'], ray['pkIkp_r'], ray['mc_I'], time_act1, time_act0))
            print(ray, t_obs)
            return None
        if self.absolute:
            return np.concatenate((a1, a2, a3), axis=0), np.array((b1, b2, b3), dtype=np.float64)
        else:
            return np.array(a1), np.array(b1, dtype=np.float64)

    def diff(self, pdx, weight=1):
        xcc = self.vx()  # 无实际意义，表示速度函数模型(padans一维矩阵)
        vx_cof = []
        for x in xcc:
            cof = (sympy.diff(pdx, x))
            vx_cof += [weight * cof]  # 构建系数矩阵A
        return np.expand_dims(vx_cof, axis=0)

    def lps(self, l1, neber):
        vk = self.vx()
        lp = []
        outer_core_layer = self.outer_core_layer
        inner_core_layer = self.inner_core_layer
        one_layer = int(len(self.rang) / (outer_core_layer[0] + inner_core_layer[0]))
        for i in range(outer_core_layer[0] + inner_core_layer[0]):
            l_n = layer_nuber = i * one_layer
            for j in range(one_layer):
                point = neber[j]
                j_l = j + l_n  # local grid nuber in one layer
                # print(j_l, point)
                # one layer
                # lp += [l1 * (6 * vk[j_l] - vk[point[0] + l_n] - vk[point[1] + l_n] - vk[
                #     point[2] + l_n] - vk[point[3] + l_n] - vk[point[4] + l_n] - vk[point[5] + l_n])]

                # upper boundary
                if i == 0 or i == outer_core_layer[0]:
                    lp += [l1 * (7 * vk[j_l] - vk[j_l + one_layer] - vk[point[0] + l_n] - vk[point[1] + l_n] - vk[
                        point[2] + l_n] - vk[point[3] + l_n] - vk[point[4] + l_n] - vk[point[5] + l_n])]
                # bottle boundary
                elif i == outer_core_layer[0] - 1 or i == outer_core_layer[0] + inner_core_layer[0] -1:
                    lp += [l1 * (7 * vk[j_l] - vk[j_l - one_layer] - vk[point[0] + l_n] - vk[
                        point[1] + l_n] - vk[point[2] + l_n] - vk[point[3] + l_n] - vk[point[4] + l_n] - vk[
                                     point[5] + l_n])]
                # middle
                else :
                    lp += [l1 * (8 * vk[j_l] - vk[j_l + one_layer] - vk[j_l - one_layer] - vk[point[0] + l_n] - vk[
                        point[1] + l_n] - vk[
                                     point[2] + l_n] - vk[point[3] + l_n] - vk[point[4] + l_n] - vk[point[5] + l_n])]
        # test
        # for i in range(10):
        #     a = random.randint(0, 1809)
        #     print(lp[a], a,a // 362,  [x + 362 * (a // 362) for x in neber[a % 362]])
        m = []
        for i in lp:
            sub_m = []
            for i1 in vk:
                sub_m += [sympy.diff(i, i1)]
            m += [sub_m]
        lps_a = np.array(m, dtype=np.float64)
        lps_d = np.zeros((len(m),))
        return lps_a, lps_d

    def plot_model(self, vp):
        print('pointing')
        vxx = np.load(vp, allow_pickle=True)
        # ve = self.ve_mar(name='own', m=(0.03, 600))
        # for i in range(len(vxx)):
        #     print(i, vxx[i])362
        if len(vxx) < 2000:
            return
        print(np.max(vxx), np.min(vxx))
        grid = list(self.grid_6_latlon[0:1002])

        print(np.max(vxx[0:1002]), np.min(vxx[0:1002]))
        print(np.max(vxx[1002:2004]), np.min(vxx[1002:2004]))

        pygmt.config(MAP_TICK_LENGTH=-0.1, FONT='Times-Roman', FONT_ANNOT_PRIMARY='3p', FONT_LABEL='8p',
                     MAP_FRAME_PEN='0.5p', FONT_TAG='5p')
        fig = pygmt.Figure()
        center1 = 'H180/8c'
        # center1 = 'M8c'
        R = 'g'
        # R = '0/360/-70/70'
        colormap = 'file/flayer2.cpt'
        with fig.subplot(nrows=2, ncols=1, figsize=('16c', '14c'),
                         margins='0.5c', xshift='30c'):
            grid_number = 0
            nanme = ['Outer Core ','OUTER  ICB', 'Inner Core', 'INNER 200', 'INNER 400'] if grid_number == 0 else ['A', "B"]
            with fig.set_panel(panel=0):
                vs = []
                fig.basemap(region=R, projection=center1, frame=["a30f30", "WSNE+t" +nanme[0]])
                for i in range(len(grid)):
                    vx = float(vxx[i + grid_number * len(grid)][0])
                    vs += [[grid[i][0], grid[i][1], vx]]
                print(i + grid_number * len(grid), vs)
                # fig.plot(style='c0.15c', data=vs, cmap=colormap, region=R, projection=center1)
                pygmt.surface(vs, outgrid='file/flayer.net'+str(grid_number), spacing=1, region='-180/180/-90/90')
                fig.grdimage('file/flayer.net'+str(grid_number), cmap= colormap, projection=center1, region=R,
                             interpolation='c')
                #fig.coast(region=R, projection=center1, shorelines='0.01p', area_thresh=100000)
            grid_number += 1
            with fig.set_panel(panel=1):
                vs = []
                for i in range(len(grid)):
                    vx = float(vxx[i + grid_number * len(grid)][0])
                    vs += [[grid[i][0], grid[i][1], vx]]
                # print(vs[3])
                fig.basemap(region=R, projection=center1, frame=["a20f20", "WSNE+t" + nanme[2]])
                pygmt.surface(vs, outgrid='file/flayer.net'+str(grid_number), spacing=1, region='-180/180/-90/90')
                fig.grdimage('file/flayer.net'+str(grid_number), cmap= colormap, projection=center1, region=R,
                             interpolation='c')
                # fig.plot(style='c0.15c', data=vs, cmap=colormap, region=R, projection=center1)
                #fig.coast(region=R, projection=center1, shorelines='0.01p', area_thresh=100000)
                fig.colorbar(cmap=colormap, frame='xa0.04+lVs Relative PREM', position='jTC+w8c/0.5c+o-4c/-0.5c')
            # grid_number += 1
            # with fig.set_panel(panel=2):
            #     vs = []
            #     for i in range(len(grid)):
            #         vx = 2*float(vxx[i + grid_number * len(grid)][0])
            #         vs += [[grid[i][0], grid[i][1], vx]]
            #         depth = str(grid[i][2])
            #
            #     pygmt.surface(vs, outgrid='file/flayer.net'+str(grid_number), spacing=1, region='-180/180/-90/90')
            #     fig.grdimage('file/flayer.net'+str(grid_number), cmap= colormap, projection=center1, region=R,
            #                  interpolation='c')
            #     fig.basemap(region=R, projection=center1, frame=["a20f20", "WSNE+t "+nanme[2]])
            #     # fig.plot(style='c0.15c', data=vs, cmap=colormap, region=R, projection=center1)
            #     fig.coast(region=R, projection=center1, shorelines='0.01p', area_thresh=50)
            # grid_number += 1
            # with fig.set_panel(panel=3):
            #     vs = []
            #     for i in range(len(grid)):
            #         vx = 2*float(vxx[i + grid_number * len(grid)][0])
            #         vs += [[grid[i][0], grid[i][1], vx]]
            #         depth = str(grid[i][2])
            #     pygmt.surface(vs, outgrid='file/flayer.net'+str(grid_number), spacing=1, region='-180/180/-90/90')
            #     fig.grdimage('file/flayer.net'+str(grid_number), cmap= colormap, projection=center1, region=R,
            #                  interpolation='c')
            #     fig.basemap(region=R, projection=center1, frame=["a20f20", "WSNE+t "+nanme[3]])
            #     # fig.plot(style='c0.15c', data=vs, cmap=colormap, region=R, projection=center1)
            #     fig.coast(region=R, projection=center1, shorelines='0.01p', area_thresh=50)
            # grid_number += 1
            # with fig.set_panel(panel=4):
            #     vs = []
            #     for i in range(len(grid)):
            #         vx = 2*float(vxx[i + grid_number * len(grid)][0])
            #         vs += [[grid[i][0], grid[i][1], vx]]
            #         depth = str(grid[i][2])
            #     pygmt.surface(vs, outgrid='file/flayer.net'+str(grid_number), spacing=1, region='-180/180/-90/90')
            #     fig.grdimage('file/flayer.net'+str(grid_number), cmap= colormap, projection=center1, region=R,
            #                  interpolation='c')
            #     fig.basemap(region=R, projection=center1, frame=["a20f20", "WSNE+t "+nanme[4]])
            #     # fig.plot(style='c0.15c', data=vs, cmap=colormap, region=R, projection=center1)
            #     fig.coast(region=R, projection=center1, shorelines='0.01p', area_thresh=50)
                # grid_number += 1
        # fig.colorbar(cmap=colormap, frame='xa0.005+lVs Relative PREM', position='jBR+h+w8c/0.5c+o-2.5c/2c')
        fig.show()
        fig.savefig( vp+'.pdf')

    def sys_test(self, ):
        if self.sys_inf is None:
            return
        outer,inner = self.sys_inf
        velo = [0 for x in self.rang]
        for i in range(0, len(self.rang), 1):
            k = (6371 - self.rang[i] - 2891) / 6371
            a, b, c, d = 11.0487, -4.0362, 4.8023, -13.5732
            depths = self.rang[i]
            v = a + b * k + c * (k ** 2) + d * (k ** 3)
            velo[i] = v + dv / dep * depths - dv if depths <= dep else v
            # if self.rang[i] >= dep:
            #     dv = 0
            # velo[i] = a + b * k + c * (k ** 2) + d * (k ** 3) + dv * (-1) ** (self.rang[i] // inert)
        self.sys_v = Matrix(velo)
        self.model(Matrix(velo), out='sys_test')
        return

    def body(self, cpu_number):
        pool = Pool(cpu_number)
        cc = remv(pool.map(self.process, self.file))
        a1 = np.zeros((0, len(self.v)), dtype=np.float64)
        b1 = np.zeros((0,), dtype=np.float64)
        for i in cc:
            a1 = np.concatenate((a1, np.array(i[0], dtype=np.float64)), axis=0)
            b1 = np.append(b1, i[1])
        b = b1.reshape(int(len(b1) / 3), 3)[:, 0] if self.absolute else b1
        b = b / self.he
        ren = norm(b)
        self.misfit = norm(b) / ren
        # if self.cycle > 0:
        #     self.plot_model(self.v, fi)
        # if self.misfit <= self.final_fit or self.cycle >= cycle_max:
        #     self.model(self.v, out=fi)
        #     break


        alpha = self.lps(self.lpm, self.grid_ne)

        a = np.concatenate((a1, alpha[0]), axis=0)
        d = np.append(b1, alpha[1])
        lsq_v = lsqr(a, d, damp=self.dam, atol=1e-06, btol=1e-06, conlim=1e+8, iter_lim=None, show=False,
                     calc_var=False, x0=None)
        # print(a[0:3], d, lsq_v[0])
        dv = lsq_v[0]
        '''
        indx = int(600 / self.step)
        print(dv)
        if self.cycle == 0:
            for c in np.arange(0, len(dv), 1):
                dv[c] = dv[c] if c <= indx else 0
        print(dv
        '''
        # self.v = Matrix(dv) + self.v
        print(Matrix(dv))
        np.save('V_pkp_5layer.npy', Matrix(dv))


def remv(lis):
    lis = list(filter(None, lis))
    return lis


def change(fil, add=False, step=1, window=2):
    toa = []
    file = fil.keys() if type(fil) == dict else fil
    for key in fil:
        sks = fil[key] if type(fil) == dict else key
        h = sks.head
        d = sks.data
        if np.size(d) == 1:
            # 忽略只有一条数据的事件
            continue
        h = h.tolist()
        # d = d[d['coa'] >= coa]
        for i in d:
            i = i.tolist()
            toa += [h + i]

    return np.array(toa,
                    dtype={'names': ['eo', 'ea', 'dp', 'so', 'sa', 'gc', 'pkIkp_r', 'pkpc_r', 'pkIkp_p', 'pkpc_p',
                                     'mc_I', 'mc_c', 'az', 't3'],
                           'formats': ['f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4',
                                       'f4']})


def i_int(mun):
    # print(mun//10)
    if mun % 10 >= 5:
        return (mun // 10) * 10 + 10
    else:
        return (mun // 10) * 10


if __name__ == '__main__':
    prem = TauPyModel(model='prem')
    n = np.load('pkcdiffp.npy', allow_pickle=True).item()
    f = change(n)
    # taup_create.build_taup_model('npz/model_inner_low.nd', output_folder=os.getcwd() + '/npz')
    # print(len(f))
    # print(f[0:100])
    # print(f)
    r = Iv(f, lpm=300, dam=300, sys_inf=False)
    # for i, index in enumerate(r.ve_mar(name='own', m=(0.03, 600))):
    #     print(i, index)
    # for i in f:
    #     print(i)
    #     a = time.time()
    #     for i3 in r.process(i):d
    #         print(i3)
    #     print(time.time() -a)
    #     break
    # r.body(55)
    #r.plot_model('V_pkp_real_sys_30_nosie0.05s_recovered.npy')
    r.plot_model('V_pkp_real_in_200km.npy')
    # for i in glob.glob('V*npy'):
    #     # if  not os.path.exists(i + '.pdf' ):
    #     r.plot_model(i)



