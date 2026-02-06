"""
================================================================================
Physics-Informed Kolmogorov-Arnold Network (PIKAN) for Multi-Material Problems
================================================================================

This code implements the PIKAN method for analyzing cantilever beam with 
straight material interface (Section 5.1 in the paper), as described in:

Gong, Y., He, Y., Mei, Y., Qin, F., Zhuang, X., & Rabczuk, T. (2026). 
Physics-Informed Kolmogorov-Arnold Networks for multi-material elasticity 
problems in electronic packaging. Applied Mathematical Modelling, 156, 116793. 
https://doi.org/10.1016/j.apm.2026.116793

Description:
    This script solves a two-material cantilever beam bending problem using
    Physics-Informed Kolmogorov-Arnold Networks (PIKAN) with triangular 
    integration scheme. The method employs a single KAN to approximate 
    displacement fields across the entire domain without requiring subdomain 
    decomposition or interface continuity constraints.

Author:
    Yanpeng Gong
    Beijing University of Technology
    Email: yanpenggong@gmail.com

Repository:
    https://github.com/yanpeng-gong/PIKAN-MultiMaterial

Citation:
    Please cite:
    
    @article{Gong2026PIKAN,
        author  = {Gong, Yanpeng and He, Yida and Mei, Yue and Qin, Fei and 
             Zhuang, Xiaoying and Rabczuk, Timon},
        title   = {Physics-informed {Kolmogorov-Arnold} networks for multi-material 
             elasticity problems in electronic packaging},
        journal = {Appl. Math. Model.},
        volume  = {156},
        pages   = {116793},
        year    = {2026},
        doi     = {10.1016/j.apm.2026.116793},
        url     = {https://doi.org/10.1016/j.apm.2026.116793}
    }

Contact:
    For questions, issues, or collaboration inquiries, please contact:
    Yanpeng Gong (yanpenggong@gmail.com)

Last updated: 2025-02-06
================================================================================
"""


import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
from  torch.autograd import grad
from sklearn.neighbors import KDTree
import time
from itertools import chain
import meshio
import sys
import xlrd
import matplotlib.tri as tri
import math
import torch.nn.functional as F

#random seed 定义设置随机数种子的函数
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(2025)

#设置 Matplotlib 的绘图参数，以确保生成的图形具有统一的样式和外观。
mpl.rcParams['figure.dpi'] = 100 #设置图形的分辨率
axes = {'labelsize' : 'large'} #设置坐标轴标签的字体大小为 “large”（大）
font = {'family' : 'Times New Roman', 'weight' : 'normal', 'size' : 17} 
#设置字体类型为 “Times New Roman”，字体粗细为 “normal”（正常），size：设置字体大小为 17。
legend = {'fontsize': 'medium'} #设置图例的字体大小为 “medium”（中等）
lines = {'linewidth': 3, 'markersize' : 7} #设置线条的宽度为 3，设置标记的大小为 7
#使用 mpl.rc 函数将前面定义的样式字典应用到 Matplotlib 的全局设置中
mpl.rc('font', **font) #应用字体样式。
mpl.rc('axes', **axes) #应用坐标轴标签的样式。
mpl.rc('legend', **legend) #应用图例的样式。
mpl.rc('lines', **lines) #应用线条的样式。

# ------------------------------ network settings ---------------------------------------------------
nepoch_u0 = 1000
D_in = 2
D_out = 2
learning_rate = 0.001
model = [2, 5, 5, 5, 2]
# ------------------------------ material parameter -------------------------------------------------
model_energy = 'elasticityMP'
E1 = 8500
nu1 = 0.3
E2 = 43000
nu2 = 0.3
# ----------------------------- define structural parameters ---------------------------------------
dim = 2
Length = 8.0
Height = 2.0
Depth = 1.0
known_left_ux = 0
known_left_uy = 0
bc_left_penalty = 1.0

known_right_tx = 0
known_right_ty = -6.0
bc_right_penalty = 1.0
# ------------------------------ define domain and collocation points -------------------------------
Nx = 401 
Ny = 101
Ni = 1001
x_min, y_min = (0.0, 0.0)
hx = Length / (Nx - 1)
hy = Height / (Ny - 1)
shape = [Nx, Ny]
dxdy = [hx, hy]
# ------------------------------ data testing -------------------------------------------------------
num_test_x = 201
num_test_y = 51

# ----------------------------------------------------------------------
#             STEP 1: SETUP DOMAIN - 三角积分配点
# ----------------------------------------------------------------------
def train_data(Nx, Ny):
    def generate_points(x_min, y_min, Length, Height):
        x_dom = x_min, Length, Nx
        y_dom = y_min, Height, Ny
        # create points
        lin_x = np.linspace(x_dom[0], x_dom[1], x_dom[2])
        lin_y = np.linspace(y_dom[0], y_dom[1], y_dom[2])
        dom = np.zeros((Nx * Ny, 2))
        c = 0
        for x in np.nditer(lin_x):
            tb = y_dom[2] * c
            te = tb + y_dom[2]
            c += 1
            dom[tb:te, 0] = x
            dom[tb:te, 1] = lin_y
        return np.array(dom)

    # Generate the points
    points = generate_points(x_min, y_min, Length, Height)   #生成均匀分布点
    
    points1 = points[(points[:, 1] >= 1)]
    points2 = points[(points[:, 1] < 1)]
    
    def generate_triangle(points):
        # Create a Delaunay triangulation
        triangulation = tri.Triangulation(points[:, 0], points[:, 1])
    
        # Plot the triangulation  画出三角形网格
        #plt.figure(figsize=(10, 10))
        #plt.gca().set_aspect('equal')
        #plt.triplot(triangulation, lw=0.5, color='blue')
        #plt.xlim(x_min-0.2, Length+0.2)
        #plt.ylim(y_min-0.2, Height+0.2)
        #plt.title("Triangular Mesh for Domain")
        #plt.xlabel("x")
        #plt.ylabel("y")
        #plt.grid(True)
        #plt.show()
    
        # Calculate the area of each triangle
        triangles = points[triangulation.triangles]
        a = np.linalg.norm(triangles[:, 0] - triangles[:, 1], axis=1)
        b = np.linalg.norm(triangles[:, 1] - triangles[:, 2], axis=1)
        c = np.linalg.norm(triangles[:, 2] - triangles[:, 0], axis=1)
        s = 0.5 * (a + b + c)
        areas = np.sqrt(s * (s - a) * (s - b) * (s - c))
    
        # Sum of all triangle areas
        dom_point = triangles.mean(1)
        total_area = np.sum(areas)
        Xf = np.hstack((dom_point, areas[:, np.newaxis]))
        return Xf
    
    Xf1 = generate_triangle(points1) # 上区域点，去除内部多配的点 
    Xf2 = generate_triangle(points2) # 下区域点，去除内部多配的点 

    # ------------------------------------ BOUNDARY ----------------------------------------
    dom = generate_points(x_min, y_min, Length, Height)
    
    # Left boundary condition (Dirichlet BC)
    bcl_u_pts_idx_left = np.where(dom[:, 0] == x_min) # 找到 x=0的坐标位置
    bcl_u_pts_left = dom[bcl_u_pts_idx_left, :][0] # 找到左边界条件的坐标点
    bcl_u_left = np.ones(np.shape(bcl_u_pts_left)) * [known_left_ux, known_left_uy]
    boundary_dirichlet = {
        # condition on the left
        "dirichlet_1": {
            "coord": bcl_u_pts_left,
            "known_value": bcl_u_left,
        },
        # adding more boundary condition here ...
    }
    
    # Right boundary condition (Neumann BC)
    bcr_t_pts_idx = np.where(dom[:, 0] == Length)
    bcr_t_pts = dom[bcr_t_pts_idx, :][0]
    bcr_t_pts1 = bcr_t_pts[bcr_t_pts[:, 1]>=1]
    bcr_t_pts2 = bcr_t_pts[bcr_t_pts[:, 1]<1]   
    bcr_t = np.ones(np.shape(bcr_t_pts)) * [known_right_tx, known_right_ty]
    bcr_t1 = bcr_t[bcr_t_pts[:, 1]>=1]
    bcr_t2 = bcr_t[bcr_t_pts[:, 1]<1]
    boundary_neumann = {
        # condition on the right
        "neumann": {
            "coord": bcr_t_pts,
            "known_value": bcr_t,
        }
    }
    boundary_neumann1 = {
        # 材料 1
        "neumann_1": {
            "coord": bcr_t_pts1,
            "known_value": bcr_t1,
        }
    }
    boundary_neumann2 = {
        # 材料 2
        "neumann_2": {
            "coord": bcr_t_pts2,
            "known_value": bcr_t2,
        }
    }
    #Xn = torch.tensor(bcr_t_pts, requires_grad=True, device='cuda').float()

    return Xf1, Xf2, boundary_dirichlet, boundary_neumann, boundary_neumann1, boundary_neumann2

# ----------------------------------------------------------------------
#                   STEP 2: SETUP MODEL
# ----------------------------------------------------------------------
class EnergyModel:
    # ---------------------------------------------------------------------------------------------------------------------------
    def __init__(self, energy, dim, E=None, nu=None):
        """
        Parameters
        ----------
        energy : TYPE 字符串，表示能量模型的类型
            DESCRIPTION.
        dim : TYPE 整数，表示模型的维度（2 或 3）
            DESCRIPTION.
        E : TYPE, optional 弹性模量（Young's modulus），默认值为 None
            DESCRIPTION. The default is None.
        nu : TYPE, optional 泊松比（Poisson's ratio），默认值为 None
            DESCRIPTION. The default is None.
        Returns
        -------
        None.
        """
        self.type = energy
        self.dim = dim
        if self.type == 'elasticityMP':
            if dim == 2: #平面应力问题的弹性矩阵
                self.D11_mat = E/(1-nu**2)
                self.D22_mat = E/(1-nu**2)
                self.D12_mat = E*nu/(1-nu**2)
                self.D21_mat = E*nu/(1-nu**2)
                self.D33_mat = E/(2*(1+nu))
            if dim ==3: # 以后在三维问题再进行定义
                pass
    def getStoredEnergy(self, u, x): # 根据模型类型选择相应计算应变能密度的函数，将每一个点位移以及位置输入到这个函数中，获得每一个点的应变能密度
        """
        根据模型类型选择相应计算应变能密度的函数，将每一个点位移以及位置输入到这个函数中，获得每一个点的应变能密度。
        """
        if self.type == 'elasticityMP': # 最小势能原理，线弹性
            if self.dim == 2:
                return self.Elasticity2DMP(u, x)
            if self.dim == 3:
                return self.Elasticity3DMP(u, x)
    # 最小势能原理，线弹性
    def Elasticity2DMP(self, ust, x):
        """
        参数：
        ust：一个二维张量，包含位移场的信息。ust[:, 0] 表示 x 方向的位移，ust[:, 1] 表示 y 方向的位移。
        x：一个二维张量，表示节点的坐标。
        """
        duxdxy = grad(ust[:, 0].unsqueeze(1), x, torch.ones(x.size()[0], 1, device='cuda'), create_graph=True, retain_graph=True)[0] #duxdxy=(∂ux/∂x,∂ux/∂y)
        duydxy = grad(ust[:, 1].unsqueeze(1), x, torch.ones(x.size()[0], 1, device='cuda'), create_graph=True, retain_graph=True)[0] #duydxy=(∂uy/∂x,∂uy/∂y)       
        dudx = duxdxy[:, 0].unsqueeze(1) #∂ux/∂x
        dudy = duxdxy[:, 1].unsqueeze(1) #∂ux/∂y
        dvdx = duydxy[:, 0].unsqueeze(1) #∂uy/∂x
        dvdy = duydxy[:, 1].unsqueeze(1) #∂uy/∂y
        #根据线弹性理论，计算应变分量
        sxx = dudx #εxx
        syy = dvdy #εyy
        s2xy = dudy + dvdx #εxy

        # 经过张量分析理论分析，这是对的  Energy = 1/2*(D*ε*ε)
        strainEnergy = 0.5 * (self.D11_mat * sxx ** 2  + 2*self.D12_mat * sxx * syy + self.D22_mat * syy ** 2 + self.D33_mat * s2xy ** 2)
        # 获得应变能密度的泛函
        return strainEnergy
    
class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        '''
        这是类的初始化方法，定义了模块的参数
        '''
        super(KANLinear, self).__init__() #将输入参数保存为类的属性。
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size #作用：计算网格点之间的间隔 h。
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        ) #作用：生成网格点的位置，并将其扩展为与输入特征维度匹配的张量。
        self.register_buffer("grid", grid) #作用：将生成的网格点张量注册为一个缓冲区（buffer）

        #定义基础权重和样条权重作为可训练参数，定义了 KANLinear。模块中的两个可训练参数：base_weight 和 spline_weight，它们分别用于线性变换和样条曲线变换。
        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        #如果启用了独立的样条缩放因子，则定义一个额外的可训练参数 spline_scaler
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        #将其他参数保存为类的属性。
        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()
        
    
    #定义 reset_parameters 方法，用于重置 KANLinear模块中的参数。它包括基础权重 base_weight和样条系数 spline_weight的更新，
    #以及可选的独立样条缩放因子 spline_scaler的更新。
    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        #基础权重 base_weight：使用 Kaiming 均匀初始化方法初始化，范围由 math.sqrt(5) * self.scale_base 控制。
        
        #初始化样条权重 spline_weight
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            #如果启用了独立样条缩放因子，（类最开始部分设置为 True，因此执行此部分）则使用 Kaiming均匀初始化方法初始化。
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)
    
    def b_splines(self, x: torch.Tensor): #定义了计算 B-spline基函数的方法 
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        #将输入张量 x 扩展为三维张量，并计算初始的基函数
        
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )
        #通过递归计算更高阶的 B-spline 基函数

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous() #返回计算得到的基函数
    
    #用于计算 B样条系数的函数
    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()
    
    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )


    #定义前向传播方法，将输入张量 x 转换为输出张量。这个方法结合了传统的线性变换和基于样条曲线的非线性变换
    def forward(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features #作用：验证输入张量 x 的形状是否符合预期。
        
        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        ) 
        return base_output + spline_output
    
    #定义了一个名为 update_grid 的方法，用于更新 KANLinear 模块中的网格点（grid）。这个方法的作用是根据输入数据 x 动态调整网格的范围，以更好地适应数据的分布。
    @torch.no_grad() #使用 @torch.no_grad() 装饰器可以确保 update_grid 方法在执行时不会计算梯度，从而节省计算资源并提高效率
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features #代码中只定义了输入验证部分,并没有实际更新 grid部分。


    #定义了一个名为 regularization_loss 的方法，用于计算正则化损失。正则化损失通常用于防止模型过拟合，并且在这段代码中，它结合了 L1 正则化和熵正则化。
    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss. 计算正则化损失

        L1 and the entropy loss is for the feature selection, i.e., let the weight of the activation function be small.
        L1 损失和熵损失用于特征选择，即让激活函数的权重(样条系数 Cij)变得更小
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )
    
class KAN(torch.nn.Module): #定义了一个名为 KAN的神经网络模型，它由多个 KANLinear层组成。
    # 定义了一个继承自 torch.nn.Module的类 KAN，用于构建一个包含多个 KANLinear层的神经网络
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order #将 grid_size 和 spline_order 保存为类的属性，供后续使用。

        #创建一个 torch.nn.ModuleList，用于存储网络的所有层。ModuleList 是 PyTorch 提供的一个容器，用于管理多个模块
        self.layers = torch.nn.ModuleList() 
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )


    def forward(self, x: torch.Tensor, update_grid=False): #定义了前向传播方法 forward，这是 PyTorch 中所有神经网络模块必须实现的方法。
        for index, layer in enumerate(self.layers):
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
            if index < len(self.layers)-1: # 如果不是最后一层，拉到 -1到 1之间，最后一层不需要 tanh,检查当前层是否是最后一层。如果不是最后一层，则对输出应用 torch.tanh 激活函数
                x = torch.tanh(x) #将输入张量 x 的每个元素应用双曲正切函数，将其值限制在 [-1, 1] 范围内
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0): #这是 KAN 类的一个方法，用于计算整个网络的正则化损失。
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )
    
class IntegrationLoss:
    def __init__(self, numIntType, dim):
        print("Constructor: IntegrationLoss ", numIntType, " in ", dim, " dimension ")
        self.type = numIntType
        self.dim = dim

    def lossInternalEnergy(self, f, x=None, dx=1.0, dy=1.0, dz=1.0, shape=None): #计算应变能密度的函数
        return self.approxIntegration(f, x, dx, dy, dz, shape) #它调用 approxIntegration 方法来执行实际的数值积分

    def lossExternalEnergy(self, f, x=None, dx=1.0, dy=1.0, dz=1.0, shape=None): #计算外力功的函数
    
        if self.type == 'trapezoidal': # 'trapezoidal'使用梯形积分方法。
            # print("Trapezoidal rule")
            if self.dim == 2:
                if x is not None: 
                    return self.trapz1D(f, x=x) 
                else: 
                    return self.trapz1D(f, dx=dx)
            if self.dim == 3:
                if x is not None:
                    return self.trapz2D(f, xy=x, shape=shape)
                else:
                    return self.trapz2D(f, dx=dx, dy=dy, shape=shape)
        if self.type == 'simpson': #'simpson'使用辛普森积分方法。           
            # print("Simpson rule")
            if self.dim == 2:
                if x is not None: 
                    return self.simps1D(f, x=x)
                else: 
                    return self.simps1D(f, dx=dx)
            if self.dim == 3: 
                if x is not None:
                    return self.simps2D(f, xy=x, shape=shape)
                else:
                    return self.simps2D(f, dx=dx, dy=dy, shape=shape)

    def approxIntegration(self, f, x=None, dx=1.0, dy=1.0, dz=1.0, shape=None): #实际计算应变能密度的函数
        if self.type == 'trapezoidal': #'trapezoidal'使用梯形积分方法。
            # print("Trapezoidal rule")
            if self.dim == 1: #一维问题
                if x is not None: 
                    return self.trapz1D(f, x=x)
                else:
                    return self.trapz1D(f, dx=dx) 
            if self.dim == 2: #二维问题
                if x is not None:
                    return self.trapz2D(f, xy=x, shape=shape)
                else: 
                    return self.trapz2D(f, dx=dx, dy=dy, shape=shape)
            if self.dim == 3: #三维问题
                if x is not None:
                    return self.trapz3D(f, xyz=x, shape=shape)
                else:
                    return self.trapz3D(f, dx=dx, dy=dy, dz=dz, shape=shape)
        if self.type == 'simpson':
            # print("Simpson rule")
            if self.dim == 1: #一维
                if x is not None:
                    return self.simps1D(f, x=x) 
                else:
                    return self.simps1D(f, dx=dx) #等间距配点，间距 dx
            if self.dim == 2: #二维
                if x is not None:
                    return self.simps2D(f, xy=x, shape=shape)
                else:
                    return self.simps2D(f, dx=dx, dy=dy, shape=shape)
            if self.dim == 3: #三维
                if x is not None:
                    return self.simps3D(f, xyz=x, shape=shape)
                else:
                    return self.simps3D(f, dx=dx, dy=dy, dz=dz, shape=shape)

    def trapz1D(self, y, x=None, dx=1.0, axis=-1):
        
        '''
        定义了一个名为 trapz1D 的方法，用于计算一维数据的梯形积分。
        '''
        y1D = y.flatten() 
        if x is not None: 
            x1D = x.flatten()
            return self.trapz(y1D, x1D, dx=dx, axis=axis)
        else:
            return self.trapz(y1D, dx=dx)

    def trapz2D(self, f, xy=None, dx=None, dy=None, shape=None):
        '''
        定义了一个名为 trapz2D 的方法，用于计算二维数据的梯形积分。
        '''
        f2D = f.reshape(shape[0], shape[1])
        if dx is None and dy is None:
            x = xy[:, 0].flatten().reshape(shape[0], shape[1])
            y = xy[:, 1].flatten().reshape(shape[0], shape[1])
            return self.trapz(self.trapz(f2D, y[0, :]), x[:, 0])
        else: #如果提供了 dx 和 dy，则表示数据点的间距是均匀的
            return self.trapz(self.trapz(f2D, dx=dy), dx=dx)

    #定义了一个名为 trapz3D 的方法，用于计算三维数据的梯形积分。
    def trapz3D(self, f, xyz=None, dx=None, dy=None, dz=None, shape=None):
        f3D = f.reshape(shape[0], shape[1], shape[2])
        if dx is None and dy is None and dz is None:
            print("dxdydz - trapz3D - Need to implement !!!")
        else:
            return self.trapz(self.trapz(self.trapz(f3D, dx=dz), dx=dy), dx=dx)

    def simps1D(self, f, x=None, dx=1.0, axis=-1):
        '''
        定义了一个名为 simps1D 的方法，用于计算一维数据的辛普森积分（Simpson's Rule）。
        '''
        f1D = f.flatten()
        if x is not None:               
            x1D = x.flatten()
            return self.simps(f1D, x1D, dx=dx, axis=axis)
        else: #如果未提供 x 参数，则表示数据点的间距是均匀的
            return self.simps(f1D, dx=dx, axis=axis)

    def simps2D(self, f, xy=None, dx=None, dy=None, shape=None):
        '''
        定义了一个名为 simps2D 的方法，用于计算二维数据的辛普森积分（Simpson's Rule）。
        '''
        f2D = f.reshape(shape[0], shape[1])
        if dx is None and dy is None: 
            x = xy[:, 0].flatten().reshape(shape[0], shape[1]) 
            y = xy[:, 1].flatten().reshape(shape[0], shape[1])  
            return self.simps(self.simps(f2D, y[0, :]), x[:, 0])
        else:
            return self.simps(self.simps(f2D, dx=dy), dx=dx)
            
    def simps3D(self, f, xyz=None, dx=None, dy=None, dz=None, shape=None):
        f3D = f.reshape(shape[0], shape[1], shape[2])
        if dx is None and dy is None and dz is None:
            print("dxdydz - trapz3D - Need to implement !!!")
        else:
            return self.simps(self.simps(self.simps(f3D, dx=dz), dx=dy), dx=dx)
    
    # 计算辛普森积分的函数
    def simps(self, y, x=None, dx=1, axis=-1, even='avg'):
        # import scipy.integrate as sp
        # sp.simps()
        # y = torch.tensor(y)
        nd = len(y.shape)
        N = y.shape[axis]
        last_dx = dx
        first_dx = dx
        returnshape = 0
        if x is not None:
            # x = torch.tensor(x)
            if len(x.shape) == 1:
                shapex = [1] * nd
                shapex[axis] = x.shape[0]
                saveshape = x.shape
                returnshape = 1
                x = x.reshape(tuple(shapex))
            elif len(x.shape) != len(y.shape):
                raise ValueError("If given, shape of x must be 1-d or the "
                                 "same as y.")
            if x.shape[axis] != N:
                raise ValueError("If given, length of x along axis must be the "
                                 "same as y.")
        if N % 2 == 0:
            val = 0.0
            result = 0.0
            slice1 = (slice(None),) * nd
            slice2 = (slice(None),) * nd
            if even not in ['avg', 'last', 'first']:
                raise ValueError("Parameter 'even' must be "
                                 "'avg', 'last', or 'first'.")
            # Compute using Simpson's rule on first intervals
            if even in ['avg', 'first']:
                slice1 = self.tupleset(slice1, axis, -1) # 这里axis=-1
                slice2 = self.tupleset(slice2, axis, -2)
                if x is not None:
                    last_dx = x[slice1] - x[slice2]
                val += 0.5 * last_dx * (y[slice1] + y[slice2]) # slice是两个维度的切片，元组的第一个元素是行切片，第二个元素是列切片,这是一个梯形公式
                result = self._basic_simps(y, 0, N - 3, x, dx, axis)
            # Compute using Simpson's rule on last set of intervals
            if even in ['avg', 'last']:
                slice1 = self.tupleset(slice1, axis, 0)
                slice2 = self.tupleset(slice2, axis, 1)
                if x is not None:
                    first_dx = x[tuple(slice2)] - x[tuple(slice1)]
                val += 0.5 * first_dx * (y[slice2] + y[slice1])
                result += self._basic_simps(y, 1, N - 2, x, dx, axis)
            if even == 'avg':
                val /= 2.0
                result /= 2.0
            result = result + val
        else:
            result = self._basic_simps(y, 0, N - 2, x, dx, axis)
        if returnshape:
            x = x.reshape(saveshape)
        return result

    #定义计算辛普森积分公式中用到
    def tupleset(self, t, i, value):
        l = list(t) # 因为 tuple是不可 change的，所以这里将 tuple先变成了 list
        l[i] = value # 对 i位置上进行赋值
        return tuple(l) # 将 list变回 tuple

    #定义计算辛普森积分公式中用到
    def _basic_simps(self, y, start, stop, x, dx, axis):
        nd = len(y.shape)
        if start is None:
            start = 0
        step = 2
        slice_all = (slice(None),) * nd
        slice0 = self.tupleset(slice_all, axis, slice(start, stop, step))
        slice1 = self.tupleset(slice_all, axis, slice(start + 1, stop + 1, step))
        slice2 = self.tupleset(slice_all, axis, slice(start + 2, stop + 2, step))

        if x is None:  # Even spaced Simpson's rule.
            result = torch.sum(dx / 3.0 * (y[slice0] + 4 * y[slice1] + y[slice2]), axis)
        else:
            # Account for possibly different spacings.
            #    Simpson's rule changes a bit.
            # h = np.diff(x, axis=axis)
            if axis == 0:
                h = x[1:, 0:1] - x[:-1, 0:1]
            elif axis == -1:
                if len(x.shape)==2:
                    ht = x[:, 1:] - x[:, :-1] 
                if len(x.shape)==1:
                    ht = x[1:] - x[:-1]
            
            sl0 = self.tupleset(slice_all, axis, slice(start, stop, step))
            sl1 = self.tupleset(slice_all, axis, slice(start + 1, stop + 1, step))
            h0 = ht[sl0]
            h1 = ht[sl1]
            hsum = h0 + h1
            hprod = h0 * h1
            h0divh1 = h0 / h1
            tmp = hsum / 6.0 * (y[slice0] * (2 - 1.0 / h0divh1) +
                                y[slice1] * hsum * hsum / hprod +
                                y[slice2] * (2 - h0divh1))
            result = torch.sum(tmp, dim=axis)
        return result

    #梯形积分计算函数
    def trapz(self, y, x=None, dx=1.0, axis=-1):
        # y = np.asanyarray(y)
        if x is None:
            d = dx
        else:
            d = x[1:] - x[0:-1]
            # reshape to correct shape
            shape = [1] * y.ndimension()
            shape[axis] = d.shape[0]
            d = d.reshape(shape)
        nd = y.ndimension()
        slice1 = [slice(None)] * nd
        slice2 = [slice(None)] * nd
        slice1[axis] = slice(1, None)
        slice2[axis] = slice(None, -1)
        ret = torch.sum(d * (y[tuple(slice1)] + y[tuple(slice2)]) / 2.0, axis)
        return ret
    
def ele2d(fxy, area): # 计算三角形积分的应变能密度
    '''
    fxy : energy density
    area: element area
    '''
    return torch.sum(fxy*area)

def pred(xy):
    xy_scale = xy / Length    
    #上半区域的预测
    pred = model_k(xy_scale)
    Ux = xy[:, 0] * pred[:, 0]
    Uy = xy[:, 0] * pred[:, 1]
    Ux = Ux.reshape(Ux.shape[0], 1)
    Uy = Uy.reshape(Uy.shape[0], 1)
    u_pred = torch.cat((Ux, Uy), -1)
    return u_pred 

# ----------------------------------------------------------------------
#                   STEP 3: TRAINING MODEL
# ----------------------------------------------------------------------
#实例化 kan网络
model_k = KAN(model, base_activation=torch.nn.SiLU, grid_size=10, grid_range=[0, 1], spline_order=3).cuda()
criterion = torch.nn.MSELoss() #计算 MSE
optimizer = torch.optim.LBFGS(params=model_k.parameters(), lr=learning_rate, max_iter=20)
#scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[500, 3000, 5000], gamma = 0.1)
loss_array = [] #总模型势能历史
loss1_array = [] #材料 1中势能变化历史
loss2_array = [] #材料 2中势能变化历史  
loss_internal_array = [] #总模型内能历史
loss_external_array = [] #总模型外力功历史

#三角形积分配点
Xf1_J, Xf2_J, dirichletBC, neumannBC, neumannBC1, neumannBC2 = train_data(Nx, Ny)
Xf1 = torch.from_numpy(Xf1_J[:,0:2]).float() # 将 data中的前两个维度放入X作为输入
J1 = torch.from_numpy(Xf1_J[:,2, np.newaxis]).float()
Xf2 = torch.from_numpy(Xf2_J[:,0:2]).float() # 将 data中的前两个维度放入X作为输入
J2 = torch.from_numpy(Xf2_J[:,2, np.newaxis]).float()
    
# 将 Xf1 和 Xf2 设置为 requires_grad=True 并移动到 GPU
Xf1 = Xf1.requires_grad_(True).to(device='cuda')
Xf2 = Xf2.requires_grad_(True).to(device='cuda')
# 将 J1 和 J2 仅移动到 GPU
J1 = J1.to(device='cuda')
J2 = J2.to(device='cuda')

# -------------------------------------------------------------------------------
#                             Dirichlet BC
# -------------------------------------------------------------------------------
dirBC_coordinates = {}  # declare a dictionary
dirBC_values = {}  # declare a dictionary
for i, keyi in enumerate(dirichletBC):
    dirBC_coordinates[i] = torch.from_numpy(dirichletBC[keyi]['coord']).float().to(device='cuda')
    dirBC_values[i] = torch.from_numpy(dirichletBC[keyi]['known_value']).float().to(device='cuda')
# -------------------------------------------------------------------------------
#                           Neumann BC
# -------------------------------------------------------------------------------
neuBC_coordinates = {}  # declare a dictionary
neuBC_values = {}  # declare a dictionary
for i, keyi in enumerate(neumannBC):
    neuBC_coordinates[i] = torch.from_numpy(neumannBC[keyi]['coord']).float().to(device='cuda')
    neuBC_coordinates[i].requires_grad_(True) # 这里感觉不用设置梯度为True
    neuBC_values[i] = torch.from_numpy(neumannBC[keyi]['known_value']).float().to(device='cuda')

neuBC_coordinates1 = {}  # declare a dictionary
neuBC_values1 = {}  # declare a dictionary
for i, keyi in enumerate(neumannBC1):
    neuBC_coordinates1[i] = torch.from_numpy(neumannBC1[keyi]['coord']).float().to(device='cuda')
    #neuBC_coordinates1[i].requires_grad_(True) # 这里感觉不用设置梯度为True
    neuBC_values1[i] = torch.from_numpy(neumannBC1[keyi]['known_value']).float().to(device='cuda')

neuBC_coordinates2 = {}  # declare a dictionary
neuBC_values2 = {}  # declare a dictionary
for i, keyi in enumerate(neumannBC2):
    neuBC_coordinates2[i] = torch.from_numpy(neumannBC2[keyi]['coord']).float().to(device='cuda')
    #neuBC_coordinates2[i].requires_grad_(True) # 这里感觉不用设置梯度为True
    neuBC_values2[i] = torch.from_numpy(neumannBC2[keyi]['known_value']).float().to(device='cuda')
    
# ----------------------------------------------------------------------------------
# Minimizing loss function (energy and boundary conditions)
# ----------------------------------------------------------------------------------
neo0 = EnergyModel('elasticityMP', dim, E1, nu1) #计算上区域材料的应变能的类
neo1 = EnergyModel('elasticityMP', dim, E2, nu2) #计算下区域材料的应变能的类
intLoss = IntegrationLoss('simpson', dim)
nepoch_u0 = int(nepoch_u0)
start = time.time() #记录训练开始的时间
for epoch in range(nepoch_u0):   
    def closure():
        # ----------------------------------------------------------------------------------
        # Internal Energy
        # ----------------------------------------------------------------------------------
        #使用 pred 函数对两个域的点进行预测
        u_pred1 = pred(Xf1) #材料 1域位移
        u_pred2 = pred(Xf2) #材料 2域位移
        #计算应变能     
        storedEnergy1 = neo0.getStoredEnergy(u_pred1, Xf1) #计算材料1域的应变能
        storedEnergy2 = neo1.getStoredEnergy(u_pred2, Xf2) #计算材料2域的应变能
        
        internal1 = ele2d(storedEnergy1, J1)
        internal2 = ele2d(storedEnergy2, J2)
        
        # ----------------------------------------------------------------------------------
        # External Energy
        # ----------------------------------------------------------------------------------
        external = torch.zeros(len(neuBC_coordinates))
        for i, vali in enumerate(neuBC_coordinates):
            neu_ust_pred = pred(neuBC_coordinates[i])
            neu_u_pred = neu_ust_pred[:,(0, 1)]
            fext = torch.bmm((neu_u_pred).unsqueeze(1), neuBC_values[i].unsqueeze(2))
            external[i] = intLoss.lossExternalEnergy(fext, dx=dxdy[1])
        
        energy_loss = internal1 + internal2 - torch.sum(external) #整个计算域势能
        
        external1 = torch.zeros(len(neuBC_coordinates1))
        for i, vali in enumerate(neuBC_coordinates1):
            neu_ust_pred1 = pred(neuBC_coordinates1[i])
            neu_u_pred1 = neu_ust_pred1[:,(0, 1)]
            fext1 = torch.bmm((neu_u_pred1).unsqueeze(1), neuBC_values1[i].unsqueeze(2))
            external1[i] = intLoss.lossExternalEnergy(fext1, dx=dxdy[1])
            
        external2 = torch.zeros(len(neuBC_coordinates2))
        for i, vali in enumerate(neuBC_coordinates2):
            neu_ust_pred2 = pred(neuBC_coordinates2[i])
            neu_u_pred2 = neu_ust_pred2[:,(0, 1)]
            fext2 = torch.bmm((neu_u_pred2).unsqueeze(1), neuBC_values2[i].unsqueeze(2))
            external2[i] = intLoss.lossExternalEnergy(fext2, dx=dxdy[1])
        
        
        energy_loss1 = internal1 #材料 1势能
        energy_loss2 = internal2 #材料 2势能
        internal_loss = internal1 + internal2 #模型内能
        external_loss = torch.sum(external) #模型外力功
        
        loss = energy_loss
        optimizer.zero_grad()
        loss.backward()
        loss_array.append(loss.data.cpu())
        loss1_array.append(energy_loss1.data.cpu())
        loss2_array.append(energy_loss2.data.cpu())
        loss_internal_array.append(internal_loss.data.cpu())
        loss_external_array.append(external_loss.data.cpu())
        
        print('Iter: %d Loss: %.9e Internal Energy: %.9e  External Energy: %.9e  Energy1: %.9e  Energy2: %.9e' 
              % (epoch + 1, loss.item(), (internal1 + internal2).item(), torch.sum(external).item(), energy_loss1.item(), energy_loss2.item()))
        return loss
    optimizer.step(closure)
    #scheduler.step()
end = time.time()
consume_time = end-start #记录当前时间，并计算从训练开始到现在的总耗时。
print('time is %f' % consume_time)

# 保存训练好的网络
# Save the trained networks
from datetime import datetime
current_time = datetime.now().strftime("%Y%m%d_%H%M%S") # current_time变量将存储一个格式化的当前时间字符串，例如 20231211_235959。
torch.save(model_k.state_dict(), f'./Net/model_k_triangle_grid10order3{current_time}.pth')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 恢复训练好的神经网络模型
model_k = KAN(model, base_activation=torch.nn.SiLU, grid_size=10, grid_range=[0, 1], spline_order=3).to(device)
model_k.load_state_dict(torch.load(f'./NET/model_k_triangle_grid10order3{current_time}.pth'))

# 计算总参数数量
total_params = sum(p.numel() for p in model_k.parameters())
print(f"Total number of parameters: {total_params}")

def generate_testpoints(x_min, y_min, Length, Height, num_test_x, num_test_y):
        x_dom = x_min, Length, num_test_x
        y_dom = y_min, Height, num_test_y
        # create points
        lin_x = np.linspace(x_dom[0], x_dom[1], x_dom[2])
        lin_y = np.linspace(y_dom[0], y_dom[1], y_dom[2])
        dom = np.zeros((num_test_x * num_test_y, 2))
        c = 0
        for x in np.nditer(lin_x):
            tb = y_dom[2] * c
            te = tb + y_dom[2]
            c += 1
            dom[tb:te, 0] = x
            dom[tb:te, 1] = lin_y
        return np.array(dom)

num_test_x = 201
num_test_y = 51
testpoints = generate_testpoints(x_min, y_min, Length, Height, num_test_x, num_test_y)

def testpred(xy):
    xy_scale = xy / Length    
    #上半区域的预测
    pred = model_k(xy_scale)
    Ux = xy[:, 0] * pred[:, 0]
    Uy = xy[:, 0] * pred[:, 1]
    Ux = Ux.reshape(Ux.shape[0], 1)
    Uy = Uy.reshape(Uy.shape[0], 1)
    u_pred = torch.cat((Ux, Uy), -1)
    return u_pred

xy_test = torch.from_numpy(testpoints).float()
xy_test = xy_test.to(device='cuda')
pred = testpred(xy_test)
U = pred.detach().cpu().numpy()
Ux = U[:, 0].copy() # 整个计算域配点x方向位移
Uy = U[:, 1].copy()
# 计算绝对位移 Umag
Umag = np.sqrt(Ux**2 + Uy**2)
np.savetxt("testpoints.csv", testpoints, delimiter=',', fmt='%f')
np.savetxt("Ux.csv", Ux, delimiter=',', fmt='%f')
np.savetxt("Uy.csv", Uy, delimiter=',', fmt='%f')
np.savetxt("Umag.csv", Umag, delimiter=',', fmt='%f')

from matplotlib import cm #cm：matplotlib 的颜色映射模块，用于定义颜色映射表
from matplotlib.ticker import LinearLocator, FixedLocator, ScalarFormatter
#LinearLocator 和 FixedLocator：用于自定义坐标轴刻度的位置。ScalarFormatter：用于格式化颜色条的刻度标签，支持数学符号显示。

#设置绘图时使用的字体为 Times New Roman，并将全局字体大小设置为 12。这会影响图表中的标题、坐标轴标签、刻度标签等的字体样式。
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12

xmin, xmax = np.min(testpoints[:, 0]), np.max(testpoints[:, 0])  # 根据数据动态调整x轴范围
ymin, ymax = np.min(testpoints[:, 1]), np.max(testpoints[:, 1])  # 根据数据动态调整y轴范围

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(19, 1.8)) #创建一个包含两个子图的图形，排列为 1 行 2 列，整个图形的大小为宽度 12 英寸、高度 4 英寸。
fig.subplots_adjust(hspace=0.6, wspace=0.17) #调整子图之间的间距，hspace 控制垂直方向的间距，wspace 控制水平方向的间距。
#fig.suptitle("Elasticity_bend_Two") #在整个图形的顶部添加一个总标题

# 定义绘制等值图函数
def plot_contour(ax, data, title, vmin, vmax, fontsize=17, tick_fontsize=14, cbar_tick_fontsize=14):
    # fontsize：标题和坐标轴标签的字体大小，默认为 20。tick_fontsize：坐标轴刻度的字体大小，默认为 14。cbar_tick_fontsize：颜色条刻度的字体大小，默认为 12。
    cf = ax.scatter(testpoints[:, 0], testpoints[:, 1], s=5, c=data, cmap=cm.jet, vmin=vmin, vmax=vmax, rasterized=True)
    ax.axis('equal')
    cbar = plt.colorbar(cf, ax=ax, format=ScalarFormatter(useMathText=True), pad=0.03)  # 使用ScalarFormatter
    cbar.ax.tick_params(labelsize=cbar_tick_fontsize)
    
    # 设置颜色条的刻度范围，确保显示最大值、最小值和中间刻度
    ticks = LinearLocator(numticks=6)  # 指定 10个刻度
    cbar.set_ticks(ticks.tick_values(vmin, vmax))  # 设置刻度位置
    cbar.set_ticklabels([f"{tick:.2e}" for tick in ticks.tick_values(vmin, vmax)])  # 使用科学计数法格式化刻度标签

    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    ax.set_title(title, fontsize=fontsize)
    #ax.set_xlabel('x (mm)', fontsize=fontsize)
    #ax.set_ylabel('y (mm)', fontsize=fontsize)
    
    # 设置 x轴和 y轴的刻度数量
    xticks = np.linspace(xmin, xmax, 6)
    yticks = np.linspace(ymin, ymax, 6)
    ax.xaxis.set_major_locator(FixedLocator(xticks))  # 设置x轴刻度位置
    ax.yaxis.set_major_locator(FixedLocator(yticks))  # 设置y轴刻度位置
    
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize, direction='in')  # 设置刻度线朝内
    ax.axhline(y=1, color='red', linestyle='--', linewidth=1.5)  # 添加水平线 y=1，颜色为红色，虚线类型

# 绘制第一个子图
plot_contour(ax[0], Ux, '', -0.022515, 0.045339, fontsize=17, tick_fontsize=15.5, cbar_tick_fontsize=15.5)
# 绘制第二个子图
plot_contour(ax[1], Uy, '', -0.188, 0.0, fontsize=17, tick_fontsize=15.5, cbar_tick_fontsize=15.5)

save_path = 'C:/Users/yanpe/Desktop/PIKAN/论文/PIKAN/图片/two_beam/pdf/' #定义保存图像的路径
plt.savefig(save_path + 'beam_line_PIKAN-uxuy.pdf', format='pdf', bbox_inches='tight', dpi=800) #bbox_inches='tight'自动调整保存的边界范围，确保所有内容都能完整显示。

plt.show()


#设置绘图时使用的字体为 Times New Roman，并将全局字体大小设置为 12。这会影响图表中的标题、坐标轴标签、刻度标签等的字体样式。
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12

xmin, xmax = np.min(testpoints[:, 0]), np.max(testpoints[:, 0])  # 根据数据动态调整x轴范围
ymin, ymax = np.min(testpoints[:, 1]), np.max(testpoints[:, 1])  # 根据数据动态调整y轴范围

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(19, 1.8)) #创建一个包含两个子图的图形，排列为 1 行 2 列，整个图形的大小为宽度 12 英寸、高度 4 英寸。
fig.subplots_adjust(hspace=0.6, wspace=0.17) #调整子图之间的间距，hspace 控制垂直方向的间距，wspace 控制水平方向的间距。
#fig.suptitle("Elasticity_bend_Two") #在整个图形的顶部添加一个总标题

# 定义绘制等值图函数
def plot_contour(ax, data, title, vmin, vmax, fontsize=17, tick_fontsize=14, cbar_tick_fontsize=14):
    # fontsize：标题和坐标轴标签的字体大小，默认为 20。tick_fontsize：坐标轴刻度的字体大小，默认为 14。cbar_tick_fontsize：颜色条刻度的字体大小，默认为 12。
    cf = ax.scatter(testpoints[:, 0], testpoints[:, 1], s=5, c=data, cmap=cm.jet, vmin=vmin, vmax=vmax, rasterized=True)
    ax.axis('equal')
    cbar = plt.colorbar(cf, ax=ax, format=ScalarFormatter(useMathText=True), pad=0.03)  # 使用ScalarFormatter
    cbar.ax.tick_params(labelsize=cbar_tick_fontsize)
    
    # 设置颜色条的刻度范围，确保显示最大值、最小值和中间刻度
    ticks = LinearLocator(numticks=6)  # 指定 10个刻度
    cbar.set_ticks(ticks.tick_values(vmin, vmax))  # 设置刻度位置
    cbar.set_ticklabels([f"{tick:.2e}" for tick in ticks.tick_values(vmin, vmax)])  # 使用科学计数法格式化刻度标签

    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    ax.set_title(title, fontsize=fontsize)
    #ax.set_xlabel('x (mm)', fontsize=fontsize)
    #ax.set_ylabel('y (mm)', fontsize=fontsize)
    
    # 设置 x轴和 y轴的刻度数量
    xticks = np.linspace(xmin, xmax, 6)
    yticks = np.linspace(ymin, ymax, 6)
    ax.xaxis.set_major_locator(FixedLocator(xticks))  # 设置x轴刻度位置
    ax.yaxis.set_major_locator(FixedLocator(yticks))  # 设置y轴刻度位置
    
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize, direction='in')  # 设置刻度线朝内
    ax.axhline(y=1, color='red', linestyle='--', linewidth=1.5)  # 添加水平线 y=1，颜色为红色，虚线类型

plot_contour(ax[0], Umag, '', 0.0, 0.19287, fontsize=17, tick_fontsize=15.5, cbar_tick_fontsize=15.5)

# 隐藏最后一个子图
ax[1].axis('off')  # 关闭坐标轴
ax[1].set_visible(False)  # 设置子图不可见

save_path = 'C:/Users/MI/Desktop/图片/' #定义保存图像的路径
plt.savefig(save_path + 'beam_line_PIKAN-umag.pdf', format='pdf', bbox_inches='tight', dpi=1200) #bbox_inches='tight'自动调整保存的边界范围，确保所有内容都能完整显示。

plt.show()

#FEM，通过 abaqus计算
import pandas as pd
# 读取 Excel 文件
file_path1 = 'E:/ML/KINN/Numericalexample/heterogeneous_beam/line/onekan/abaqus/coord.xlsx'  #文件路径
df1 = pd.read_excel(file_path1, header=None)  # header=None 表示文件中没有列名
# 将 DataFrame 转换为 NumPy 数组
coord = df1.to_numpy()

file_path2 = 'E:/ML/KINN/Numericalexample/heterogeneous_beam/line/onekan/abaqus/u1all.xlsx'
df2 = pd.read_excel(file_path2, header=None)
u1all = df2.to_numpy()

file_path3 = 'E:/ML/KINN/Numericalexample/heterogeneous_beam/line/onekan/abaqus/u2all.xlsx'
df3 = pd.read_excel(file_path3, header=None)
u2all = df3.to_numpy()

file_path4 = 'E:/ML/KINN/Numericalexample/heterogeneous_beam/line/onekan/abaqus/uall.xlsx'
df4 = pd.read_excel(file_path4, header=None)
uall = df4.to_numpy()

#设置绘图时使用的字体为 Times New Roman，并将全局字体大小设置为 12。这会影响图表中的标题、坐标轴标签、刻度标签等的字体样式。
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12

xmin, xmax = np.min(coord[:, 0]), np.max(coord[:, 0])  # 根据数据动态调整x轴范围
ymin, ymax = np.min(coord[:, 1]), np.max(coord[:, 1])  # 根据数据动态调整y轴范围

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(19, 1.8)) #创建一个包含两个子图的图形，排列为 1 行 2 列，整个图形的大小为宽度 12 英寸、高度 4 英寸。
fig.subplots_adjust(hspace=0.6, wspace=0.17) #调整子图之间的间距，hspace 控制垂直方向的间距，wspace 控制水平方向的间距。
#fig.suptitle("Elasticity_bend_Two") #在整个图形的顶部添加一个总标题

# 定义绘制等值图函数
def plot_contour(ax, data, title, vmin, vmax, fontsize=17, tick_fontsize=14, cbar_tick_fontsize=14):
    # fontsize：标题和坐标轴标签的字体大小，默认为 20。tick_fontsize：坐标轴刻度的字体大小，默认为 14。cbar_tick_fontsize：颜色条刻度的字体大小，默认为 12。
    cf = ax.scatter(coord[:, 0], coord[:, 1], s=5, c=data, cmap=cm.jet, vmin=vmin, vmax=vmax, rasterized=True)
    ax.axis('equal')
    cbar = plt.colorbar(cf, ax=ax, format=ScalarFormatter(useMathText=True), pad=0.03)  # 使用ScalarFormatter
    cbar.ax.tick_params(labelsize=cbar_tick_fontsize)
    
    # 设置颜色条的刻度范围，确保显示最大值、最小值和中间刻度
    ticks = LinearLocator(numticks=6)  # 指定 10个刻度
    cbar.set_ticks(ticks.tick_values(vmin, vmax))  # 设置刻度位置
    cbar.set_ticklabels([f"{tick:.2e}" for tick in ticks.tick_values(vmin, vmax)])  # 使用科学计数法格式化刻度标签

    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    ax.set_title(title, fontsize=fontsize)
    #ax.set_xlabel('x (mm)', fontsize=fontsize)
    #ax.set_ylabel('y (mm)', fontsize=fontsize)
    
    # 设置 x轴和 y轴的刻度数量
    xticks = np.linspace(xmin, xmax, 6)
    yticks = np.linspace(ymin, ymax, 6)
    ax.xaxis.set_major_locator(FixedLocator(xticks))  # 设置x轴刻度位置
    ax.yaxis.set_major_locator(FixedLocator(yticks))  # 设置y轴刻度位置
    
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize, direction='in')  # 设置刻度线朝内
    ax.axhline(y=1, color='red', linestyle='--', linewidth=1.5)

# 绘制第一个子图
plot_contour(ax[0], u1all, '', -0.0226, 0.0453, fontsize=17, tick_fontsize=15.5, cbar_tick_fontsize=15.5)
# 绘制第二个子图
plot_contour(ax[1], u2all, '', -0.188, 0.0, fontsize=17, tick_fontsize=15.5, cbar_tick_fontsize=15.5)

save_path = 'C:/Users/yanpe/Desktop/PIKAN/论文/PIKAN/图片/two_beam/pdf/' #定义保存图像的路径
plt.savefig(save_path + 'beam_line_abaqus_uxuy.pdf', format='pdf', bbox_inches='tight', dpi=800) #bbox_inches='tight'自动调整保存的边界范围，确保所有内容都能完整显示。

plt.show()


#设置绘图时使用的字体为 Times New Roman，并将全局字体大小设置为 12。这会影响图表中的标题、坐标轴标签、刻度标签等的字体样式。
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12

xmin, xmax = np.min(coord[:, 0]), np.max(coord[:, 0])  # 根据数据动态调整x轴范围
ymin, ymax = np.min(coord[:, 1]), np.max(coord[:, 1])  # 根据数据动态调整y轴范围

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(19, 1.8)) #创建一个包含两个子图的图形，排列为 1 行 2 列，整个图形的大小为宽度 12 英寸、高度 4 英寸。
fig.subplots_adjust(hspace=0.6, wspace=0.17) #调整子图之间的间距，hspace 控制垂直方向的间距，wspace 控制水平方向的间距。
#fig.suptitle("Elasticity_bend_Two") #在整个图形的顶部添加一个总标题

# 定义绘制等值图函数
def plot_contour(ax, data, title, vmin, vmax, fontsize=17, tick_fontsize=14, cbar_tick_fontsize=14):
    # fontsize：标题和坐标轴标签的字体大小，默认为 20。tick_fontsize：坐标轴刻度的字体大小，默认为 14。cbar_tick_fontsize：颜色条刻度的字体大小，默认为 12。
    cf = ax.scatter(coord[:, 0], coord[:, 1], s=5, c=data, cmap=cm.jet, vmin=vmin, vmax=vmax, rasterized=True)
    ax.axis('equal')
    cbar = plt.colorbar(cf, ax=ax, format=ScalarFormatter(useMathText=True), pad=0.03)  # 使用ScalarFormatter
    cbar.ax.tick_params(labelsize=cbar_tick_fontsize)
    
    # 设置颜色条的刻度范围，确保显示最大值、最小值和中间刻度
    ticks = LinearLocator(numticks=6)  # 指定 10个刻度
    cbar.set_ticks(ticks.tick_values(vmin, vmax))  # 设置刻度位置
    cbar.set_ticklabels([f"{tick:.2e}" for tick in ticks.tick_values(vmin, vmax)])  # 使用科学计数法格式化刻度标签

    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    ax.set_title(title, fontsize=fontsize)
    #ax.set_xlabel('x (mm)', fontsize=fontsize)
    #ax.set_ylabel('y (mm)', fontsize=fontsize)
    
    # 设置 x轴和 y轴的刻度数量
    xticks = np.linspace(xmin, xmax, 6)
    yticks = np.linspace(ymin, ymax, 6)
    ax.xaxis.set_major_locator(FixedLocator(xticks))  # 设置x轴刻度位置
    ax.yaxis.set_major_locator(FixedLocator(yticks))  # 设置y轴刻度位置
    
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize, direction='in')  # 设置刻度线朝内
    ax.axhline(y=1, color='red', linestyle='--', linewidth=1.5)

plot_contour(ax[0], uall, '', 0.0, 0.193, fontsize=17, tick_fontsize=15.5, cbar_tick_fontsize=15.5)

# 隐藏最后一个子图
ax[1].axis('off')  # 关闭坐标轴
ax[1].set_visible(False)  # 设置子图不可见

save_path = 'C:/Users/MI/Desktop/图片/' #定义保存图像的路径
plt.savefig(save_path + 'beam_line_abaqus_umag.pdf', format='pdf', bbox_inches='tight', dpi=1200) #bbox_inches='tight'自动调整保存的边界范围，确保所有内容都能完整显示。

plt.show()

import pandas as pd
# 读取 Excel 文件 读取 fem节点坐标
file_path_coord = 'D:/ML/KINN/Numerical_example/heterogeneous_beam/line/onekan/result/absoluteerror/coord.xlsx'  #文件路径
df_coord = pd.read_excel(file_path_coord, header=None)  # header=None 表示文件中没有列名
# 将 DataFrame 转换为 NumPy 数组
femcoord = df_coord.to_numpy()

femcoord = torch.from_numpy(femcoord).float()
femcoord = femcoord.to(device='cuda')
predforerror = testpred(femcoord)
Uforerror = predforerror.detach().cpu().numpy()
Uxforerror = Uforerror[:, 0].copy() # 整个计算域配点x方向位移
Uyforerror = Uforerror[:, 1].copy()
save_path = "D:/ML/KINN/Numerical_example/heterogeneous_beam/line/onekan/result/absoluteerror/"
np.savetxt(save_path + "Uxforerror.csv", Uxforerror, delimiter=',', fmt='%f')
np.savetxt(save_path + "Uyforerror.csv", Uyforerror, delimiter=',', fmt='%f')

# 与 abaqus_fem解计算完绝对误差后，读取 Excel 文件，画绝对误差图
file_path_coord = 'D:/ML/KINN/Numerical_example/heterogeneous_beam/line/onekan/result/absoluteerror/coord.xlsx'  #文件路径
df_coord = pd.read_excel(file_path_coord, header=None)  # header=None 表示文件中没有列名
# 将 DataFrame 转换为 NumPy 数组
femcoord = df_coord.to_numpy()

file_path_ux = 'D:/ML/KINN/Numerical_example/heterogeneous_beam/line/onekan/result/absoluteerror/absoluteerror_ux.xlsx'
df_ux = pd.read_excel(file_path_ux, header=None)
absoluteerror_ux = df_ux.to_numpy()

file_path_uy = 'D:/ML/KINN/Numerical_example/heterogeneous_beam/line/onekan/result/absoluteerror/absoluteerror_uy.xlsx'
df_uy = pd.read_excel(file_path_uy, header=None)
absoluteerror_uy = df_uy.to_numpy()

from matplotlib import cm #cm：matplotlib 的颜色映射模块，用于定义颜色映射表
from matplotlib.ticker import LinearLocator, FixedLocator, ScalarFormatter
#LinearLocator 和 FixedLocator：用于自定义坐标轴刻度的位置。ScalarFormatter：用于格式化颜色条的刻度标签，支持数学符号显示。
from matplotlib.font_manager import FontProperties

#设置绘图时使用的字体为 Times New Roman，并将全局字体大小设置为 12。这会影响图表中的标题、坐标轴标签、刻度标签等的字体样式。
plt.rcParams['font.family'] = 'Times New Roman'

xmin, xmax = np.min(femcoord[:, 0]), np.max(femcoord[:, 0])  # 根据数据动态调整x轴范围
ymin, ymax = np.min(femcoord[:, 1]), np.max(femcoord[:, 1])  # 根据数据动态调整y轴范围

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(19, 1.8)) #创建一个包含两个子图的图形，排列为 1 行 2 列，整个图形的大小为宽度 12 英寸、高度 4 英寸。
fig.subplots_adjust(hspace=0.6, wspace=0.17) #调整子图之间的间距，hspace 控制垂直方向的间距，wspace 控制水平方向的间距。
#fig.suptitle("Elasticity_bend_Two") #在整个图形的顶部添加一个总标题

# 定义绘制等值图函数
def plot_contour(ax, data, title, vmin, vmax, fontsize=17, tick_fontsize=14, cbar_tick_fontsize=14):
    # fontsize：标题和坐标轴标签的字体大小，默认为 20。tick_fontsize：坐标轴刻度的字体大小，默认为 14。cbar_tick_fontsize：颜色条刻度的字体大小，默认为 12。
    cf = ax.scatter(femcoord[:, 0], femcoord[:, 1], s=5, c=data, cmap=cm.jet, vmin=vmin, vmax=vmax)
    ax.axis('equal')
    cbar = plt.colorbar(cf, ax=ax, format=ScalarFormatter(useMathText=True), pad=0.03)  # 使用ScalarFormatter
    cbar.ax.tick_params(labelsize=cbar_tick_fontsize)
    
    # 设置颜色条的刻度范围，确保显示最大值、最小值和中间刻度
    ticks = LinearLocator(numticks=6)  # 指定 10个刻度
    cbar.set_ticks(ticks.tick_values(vmin, vmax))  # 设置刻度位置
    cbar.set_ticklabels([f"{tick:.2e}" for tick in ticks.tick_values(vmin, vmax)])  # 使用科学计数法格式化刻度标签

    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    ax.set_title(title, fontsize=fontsize)
    #ax.set_xlabel(r'$\mathit{x}$ (mm)', fontsize=fontsize)  # x 设置为斜体
    #ax.set_ylabel(r'$\mathit{y}$ (mm)', fontsize=fontsize)  # y 设置为斜体
    
    # 设置 x轴和 y轴的刻度数量
    xticks = np.linspace(xmin, xmax, 6)
    yticks = np.linspace(ymin, ymax, 6)
    ax.xaxis.set_major_locator(FixedLocator(xticks))  # 设置x轴刻度位置
    ax.yaxis.set_major_locator(FixedLocator(yticks))  # 设置y轴刻度位置
    
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize, direction='in')  # 设置刻度线朝内
    ax.axhline(y=1, color='red', linestyle='--', linewidth=1.5)

# 绘制第一个子图
plot_contour(ax[0], absoluteerror_ux, '', 0.0, 0.0004154, fontsize=17, tick_fontsize=15.5, cbar_tick_fontsize=15.5)
# 绘制第二个子图
plot_contour(ax[1], absoluteerror_uy, '', 0.0, 0.00047536, fontsize=17, tick_fontsize=15.5, cbar_tick_fontsize=15.5)

save_path = 'D:/ML/KINN/Numerical_example/heterogeneous_beam/line/onekan/result/absoluteerror/' #定义保存图像的路径
plt.savefig(save_path + 'absoluteerror_line.pdf', format='pdf', bbox_inches='tight', dpi=800)  # 保存为 PDF 格式 bbox_inches='tight'自动调整保存的边界范围，确保所有内容都能完整显示。

plt.show()


# loss_array、loss1_array 和 loss2_array 是包含 torch.Tensor 的列表
# 将 tensor 转换为纯数值
loss_array = [item.item() if isinstance(item, torch.Tensor) else item for item in loss_array]
loss1_array = [item.item() if isinstance(item, torch.Tensor) else item for item in loss1_array]
loss2_array = [item.item() if isinstance(item, torch.Tensor) else item for item in loss2_array]
loss_internal_array = [item.item() if isinstance(item, torch.Tensor) else item for item in loss_internal_array]
loss_external_array = [item.item() if isinstance(item, torch.Tensor) else item for item in loss_external_array]

# 创建一个 DataFrame 来存储这些数据
data = {
    "迭代次数": list(range(len(loss_array))),  # 假设所有数组长度相同
    "总模型势能": loss_array,
    "材料 1 应变能": loss1_array,
    "材料 2 应变能": loss2_array,
    "总模型应变能": loss_internal_array,
    "总模型外力功": loss_external_array
}

df = pd.DataFrame(data)

# 保存为 Excel 文件
excel_file = "D:/ML/KINN/Numericalexample/heterogeneous_beam/line/onekan/triangle/gird10-order3/Losshistory.xlsx"
df.to_excel(excel_file, index=False, engine="openpyxl")

# plot the prediction solution
fig = plt.figure(figsize=(20, 20))

plt.subplot(2, 2, 1)
#plt.yscale('log') #将y轴设置为对数刻度（log scale）。    无法使用log，因为损失是负数
plt.grid(axis='y')
plt.plot(loss1_array, ls = '--')
plt.legend(['loss1'], loc = 'upper right')
plt.xlabel('the iteration') #设置x轴的标签为 “the iteration”，表示横轴表示的是迭代次数
plt.ylabel('loss') #设置y轴的标签为 “loss”，表示纵轴表示的是损失值。
plt.title('loss1', fontsize = 20)

plt.subplot(2, 2, 2)
#plt.yscale('log')
plt.grid(axis='y') #在y轴方向添加网格线，便于观察y轴的变化。
plt.plot(loss2_array, ls = '--')
plt.legend(['loss2'], loc = 'upper right')
plt.xlabel('the iteration') #设置x轴标签为 “the iteration”，表示横轴表示的是迭代次数。
plt.ylabel('loss') #设置y轴标签为 “loss”，表示纵轴表示的是损失值。
plt.title('loss2', fontsize = 20) #设置子图的标题为“cenn external”，表示这是关于外域能量损失的图表。参数 fontsize=20：标题的字体大小为20。

plt.subplot(2, 2, 3)
#plt.yscale('log')
plt.grid(axis='y') #在y轴方向添加网格线，便于观察y轴的变化。
plt.plot(loss_array, ls = '--')
plt.legend(['loss'], loc = 'upper right')
plt.xlabel('the iteration') #设置x轴标签为 “the iteration”，表示横轴表示的是迭代次数。
plt.ylabel('loss') #设置y轴标签为 “loss”，表示纵轴表示的是损失值。
plt.title('loss', fontsize = 20) #设置子图的标题为“cenn external”，表示这是关于外域能量损失的图表。参数 fontsize=20：标题的字体大小为20。


# --------------------------------------------------------------------------------
# 主要用于计算 VonMises应力
# --------------------------------------------------------------------------------
def evaluate_model(datatest, E, nu):
    dim = 2
    if dim == 2:
        Nx = len(datatest) #测试集配点个数
        x = datatest[:, 0].reshape(Nx, 1) #得到测试集的 x坐标，形状为[Nx,1]
        y = datatest[:, 1].reshape(Nx, 1) #得到测试集的 y坐标，形状为[Nx,1]
            
        xy = np.concatenate((x, y), axis=1) #重新组合得到测试集配点坐标，传到 cuda上
        xy_tensor = torch.from_numpy(xy).float()
        xy_tensor = xy_tensor.to(device='cuda')
        xy_tensor.requires_grad_(True)
        
        ust_pred_torch = testpred(xy_tensor) #预测位移场
        duxdxy = grad(ust_pred_torch[:, 0].unsqueeze(1), xy_tensor, torch.ones(xy_tensor.size()[0], 1, device='cuda'),
                        create_graph=True, retain_graph=True)[0] #duxdxy=(∂ux/∂x,∂ux/∂y)
        duydxy = grad(ust_pred_torch[:, 1].unsqueeze(1), xy_tensor, torch.ones(xy_tensor.size()[0], 1, device='cuda'),
                        create_graph=True, retain_graph=True)[0] #duydxy=(∂uy/∂x,∂uy/∂y)
        
        #计算应变
        dudx = duxdxy[:, 0] #∂ux/∂x
        dudy = duxdxy[:, 1] #∂ux/∂y
        dvdx = duydxy[:, 0] #∂uy/∂x
        dvdy = duydxy[:, 1] #∂uy/∂y
        exx_pred = dudx #εx
        eyy_pred = dvdy #‌εy
        e2xy_pred = dudy + dvdx #‌εxy
        #计算应力
        D11_mat = E/(1-nu**2)
        D22_mat = E/(1-nu**2)
        D12_mat = E*nu/(1-nu**2)
        D21_mat = E*nu/(1-nu**2)
        D33_mat = E/(2*(1+nu))
        sxx_pred = D11_mat * exx_pred + D12_mat * eyy_pred #σx
        syy_pred = D12_mat * exx_pred + D22_mat * eyy_pred #σy
        sxy_pred = D33_mat * e2xy_pred #τxy
        
        #将 PyTorch 张量转换为 NumPy 数组
        ust_pred = ust_pred_torch.detach().cpu().numpy()
        
        exx_pred = exx_pred.detach().cpu().numpy()
        eyy_pred = eyy_pred.detach().cpu().numpy()
        e2xy_pred = e2xy_pred.detach().cpu().numpy()
        
        sxx_pred = sxx_pred.detach().cpu().numpy()
        syy_pred = syy_pred.detach().cpu().numpy()
        sxy_pred = sxy_pred.detach().cpu().numpy()
        ust_pred = ust_pred_torch.detach().cpu().numpy()
        
        #提取位移分量
        surUx = ust_pred[:, 0]
        surUy = ust_pred[:, 1]
        surUz = np.zeros(Nx)
        #提取应变分量
        surE11 = exx_pred
        surE12 = 0.5*e2xy_pred
        surE13 = np.zeros(Nx)
        surE21 = 0.5*e2xy_pred
        surE22 = eyy_pred
        surE23 = np.zeros(Nx)
        surE33 = np.zeros(Nx)
        #提取应力分量   
        surS11 = sxx_pred
        surS12 = sxy_pred
        surS13 = np.zeros(Nx)
        surS21 = sxy_pred
        surS22 = syy_pred
        surS23 = np.zeros(Nx)
        surS33 = np.zeros(Nx)
        
        #计算 VonMises应力
        SVonMises = np.float64(np.sqrt(0.5 * ((surS11 - surS22) ** 2 + (surS22) ** 2 + (-surS11) ** 2 + 6 * (surS12 ** 2))))
        
        U = (np.float64(surUx), np.float64(surUy), np.float64(surUz))
        U_mag = (np.float64(surUx)**2 + np.float64(surUy)**2 + np.float64(surUz)**2)**(0.5) #绝对位移
        
        return U, np.float64(U_mag), np.float64(SVonMises)
    
# 筛选指定 x坐标的点
tolerance = 1e-6
x_points = testpoints[np.abs(testpoints[:, 0] - 2) < tolerance] #x=2
# 分为二部分
x1 = x_points[x_points[:, 1] >= 1]
x2 = x_points[x_points[:, 1] <= 1]
np.savetxt("x1.csv", x1, delimiter=',', fmt='%f')
np.savetxt("x2.csv", x2, delimiter=',', fmt='%f')

_, _, x_SVonMises1 = evaluate_model(x1, E1, nu1)
_, _, x_SVonMises2 = evaluate_model(x2, E2, nu2)

np.savetxt("x_points_x2.csv", x_points, delimiter=',', fmt='%f')
np.savetxt("x_SVonMises1_x2.csv", x_SVonMises1, delimiter=',', fmt='%f')
np.savetxt("x_SVonMises2_x2.csv", x_SVonMises2, delimiter=',', fmt='%f')
