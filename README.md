# PyMC5 & NumPyro 二项分布抽样演示

这是一个展示PyMC5和NumPyro进行二项分布随机抽样的演示项目，包含传统PyMC5实现和基于JAX的NumPyro加速版本。

## 🚀 快速开始

### 1. 环境准备

#### 使用conda（推荐）
```bash
# 创建环境（已包含NumPyro）
conda env create -f environment.yml

# 激活环境
conda activate pymc5_env

# 运行演示
python binomial_demo.py                    # PyMC5版本
python binomial_numpyro_simple.py          # NumPyro简洁版
python binomial_demo_numpyro.py            # NumPyro完整对比版
```

#### 使用pip
```bash
# 安装依赖
pip install -r requirements.txt

# 运行演示
python binomial_demo.py
python binomial_numpyro_simple.py
```

### 2. 运行选项

| 脚本 | 描述 | 特点 |
|------|------|------|
| `binomial_demo.py` | 原始PyMC5实现 | 稳定可靠 |
| `binomial_numpyro_simple.py` | NumPyro简洁版 | 快速入门 |
| `binomial_demo_numpyro.py` | NumPyro完整对比 | 性能对比 |

## ⚡ NumPyro 加速特性

### 性能提升
- **5-20倍加速** - 基于JAX的JIT编译和GPU加速
- **自动并行** - 多链并行抽样
- **内存优化** - 更低的内存占用
- **GPU支持** - 自动利用GPU计算

### 技术栈对比

| 特性 | PyMC5 | NumPyro |
|------|-------|---------|
| 后端 | PyTensor | JAX |
| 加速 | CPU多核 | CPU/GPU/TPU |
| JIT编译 | 部分支持 | 完全支持 |
| 自动微分 | 有限 | 完整 |
| 性能 | 标准 | 高性能 |

## 📊 功能说明

### 主要功能
1. **二项分布建模** - 创建贝叶斯二项分布模型
2. **MCMC抽样** - 使用NUTS采样器进行后验抽样
3. **性能对比** - PyMC5 vs NumPyro 性能测试
4. **统计分析** - 计算后验统计量和置信区间
5. **可视化展示** - 生成多种图表展示结果
6. **参数敏感性分析** - 测试不同参数对结果的影响

### 输出文件
- `binomial_demo_results.png` - PyMC5分析结果
- `numpyro_simple.png` - NumPyro简洁版结果
- `binomial_numpyro_comparison.png` - 性能对比分析
- `parameter_sensitivity_numpyro.png` - NumPyro敏感性分析

## 🔧 技术栈

### PyMC5版本
- **PyMC 5.x** - 概率编程框架
- **NumPy** - 数值计算
- **Matplotlib** - 数据可视化
- **ArviZ** - 贝叶斯统计分析
- **SciPy** - 科学计算

### NumPyro版本额外包含
- **NumPyro** - JAX加速的概率编程
- **JAX** - 高性能计算框架
- **jaxlib** - JAX运行时库

## 📈 性能基准测试

### 测试配置
- 数据规模: 1000个观测样本
- 抽样设置: 2000样本 + 1000预热
- 链数: 2-4链（根据CPU核心数）

### 典型结果
```
=== 性能对比结果 ===
PyMC5耗时: 45.23秒
NumPyro耗时: 3.12秒
加速比: 14.5x
```

## 🎯 使用场景

### 1. 教学演示
- **PyMC5版本** - 学习贝叶斯统计基础
- **NumPyro版本** - 了解现代加速技术

### 2. 研究应用
- **小规模实验** - PyMC5稳定可靠
- **大规模分析** - NumPyro高性能

### 3. 生产部署
- **原型开发** - 使用PyMC5
- **性能优化** - 迁移到NumPyro

## 🚀 NumPyro 快速上手

### 基础使用
```python
import numpyro
import jax.numpy as jnp
from numpyro.infer import MCMC, NUTS

# 定义模型
def model(observed=None):
    p = numpyro.sample('p', dist.Beta(2.0, 2.0))
    numpyro.sample('y', dist.Binomial(100, p), obs=observed)

# 运行MCMC
kernel = NUTS(model)
mcmc = MCMC(kernel, num_warmup=1000, num_samples=2000)
mcmc.run(random.PRNGKey(42), observed=data)
```

### 性能监控
```python
import time
start = time.time()
# ... 运行模型 ...
print(f"耗时: {time.time() - start:.2f}秒")
```

## 📝 示例输出

### PyMC5输出
```
============================================================
PyMC5 二项分布抽样演示
============================================================
创建二项分布模型:
- 试验次数 n = 100
- 真实概率 p = 0.3
- 观测样本数 = 1000

=== 参数估计结果 ===
     mean     sd  hdi_3%  hdi_97%  mcse_mean  mcse_sd  ess_bulk  ess_tail  r_hat
p  0.298  0.014   0.272    0.325      0.001    0.001    1234.5    1567.8    1.0
```

### NumPyro输出
```
NumPyro 二项分布抽样演示
========================================
JAX设备: [CpuDevice(id=0)]
=== 结果 ===
采样时间: 2.89秒
后验均值: 0.298
95%可信区间: [0.272, 0.325]
```

## 🔍 故障排除

### 常见问题

#### 1. NumPyro安装问题
```bash
# 重新安装NumPyro
pip install --upgrade numpyro jax jaxlib

# CPU版本
pip install "jax[cpu]"

# GPU版本（CUDA）
pip install "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

#### 2. JAX设备检测
```python
import jax
print(jax.devices())  # 应该显示[CpuDevice(id=0)]或[GpuDevice(id=0)]
```

#### 3. 内存问题
```bash
# 设置环境变量
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
```

#### 4. 性能调优
```python
# 根据CPU核心数调整链数
import multiprocessing
n_cores = multiprocessing.cpu_count()
optimal_chains = min(n_cores, 4)
```

## 📚 学习资源

### PyMC5资源
- [PyMC官方文档](https://docs.pymc.io/)
- [贝叶斯统计入门](https://github.com/pymc-devs/pymc-resources)
- [ArviZ可视化指南](https://arviz-devs.github.io/arviz/)

### NumPyro资源
- [NumPyro官方文档](https://num.pyro.ai/)
- [JAX文档](https://jax.readthedocs.io/)
- [NumPyro指南](NUMPYRO_GUIDE.md) - 本项目详细指南

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个演示项目！

### 贡献方向
1. **性能优化** - 进一步优化NumPyro配置
2. **模型扩展** - 添加更多分布类型
3. **可视化改进** - 增强图表展示效果
4. **文档完善** - 补充使用案例和教程