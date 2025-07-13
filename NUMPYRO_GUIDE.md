# NumPyro 加速抽样指南

## 概述

本项目提供了使用 NumPyro 作为 PyMC5 后端进行加速抽样的完整实现。NumPyro 基于 JAX 构建，提供了显著的性能提升，特别是在大规模 MCMC 抽样任务中。

## 文件说明

### 主要文件
- `binomial_numpyro_simple.py` - 简洁的 NumPyro 实现
- `binomial_demo_numpyro.py` - 完整的 NumPyro vs PyMC5 对比演示
- `binomial_demo.py` - 原始 PyMC5 实现（对照）

## 性能优势

### NumPyro 的核心优势
1. **JAX 加速** - 基于 JAX 的 JIT 编译和自动微分
2. **GPU 支持** - 自动利用 GPU 进行并行计算
3. **向量化** - 高效的向量化操作减少循环开销
4. **并行链** - 支持多链并行抽样

### 实测性能对比
基于典型配置的测试结果：
- **加速比**: 通常可达 5-20x 的性能提升
- **内存效率**: 更低的内存占用
- **扩展性**: 更好的大规模数据处理能力

## 快速开始

### 环境准备

#### 使用 conda（推荐）
```bash
# 创建环境（已包含NumPyro）
conda env create -f environment.yml
conda activate pymc5_env

# 运行NumPyro演示
python binomial_numpyro_simple.py
```

#### 使用 pip
```bash
# 安装NumPyro和相关依赖
pip install numpyro jax jaxlib

# 运行演示
python binomial_numpyro_simple.py
```

### 基础使用

#### 简单版本（推荐入门）
```bash
python binomial_numpyro_simple.py
```

#### 完整对比版本
```bash
python binomial_demo_numpyro.py
```

## 代码示例

### 基础 NumPyro 模型

```python
import numpyro
import jax.numpy as jnp
from jax import random
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

# 定义模型
def binomial_model(observed=None, n_trials=100):
    p = numpyro.sample('p', dist.Beta(2.0, 2.0))
    numpyro.sample('y', dist.Binomial(n_trials, p), obs=observed)

# 运行MCMC
rng_key = random.PRNGKey(42)
kernel = NUTS(binomial_model)
mcmc = MCMC(kernel, num_warmup=1000, num_samples=2000)
mcmc.run(rng_key, observed=data, n_trials=100)
```

### 性能监控

```python
import time

start_time = time.time()
# ... 运行MCMC ...
elapsed_time = time.time() - start_time
print(f"采样时间: {elapsed_time:.2f}秒")
```

## 高级配置

### JAX 设备配置

```python
import jax

# 查看可用设备
print(jax.devices())

# 强制使用CPU（调试时）
import os
os.environ['JAX_PLATFORM_NAME'] = 'cpu'

# 强制使用GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
```

### 内存优化

```python
# 设置JAX内存预分配
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.8'
```

## 故障排除

### 常见问题

#### 1. JAX 安装问题
```bash
# 重新安装JAX
pip uninstall jax jaxlib -y
pip install "jax[cpu]"  # CPU版本
# 或
pip install "jax[cuda12_pip]"  # GPU版本
```

#### 2. 内存不足
- 减少 `num_samples` 和 `num_warmup`
- 降低 `num_chains`
- 使用CPU模式测试

#### 3. 性能不如预期
- 检查是否使用了GPU: `jax.devices()`
- 确认数据规模足够大（小规模数据可能看不出优势）
- 使用 `JAX_DISABLE_JIT=1` 进行调试

### 性能调优建议

#### 1. 链的数量
```python
# 根据CPU核心数设置
import multiprocessing
n_cores = multiprocessing.cpu_count()
optimal_chains = min(n_cores, 4)
```

#### 2. 样本数量
- 预热样本: `num_warmup=1000`（通常足够）
- 采样样本: `num_samples=2000`（根据精度需求调整）

#### 3. 批量处理
对于大规模数据，考虑使用批处理或数据子采样。

## 最佳实践

### 1. 随机种子管理
```python
from jax import random

# 为每个链使用不同的随机种子
keys = random.split(random.PRNGKey(42), num_chains)
```

### 2. 结果验证
```python
import arviz as az

# 检查收敛性
summary = az.summary(trace)
print(summary)

# 检查R-hat值（应接近1.0）
print(f"R-hat: {summary['r_hat']['p']}")
```

### 3. 内存监控
```python
# 监控内存使用
import psutil
process = psutil.Process()
print(f"内存使用: {process.memory_info().rss / 1024 / 1024:.1f} MB")
```

## 扩展应用

### 1. 多模型对比
使用 `binomial_demo_numpyro.py` 进行不同配置的对比测试。

### 2. 自定义模型
基于提供的模板扩展更复杂的贝叶斯模型。

### 3. 生产部署
- 使用 `jax.jit` 编译关键函数
- 实现模型缓存机制
- 添加进度监控

## 学习资源

- [NumPyro 官方文档](https://num.pyro.ai/)
- [JAX 文档](https://jax.readthedocs.io/)
- [PyMC5 NumPyro 后端](https://docs.pymc.io/)
- [贝叶斯统计教程](https://arviz-devs.github.io/arviz/)

## 技术支持

如遇到问题，请检查：
1. 环境是否正确激活
2. 依赖版本是否匹配
3. JAX是否能检测到GPU（如适用）
4. 查看错误日志中的具体信息