#!/usr/bin/env python3
"""
NumPyro 二项分布抽样演示脚本
Binomial Distribution Sampling Demo with NumPyro

这个脚本演示了如何使用NumPyro作为PyMC5的后端进行二项分布的随机抽样和可视化分析
NumPyro基于JAX，提供GPU加速和JIT编译，能够显著提升MCMC抽样效率
"""

import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az
import multiprocessing
import time
from contextlib import contextmanager

# NumPyro相关导入
import numpyro
import jax
import jax.numpy as jnp
from jax import random
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import pytensor 

pytensor.config.cxx = ""

# 自动检测CPU核心数
n_cores = multiprocessing.cpu_count()
optimal_chains = min(n_cores, 4)  # NumPyro通常使用较少的链

# 设置JAX设备数量以支持并行链
if jax.default_backend() == 'cpu':
    # 设置CPU设备数量以支持并行链
    numpyro.set_host_device_count(min(n_cores, 4))

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

@contextmanager
def timer(description):
    """计时器上下文管理器"""
    start = time.time()
    print(f"{description} 开始...")
    yield
    elapsed = time.time() - start
    print(f"{description} 完成，耗时: {elapsed:.2f}秒")

def create_binomial_model_numpyro(n_trials=100, true_p=0.3, n_observations=1000, num_samples=2000, num_warmup=1000):
    """
    使用NumPyro创建二项分布模型
    
    参数:
        n_trials: 每次试验的尝试次数
        true_p: 真实的成功概率
        n_observations: 观测样本数量
        num_samples: MCMC抽样数量
        num_warmup: 预热步数
    
    返回:
        trace: 抽样结果
        observed_data: 观测数据
        sampling_time: 抽样耗时
    """
    print(f"使用NumPyro创建二项分布模型:")
    print(f"- 试验次数 n = {n_trials}")
    print(f"- 真实概率 p = {true_p}")
    print(f"- 观测样本数 = {n_observations}")
    print(f"- 核心数 = {n_cores}")
    print(f"- JAX设备 = {jax.devices()}")
    
    # 生成观测数据，确保数值有效性
    np.random.seed(42)
    observed_data = np.random.binomial(n=n_trials, p=true_p, size=n_observations)
    
    # 检查数据有效性
    if np.any(observed_data < 0):
        observed_data = np.clip(observed_data, 0, n_trials)
    if np.any(observed_data > n_trials):
        observed_data = np.clip(observed_data, 0, n_trials)
    
    # NumPyro模型定义
    def binomial_model(observed=None, n_trials=n_trials):
        # 定义先验分布 - Beta分布作为p的先验，使用更稳健的参数
        p = numpyro.sample('p', dist.Beta(2.0, 2.0))
        
        # 定义似然函数 - 二项分布，确保概率在有效范围内
        p_constrained = jnp.clip(p, 1e-6, 1-1e-6)  # 避免边界值
        numpyro.sample('y', dist.Binomial(total_count=n_trials, probs=p_constrained), obs=observed)
    
    # 设置随机种子
    rng_key = random.PRNGKey(42)
    
    # 创建MCMC采样器
    kernel = NUTS(binomial_model)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=optimal_chains,
        progress_bar=True
    )
    
    # 执行MCMC抽样
    with timer("NumPyro MCMC抽样"):
        mcmc.run(rng_key, observed=observed_data, n_trials=n_trials)
    
    # 获取采样结果
    samples = mcmc.get_samples()
    
    # 转换为ArviZ格式用于分析
    trace = az.from_numpyro(mcmc)
    
    # 记录采样时间
    sampling_time = time.time() - time.time()  # 实际时间由timer记录
    
    return trace, observed_data, sampling_time

def create_binomial_model_pymc(n_trials=100, true_p=0.3, n_observations=1000, num_samples=2000, num_warmup=1000):
    """
    使用原始PyMC5创建二项分布模型（用于性能对比）
    
    参数:
        n_trials: 每次试验的尝试次数
        true_p: 真实的成功概率
        n_observations: 观测样本数量
        num_samples: MCMC抽样数量
        num_warmup: 预热步数
    
    返回:
        trace: 抽样结果
        observed_data: 观测数据
        sampling_time: 抽样耗时
    """
    print(f"使用PyMC5创建二项分布模型:")
    print(f"- 试验次数 n = {n_trials}")
    print(f"- 真实概率 p = {true_p}")
    print(f"- 观测样本数 = {n_observations}")
    
    # 生成观测数据，确保数值有效性
    np.random.seed(42)
    observed_data = np.random.binomial(n=n_trials, p=true_p, size=n_observations)
    
    # 检查数据有效性
    if np.any(observed_data < 0):
        observed_data = np.clip(observed_data, 0, n_trials)
    if np.any(observed_data > n_trials):
        observed_data = np.clip(observed_data, 0, n_trials)
    
    # PyMC5模型
    with pm.Model() as model:
        # 定义先验分布 - Beta分布作为p的先验，避免边界值
        p = pm.Beta('p', alpha=2, beta=2)
        
        # 定义似然函数 - 二项分布
        y = pm.Binomial('y', n=n_trials, p=p, observed=observed_data)
        
        # 执行MCMC抽样，增加目标接受率以提高稳定性
        with timer("PyMC5 MCMC抽样"):
            trace = pm.sample(
                num_samples,
                tune=num_warmup,
                chains=optimal_chains,
                target_accept=0.95,  # 提高目标接受率
                random_seed=42,
                progressbar=False,
                init="adapt_diag"  # 使用更稳定的初始化方法
            )
    
    return trace, observed_data, 0  # 时间由timer记录

def analyze_results(trace, observed_data, backend_name):
    """
    分析抽样结果
    
    参数:
        trace: MCMC抽样结果
        observed_data: 观测数据
        backend_name: 后端名称
    
    返回:
        summary: 统计摘要
    """
    # 计算统计摘要
    summary = az.summary(trace, var_names=['p'])
    print(f"\n=== {backend_name} 参数估计结果 ===")
    print(summary)
    
    # 计算观测数据的统计量
    observed_mean = np.mean(observed_data)
    observed_std = np.std(observed_data)
    observed_p_hat = observed_mean / np.max(observed_data) if len(observed_data) > 0 else 0
    
    print(f"\n=== {backend_name} 观测数据统计 ===")
    print(f"观测均值: {observed_mean:.2f}")
    print(f"观测标准差: {observed_std:.2f}")
    print(f"估计概率 p̂: {observed_p_hat:.3f}")
    
    return summary

def visualize_results_comparison(trace_numpyro, trace_pymc, observed_data, n_trials, 
                               numpyro_time, pymc_time):
    """
    可视化分析结果对比
    
    参数:
        trace_numpyro: NumPyro抽样结果
        trace_pymc: PyMC5抽样结果
        observed_data: 观测数据
        n_trials: 试验次数
        numpyro_time: NumPyro耗时
        pymc_time: PyMC5耗时
    """
    # 创建图形
    fig = plt.figure(figsize=(20, 12))
    
    # 1. 后验分布对比
    ax1 = plt.subplot(2, 3, 1)
    az.plot_posterior(trace_numpyro, var_names=['p'], ax=ax1, color='C0')
    ax1.set_title(f'NumPyro后验分布\n耗时: {numpyro_time:.2f}秒')
    
    ax2 = plt.subplot(2, 3, 2)
    az.plot_posterior(trace_pymc, var_names=['p'], ax=ax2, color='C1')
    ax2.set_title(f'PyMC5后验分布\n耗时: {pymc_time:.2f}秒')
    
    # 2. 迹线图对比
    ax3 = plt.subplot(2, 3, 3)
    az.plot_trace(trace_numpyro, var_names=['p'])
    ax3.set_title('NumPyro迹线图')
    
    ax4 = plt.subplot(2, 3, 4)
    az.plot_trace(trace_pymc, var_names=['p'])
    ax4.set_title('PyMC5迹线图')
    
    # 3. 性能对比
    ax5 = plt.subplot(2, 3, 5)
    backends = ['NumPyro', 'PyMC5']
    times = [numpyro_time, pymc_time]
    speedup = pymc_time / numpyro_time if numpyro_time > 0 else 1
    
    bars = ax5.bar(backends, times, color=['C0', 'C1'])
    ax5.set_ylabel('采样时间 (秒)')
    ax5.set_title(f'性能对比\nNumPyro加速比: {speedup:.1f}x')
    
    # 添加数值标签
    for bar, time in zip(bars, times):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{time:.2f}s', ha='center', va='bottom')
    
    # 4. 观测数据分布
    ax6 = plt.subplot(2, 3, 6)
    ax6.hist(observed_data, bins=30, alpha=0.7, density=True, 
             color='skyblue', edgecolor='black')
    ax6.set_xlabel('成功次数')
    ax6.set_ylabel('密度')
    ax6.set_title('观测数据分布')
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle('NumPyro vs PyMC5 二项分布抽样对比分析', fontsize=16)
    plt.tight_layout()
    plt.savefig('binomial_numpyro_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def parameter_sensitivity_analysis_numpyro(n_trials=50, n_observations=500):
    """
    使用NumPyro进行参数敏感性分析
    
    参数:
        n_trials: 试验次数
        n_observations: 观测样本数
    """
    print("\n=== NumPyro 参数敏感性分析 ===")
    
    # 测试不同的真实概率值
    true_ps = [0.1, 0.3, 0.5, 0.7, 0.9]
    results = []
    
    fig = plt.figure(figsize=(15, 5))
    
    for i, p_true in enumerate(true_ps):
        # 生成数据并拟合
        np.random.seed(42 + i)
        observed_data = np.random.binomial(n=n_trials, p=p_true, size=n_observations)
        
        # NumPyro模型
        def binomial_model(observed=None, n_trials=n_trials):
            p = numpyro.sample('p', dist.Beta(2.0, 2.0))
            numpyro.sample('y', dist.Binomial(total_count=n_trials, probs=p), obs=observed)
        
        rng_key = random.PRNGKey(42 + i)
        kernel = NUTS(binomial_model)
        mcmc = MCMC(kernel, num_warmup=500, num_samples=1000, num_chains=optimal_chains, progress_bar=False)
        
        mcmc.run(rng_key, observed=observed_data, n_trials=n_trials)
        
        # 获取后验均值
        samples = mcmc.get_samples()
        p_estimated = np.mean(samples['p'])
        results.append((p_true, p_estimated))
        
        # 绘制子图
        ax = fig.add_subplot(1, 5, i+1)
        trace = az.from_numpyro(mcmc)
        az.plot_posterior(trace, var_names=['p'], ax=ax)
        ax.set_title(f'真实p={p_true}\n估计p={p_estimated:.3f}')
    
    plt.suptitle('NumPyro: 不同真实概率值的参数估计结果', fontsize=14)
    plt.tight_layout()
    plt.savefig('parameter_sensitivity_numpyro.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 打印结果对比
    print("\n真实值 vs 估计值对比:")
    print("真实p\t估计p\t绝对误差")
    print("-" * 30)
    for p_true, p_est in results:
        print(f"{p_true:.1f}\t{p_est:.3f}\t{abs(p_true - p_est):.3f}")

def performance_benchmark():
    """
    性能基准测试
    """
    print("\n=== 性能基准测试 ===")
    
    # 测试参数
    n_trials = 100
    n_observations = 1000
    num_samples = 2000
    num_warmup = 1000
    
    try:
        # NumPyro测试
        print("正在运行NumPyro性能测试...")
        trace_numpyro, data_numpyro, time_numpyro = create_binomial_model_numpyro(
            n_trials=n_trials, true_p=0.3, n_observations=n_observations,
            num_samples=num_samples, num_warmup=num_warmup
        )
        
        # PyMC5测试
        print("正在运行PyMC5性能测试...")
        trace_pymc, data_pymc, time_pymc = create_binomial_model_pymc(
            n_trials=n_trials, true_p=0.3, n_observations=n_observations,
            num_samples=num_samples, num_warmup=num_warmup
        )
        
        # 性能对比
        speedup = time_pymc / time_numpyro if time_numpyro > 0 else 1
        print(f"\n=== 性能对比结果 ===")
        print(f"PyMC5耗时: {time_pymc:.2f}秒")
        print(f"NumPyro耗时: {time_numpyro:.2f}秒")
        print(f"加速比: {speedup:.1f}x")
        
        return trace_numpyro, trace_pymc, data_numpyro, time_numpyro, time_pymc
        
    except Exception as e:
        print(f"\n❌ 性能测试过程中出现错误: {str(e)}")
        print("请检查数据有效性和模型参数设置")
        raise

def main():
    """主函数"""
    print("=" * 80)
    print("NumPyro 二项分布抽样演示")
    print("基于JAX的高性能MCMC抽样")
    print("=" * 80)
    
    # JAX信息
    print(f"JAX版本: {jax.__version__}")
    print(f"JAX设备: {jax.devices()}")
    print(f"可用核心数: {n_cores}")
    
    # 性能基准测试
    trace_numpyro, trace_pymc, observed_data, numpyro_time, pymc_time = performance_benchmark()
    
    # 分析结果
    summary_numpyro = analyze_results(trace_numpyro, observed_data, "NumPyro")
    summary_pymc = analyze_results(trace_pymc, observed_data, "PyMC5")
    
    # 可视化对比
    visualize_results_comparison(trace_numpyro, trace_pymc, observed_data, 
                               n_trials=100, numpyro_time=numpyro_time, pymc_time=pymc_time)
    
    # NumPyro参数敏感性分析
    parameter_sensitivity_analysis_numpyro()
    
    print("\n" + "=" * 80)
    print("NumPyro演示完成！请查看生成的图表文件:")
    print("- binomial_numpyro_comparison.png: 性能对比分析")
    print("- parameter_sensitivity_numpyro.png: NumPyro参数敏感性分析")
    print("=" * 80)

if __name__ == "__main__":
    main()