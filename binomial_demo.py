#!/usr/bin/env python3
"""
PyMC5 二项分布抽样演示脚本
Binomial Distribution Sampling Demo with PyMC5

这个脚本演示了如何使用PyMC5进行二项分布的随机抽样和可视化分析
"""

import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az
import multiprocessing

# 自动检测CPU核心数
n_cores = 2#multiprocessing.cpu_count()
optimal_chains = 2#min(n_cores * 2, 8)  # 通常不超过8链

import pytensor 

pytensor.config.cxx = ""

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def create_binomial_model(n_trials=100, true_p=0.3, n_observations=1000):
    """
    创建二项分布模型
    
    参数:
        n_trials: 每次试验的尝试次数
        true_p: 真实的成功概率
        n_observations: 观测样本数量
    
    返回:
        model: PyMC模型对象
        trace: 抽样结果
    """
    print(f"创建二项分布模型:")
    print(f"- 试验次数 n = {n_trials}")
    print(f"- 真实概率 p = {true_p}")
    print(f"- 观测样本数 = {n_observations}")
    
    # 生成观测数据
    np.random.seed(42)
    observed_data = np.random.binomial(n=n_trials, p=true_p, size=n_observations)
    
    with pm.Model() as model:
        # 定义先验分布 - Beta分布作为p的先验
        p = pm.Beta('p', alpha=2, beta=2)
        
        # 定义似然函数 - 二项分布
        y = pm.Binomial('y', n=n_trials, p=p, observed=observed_data)
        
        # 执行MCMC抽样
        trace = pm.sample(2000, tune=1000, chains=optimal_chains, target_accept=0.9, random_seed=42)
    
    return model, trace, observed_data

def analyze_results(trace, observed_data):
    """
    分析抽样结果
    
    参数:
        trace: MCMC抽样结果
        observed_data: 观测数据
    
    返回:
        summary: 统计摘要
    """
    # 计算统计摘要
    summary = az.summary(trace, var_names=['p'])
    print("\n=== 参数估计结果 ===")
    print(summary)
    
    # 计算观测数据的统计量
    observed_mean = np.mean(observed_data)
    observed_std = np.std(observed_data)
    observed_p_hat = observed_mean / np.max(observed_data) if len(observed_data) > 0 else 0
    
    print(f"\n=== 观测数据统计 ===")
    print(f"观测均值: {observed_mean:.2f}")
    print(f"观测标准差: {observed_std:.2f}")
    print(f"估计概率 p̂: {observed_p_hat:.3f}")
    
    return summary

def visualize_results(trace, observed_data, n_trials):
    """
    可视化分析结果
    
    参数:
        trace: MCMC抽样结果
        observed_data: 观测数据
        n_trials: 试验次数
    """
    # 创建多个独立的图形
    fig1, axes1 = plt.subplots(1, 2, figsize=(15, 6))
    fig1.suptitle('PyMC5 二项分布抽样分析结果 - 第一部分', fontsize=16)
    
    # 1. 后验分布图
    az.plot_posterior(trace, var_names=['p'], ax=axes1[0])
    axes1[0].set_title('参数p的后验分布')
    
    # 2. 迹线图
    az.plot_trace(trace, var_names=['p'])
    plt.suptitle('MCMC迹线图')
    
    # 创建第二个图形
    fig2, axes2 = plt.subplots(1, 2, figsize=(15, 6))
    fig2.suptitle('PyMC5 二项分布抽样分析结果 - 第二部分', fontsize=16)
    
    # 3. 观测数据直方图
    axes2[0].hist(observed_data, bins=30, alpha=0.7, density=True, color='skyblue', edgecolor='black')
    axes2[0].set_xlabel('成功次数')
    axes2[0].set_ylabel('密度')
    axes2[0].set_title('观测数据分布')
    axes2[0].grid(True, alpha=0.3)
    
    # 4. 理论分布与观测对比
    from scipy.stats import binom
    
    x = np.arange(0, n_trials + 1)
    p_estimated = np.mean(trace.posterior['p'].values)
    theoretical_dist = binom.pmf(x, n_trials, p_estimated)
    
    axes2[1].hist(observed_data, bins=30, alpha=0.5, density=True, color='lightcoral', label='观测数据', edgecolor='black')
    axes2[1].plot(x, theoretical_dist, 'b-', linewidth=2, label=f'理论分布 (p={p_estimated:.3f})')
    axes2[1].set_xlabel('成功次数')
    axes2[1].set_ylabel('概率密度')
    axes2[1].set_title('理论与观测分布对比')
    axes2[1].legend()
    axes2[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('binomial_demo_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def parameter_sensitivity_analysis(n_trials=50, n_observations=500):
    """
    参数敏感性分析
    
    参数:
        n_trials: 试验次数
        n_observations: 观测样本数
    """
    print("\n=== 参数敏感性分析 ===")
    
    # 测试不同的真实概率值
    true_ps = [0.1, 0.3, 0.5, 0.7, 0.9]
    results = []
    
    fig = plt.figure(figsize=(15, 5))
    
    for i, p_true in enumerate(true_ps):
        # 生成数据并拟合
        np.random.seed(42 + i)
        observed_data = np.random.binomial(n=n_trials, p=p_true, size=n_observations)
        
        with pm.Model() as model:
            p = pm.Beta('p', alpha=2, beta=2)
            y = pm.Binomial('y', n=n_trials, p=p, observed=observed_data)
            
            trace = pm.sample(1000, tune=500, chains=optimal_chains, target_accept=0.9,
                            random_seed=42, progressbar=False)
        
        # 获取后验均值
        p_estimated = np.mean(trace.posterior['p'].values)
        results.append((p_true, p_estimated))
        
        # 绘制子图
        ax = fig.add_subplot(1, 5, i+1)
        az.plot_posterior(trace, var_names=['p'], ax=ax)
        ax.set_title(f'真实p={p_true}\n估计p={p_estimated:.3f}')
    
    plt.suptitle('不同真实概率值的参数估计结果', fontsize=14)
    plt.tight_layout()
    plt.savefig('parameter_sensitivity.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 打印结果对比
    print("\n真实值 vs 估计值对比:")
    print("真实p\t估计p\t绝对误差")
    print("-" * 30)
    for p_true, p_est in results:
        print(f"{p_true:.1f}\t{p_est:.3f}\t{abs(p_true - p_est):.3f}")

def main():
    """主函数"""
    print("=" * 60)
    print("PyMC5 二项分布抽样演示")
    print("=" * 60)
    
    # 基本演示
    model, trace, observed_data = create_binomial_model(
        n_trials=100, 
        true_p=0.3, 
        n_observations=1000
    )
    
    # 结果分析
    summary = analyze_results(trace, observed_data)
    
    # 可视化结果
    visualize_results(trace, observed_data, n_trials=100)
    
    # 参数敏感性分析
    parameter_sensitivity_analysis()
    
    print("\n" + "=" * 60)
    print("演示完成！请查看生成的图表文件:")
    print("- binomial_demo_results.png: 主要分析结果")
    print("- parameter_sensitivity.png: 参数敏感性分析")
    print("=" * 60)

if __name__ == "__main__":
    main()