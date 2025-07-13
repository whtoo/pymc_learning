#!/usr/bin/env python3
"""
NumPyro 二项分布抽样简单版本
专注于NumPyro加速特性的简洁实现
"""

import numpy as np
import matplotlib.pyplot as plt
import arviz as az
import time

import numpyro
import jax
import jax.numpy as jnp
from jax import random
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def run_numpyro_binomial(n_trials=100, true_p=0.3, n_observations=1000):
    """使用NumPyro运行二项分布抽样"""
    print("NumPyro 二项分布抽样演示")
    print("=" * 40)
    print(f"JAX设备: {jax.devices()}")
    
    # 生成数据
    np.random.seed(42)
    observed_data = np.random.binomial(n_trials, true_p, n_observations)
    
    # NumPyro模型
    def model(observed=None):
        p = numpyro.sample('p', dist.Beta(2.0, 2.0))
        numpyro.sample('y', dist.Binomial(n_trials, p), obs=observed)
    
    # 运行MCMC
    start_time = time.time()
    
    rng_key = random.PRNGKey(42)
    kernel = NUTS(model)
    mcmc = MCMC(kernel, num_warmup=1000, num_samples=2000, num_chains=2)
    
    mcmc.run(rng_key, observed=observed_data)
    
    elapsed_time = time.time() - start_time
    
    # 分析结果
    trace = az.from_numpyro(mcmc)
    summary = az.summary(trace, var_names=['p'])
    
    print(f"\n=== 结果 ===")
    print(f"采样时间: {elapsed_time:.2f}秒")
    print(f"后验均值: {summary['mean']['p']:.3f}")
    print(f"95%可信区间: [{summary['hdi_3%']['p']:.3f}, {summary['hdi_97%']['p']:.3f}]")
    
    # 可视化
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    az.plot_posterior(trace, var_names=['p'])
    plt.title('后验分布')
    
    plt.subplot(1, 3, 2)
    az.plot_trace(trace, var_names=['p'])
    plt.suptitle('MCMC诊断')
    
    plt.subplot(1, 3, 3)
    plt.hist(observed_data, bins=30, alpha=0.7, density=True)
    plt.title('观测数据')
    plt.xlabel('成功次数')
    
    plt.tight_layout()
    plt.savefig('numpyro_simple.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return trace, elapsed_time

if __name__ == "__main__":
    run_numpyro_binomial()