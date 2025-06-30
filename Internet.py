import requests
import socket
import time
import os
import wandb
from urllib.parse import urlparse

def test_basic_connectivity():
    """测试基础网络连接"""
    print("=== 基础网络连接测试 ===")
    
    # 测试基本的互联网连接
    try:
        response = requests.get("https://www.google.com", timeout=5)
        print("✅ 基本互联网连接正常")
    except Exception as e:
        print(f"❌ 基本互联网连接失败: {e}")
        return False
    
    # 测试DNS解析
    try:
        ip = socket.gethostbyname("api.wandb.ai")
        print(f"✅ DNS解析正常: api.wandb.ai -> {ip}")
    except Exception as e:
        print(f"❌ DNS解析失败: {e}")
        return False
    
    return True

def test_wandb_endpoints():
    """测试wandb各个端点"""
    print("\n=== W&B端点连接测试 ===")
    
    endpoints = [
        "https://api.wandb.ai/health",
        "https://api.wandb.ai/graphql",
        "https://wandb.ai"
    ]
    
    for endpoint in endpoints:
        try:
            response = requests.get(endpoint, timeout=10)
            print(f"✅ {endpoint}: {response.status_code}")
        except requests.exceptions.ConnectTimeout:
            print(f"❌ {endpoint}: 连接超时")
        except requests.exceptions.SSLError as e:
            print(f"⚠️ {endpoint}: SSL错误 - {e}")
        except Exception as e:
            print(f"❌ {endpoint}: {e}")

def test_different_timeouts():
    """测试不同超时设置"""
    print("\n=== 超时测试 ===")
    
    for timeout in [5, 10, 30, 60]:
        try:
            start_time = time.time()
            response = requests.get("https://api.wandb.ai/health", timeout=timeout)
            elapsed = time.time() - start_time
            print(f"✅ 超时{timeout}s: 成功 (用时{elapsed:.2f}s)")
            return timeout
        except Exception as e:
            print(f"❌ 超时{timeout}s: 失败 - {e}")
    
    return None

def test_wandb_init():
    """测试wandb初始化"""
    print("\n=== W&B初始化测试 ===")
    
    # 测试离线模式
    try:
        os.environ["WANDB_MODE"] = "offline"
        run = wandb.init(project="test", settings=wandb.Settings(init_timeout=30))
        wandb.finish()
        print("✅ 离线模式初始化成功")
    except Exception as e:
        print(f"❌ 离线模式初始化失败: {e}")
    
    # 测试在线模式
    try:
        os.environ.pop("WANDB_MODE", None)
        run = wandb.init(project="test", settings=wandb.Settings(init_timeout=60))
        wandb.finish()
        print("✅ 在线模式初始化成功")
        return True
    except Exception as e:
        print(f"❌ 在线模式初始化失败: {e}")
        return False

def check_proxy_settings():
    """检查代理设置"""
    print("\n=== 代理设置检查 ===")
    
    proxy_vars = ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY']
    
    for var in proxy_vars:
        value = os.environ.get(var)
        if value:
            print(f"📋 {var}: {value}")
        else:
            print(f"📋 {var}: 未设置")

def main():
    print("W&B连接诊断工具")
    print("=" * 50)
    
    # 检查代理设置
    check_proxy_settings()
    
    # 基础连接测试
    if not test_basic_connectivity():
        print("\n❌ 基础网络连接有问题，请检查网络设置")
        return
    
    # 测试wandb端点
    test_wandb_endpoints()
    
    # 测试超时设置
    working_timeout = test_different_timeouts()
    
    # 测试wandb初始化
    online_works = test_wandb_init()
    
    # 给出建议
    print("\n" + "=" * 50)
    print("🎯 建议:")
    
    if online_works:
        print("✅ 在线模式工作正常，可以直接使用wandb")
    elif working_timeout:
        print(f"⚠️ 连接较慢，建议设置更长的超时时间: init_timeout={working_timeout + 30}")
    else:
        print("❌ 在线模式无法工作，建议:")
        print("   1. 使用离线模式: export WANDB_MODE=offline")
        print("   2. 检查防火墙设置")
        print("   3. 尝试使用VPN")
        print("   4. 联系网络管理员")

if __name__ == "__main__":
    main()