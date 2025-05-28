import os
import argparse
import torch
import time  # 导入time模块用于时间测量
from pathlib import Path

from model import Transformer

def load_model(model_path):
    # 现有代码保持不变
    """加载保存的模型并返回配置信息"""
    checkpoint = torch.load(model_path)
    config = checkpoint['config']
    
    model = Transformer(
        num_layers=config['num_layers'],
        dim_model=config['dim_model'],
        num_heads=config['num_heads'],
        num_tokens=config['prime'] + 2,
        seq_len=5
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, config, checkpoint

def predict_result(model, x, y, prime, operation):
    """使用模型预测x和y的运算结果，并返回耗时"""
    # 计时开始
    start_time = time.time()
    
    # 创建输入序列 [x, op, y, eq]
    op_token = prime + 1  # 操作符的token
    eq_token = prime      # 等号的token
    
    inputs = torch.tensor([[x, op_token, y, eq_token]])
    
    # 执行推理
    with torch.no_grad():
        output = model(inputs)[-1, 0, :]
        predicted = torch.argmax(output).item()
    
    # 计时结束
    end_time = time.time()
    elapsed_time = (end_time - start_time) * 1000  # 转换为毫秒
    
    return predicted, elapsed_time

def calculate_expected(x, y, prime, operation):
    """计算预期的数学结果，并返回耗时"""
    # 计时开始
    start_time = time.time()
    
    if operation == "x/y":
        # 对于模除法，需要计算模逆元
        for expected in range(prime):
            if (expected * y) % prime == x:
                result = expected
                break
    elif operation == "x+y":
        result = (x + y) % prime
    elif operation == "x-y":
        result = (x - y) % prime
    else:
        result = None
    
    # 计时结束
    end_time = time.time()
    elapsed_time = (end_time - start_time) * 1000  # 转换为毫秒
    
    return result, elapsed_time

def interactive_inference(model, config):
    """交互式推理循环"""
    prime = config['prime']
    operation = config['operation']
    
    print("\n" + "="*50)
    print(f"模型信息:")
    print(f"操作类型: {operation}")
    print(f"模数 p: {prime}")
    print(f"模型尺寸: {config['dim_model']}")
    print(f"层数: {config['num_layers']}")
    print("="*50)
    
    print(f"\n请输入要计算的 {operation} 操作 (模 {prime})。输入q退出。")
    
    # 跟踪累计时间
    total_model_time = 0
    total_math_time = 0
    prediction_count = 0
    
    while True:
        try:
            x_input = input("\n输入 x (0-{}) 或 q 退出: ".format(prime-1))
            if x_input.lower() == 'q':
                # 显示平均时间统计
                if prediction_count > 0:
                    print("\n" + "="*50)
                    print(f"时间统计 ({prediction_count}次计算):")
                    print(f"平均模型预测时间: {total_model_time/prediction_count:.3f} 毫秒")
                    print(f"平均直接计算时间: {total_math_time/prediction_count:.3f} 毫秒")
                    print(f"平均速度比: 模型预测是直接计算的 {total_model_time/total_math_time:.2f}x 倍")
                    print("="*50)
                break
                
            x = int(x_input)
            if x < 0 or x >= prime:
                print(f"错误: x必须在0到{prime-1}之间")
                continue
                
            # 对于除法，y不能为0
            if operation == "x/y":
                y_range = f"1-{prime-1}"
            else:
                y_range = f"0-{prime-1}"
                
            y_input = input(f"输入 y ({y_range}): ")
            y = int(y_input)
            
            if operation == "x/y" and (y < 1 or y >= prime):
                print(f"错误: 对于除法，y必须在1到{prime-1}之间")
                continue
            elif y < 0 or y >= prime:
                print(f"错误: y必须在0到{prime-1}之间")
                continue
            
            # 执行预测并测量时间
            predicted, model_time = predict_result(model, x, y, prime, operation)
            expected, math_time = calculate_expected(x, y, prime, operation)
            
            # 更新累计时间
            total_model_time += model_time
            total_math_time += math_time
            prediction_count += 1
            
            # 显示结果
            print("\n" + "-"*40)
            print(f"计算: {x} {operation} {y} ≡ ? (mod {prime})")
            print(f"模型预测: {predicted} (耗时: {model_time:.3f} 毫秒)")
            print(f"实际结果: {expected} (耗时: {math_time:.3f} 毫秒)")
            print(f"速度比较: 模型预测是直接计算的 {model_time/math_time:.2f}x 倍")
            
            if predicted == expected:
                print("✓ 预测正确!")
            else:
                print("✗ 预测错误!")
            print("-"*40)
            
        except ValueError:
            print("输入错误: 请输入有效的数字")
        except KeyboardInterrupt:
            print("\n再见!")
            break

# main部分保持不变
if __name__ == "__main__":
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 获取项目根目录
    project_root = os.path.dirname(script_dir)
    # 构建模型路径
    MODEL_PATH = os.path.join(project_root, "saved_models", "final_model_w003_test.pt")
    
    try:
        print(f"正在加载模型: {MODEL_PATH}")
        model, config, checkpoint = load_model(MODEL_PATH)
        interactive_inference(model, config)
    except FileNotFoundError:
        print(f"错误: 找不到模型文件 {MODEL_PATH}")
    except Exception as e:
        print(f"加载模型时出错: {str(e)}")