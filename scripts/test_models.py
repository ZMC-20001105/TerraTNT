#!/usr/bin/env python
"""
快速验证所有基线模型是否可以正常导入和运行
"""
import sys
import torch
import numpy as np

def test_model(model_name, model_class):
    """测试单个模型"""
    print(f"\n{'='*60}")
    print(f"测试 {model_name}")
    print('='*60)
    
    try:
        # 创建模型配置
        config = {
            'history_len': 10,
            'future_len': 60,
            'in_channels': 18,
            'map_size': 128,
            'hidden_dim': 256,
            'num_modes': 5
        }
        model = model_class(config)
        model.eval()
        
        # 创建测试数据
        batch_size = 2
        history = torch.randn(batch_size, 10, 2)
        env_map = torch.randn(batch_size, 18, 128, 128)
        goals = torch.randn(batch_size, 5, 2)
        
        # 前向传播（不同模型接受不同参数）
        with torch.no_grad():
            if model_name in ['Social-LSTM', 'Constant Velocity']:
                # 这些模型不使用env_map
                output = model(history)
            else:
                # 其他模型使用env_map，goals作为kwargs传递
                output = model(history, env_map, goals=goals)
        
        print(f"✓ {model_name} 导入成功")
        print(f"✓ 前向传播成功")
        # 处理可能返回tuple的情况（如PECNet返回(pred, mu, log_var)）
        if isinstance(output, tuple):
            output = output[0]
        print(f"  输出形状: {output.shape}")
        print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")
        
        return True
        
    except Exception as e:
        print(f"✗ {model_name} 失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("="*60)
    print("基线模型验证测试")
    print("="*60)
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    # 添加项目路径
    sys.path.insert(0, '/home/zmc/文档/programwork')
    
    results = {}
    
    # 测试YNet
    try:
        from models.baselines.ynet import YNet
        results['YNet'] = test_model('YNet', YNet)
    except Exception as e:
        print(f"✗ YNet导入失败: {e}")
        results['YNet'] = False
    
    # 测试PECNet
    try:
        from models.baselines.pecnet import PECNet
        results['PECNet'] = test_model('PECNet', PECNet)
    except Exception as e:
        print(f"✗ PECNet导入失败: {e}")
        results['PECNet'] = False
    
    # 测试Trajectron++
    try:
        from models.baselines.trajectron import TrajectronPP
        results['Trajectron++'] = test_model('Trajectron++', TrajectronPP)
    except Exception as e:
        print(f"✗ Trajectron++导入失败: {e}")
        results['Trajectron++'] = False
    
    # 测试Social-LSTM
    try:
        from models.baselines.social_lstm import SocialLSTM
        results['Social-LSTM'] = test_model('Social-LSTM', SocialLSTM)
    except Exception as e:
        print(f"✗ Social-LSTM导入失败: {e}")
        results['Social-LSTM'] = False
    
    # 测试Constant Velocity
    try:
        from models.baselines.constant_velocity import ConstantVelocity
        results['CV'] = test_model('Constant Velocity', ConstantVelocity)
    except Exception as e:
        print(f"✗ Constant Velocity导入失败: {e}")
        results['CV'] = False
    
    # 汇总结果
    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)
    
    success_count = sum(results.values())
    total_count = len(results)
    
    for model_name, success in results.items():
        status = "✓ 通过" if success else "✗ 失败"
        print(f"{model_name:20s} {status}")
    
    print(f"\n通过: {success_count}/{total_count}")
    
    if success_count == total_count:
        print("\n✅ 所有模型验证通过！可以开始训练。")
        return 0
    else:
        print(f"\n⚠️  {total_count - success_count} 个模型验证失败")
        return 1

if __name__ == '__main__':
    sys.exit(main())
