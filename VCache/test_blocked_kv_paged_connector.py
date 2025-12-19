#!/usr/bin/env python3
"""
测试 BlockedKVPagedMemConnector
"""

import torch
import sys
import os

# 添加路径以便导入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lmcache.test.blocked_kv_paged_connector import BlockedKVPagedMemConnector


def test_connector_initialization():
    """测试connector初始化"""
    print("测试connector初始化...")
    
    # 配置参数
    num_layers = 4
    block_size = 16
    num_kv_heads = 8
    head_size = 64
    
    # 创建connector
    connector = BlockedKVPagedMemConnector(
        num_layers=num_layers,
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        use_gpu=False  # 不使用GPU缓冲区进行测试
    )
    
    print(f"✓ Connector初始化成功")
    print(f"  - 层数: {connector.num_layers}")
    print(f"  - block大小: {connector.block_size}")
    print(f"  - KV头数: {connector.num_kv_heads}")
    print(f"  - 头大小: {connector.head_size}")
    print(f"  - 隐藏维度大小: {connector.hidden_dim_size}")
    
    return connector


def test_format_conversion(connector):
    """测试格式转换函数"""
    print("\n测试格式转换函数...")
    
    # 测试5D格式转换
    print("测试5D格式转换...")
    blocked_5d = torch.randn(
        connector.num_layers, 2, connector.block_size, 
        connector.num_kv_heads, connector.head_size,
        dtype=torch.float32
    )
    
    start = 0
    end = connector.block_size
    flash_tensor = connector._convert_to_flash_attention_format(blocked_5d, start, end)
    
    print(f"✓ 5D格式转换成功")
    print(f"  - 输入形状: {blocked_5d.shape}")
    print(f"  - 输出形状: {flash_tensor.shape}")
    print(f"  - 期望形状: (2, {connector.num_layers}, {connector.block_size}, {connector.hidden_dim_size})")
    
    # 测试反向转换
    num_blocks = 1
    blocked_6d = connector._convert_from_flash_attention_format(flash_tensor, num_blocks)
    
    print(f"✓ 反向转换成功")
    print(f"  - 输入形状: {flash_tensor.shape}")
    print(f"  - 输出形状: {blocked_6d.shape}")
    print(f"  - 期望形状: ({connector.num_layers}, {num_blocks}, 2, {connector.block_size}, {connector.num_kv_heads}, {connector.head_size})")
    
    # 测试6D格式转换
    print("\n测试6D格式转换...")
    num_blocks = 2
    blocked_6d = torch.randn(
        connector.num_layers, num_blocks, 2, connector.block_size,
        connector.num_kv_heads, connector.head_size,
        dtype=torch.float32
    )
    
    start = 0
    end = num_blocks * connector.block_size
    flash_tensor = connector._convert_to_flash_attention_format(blocked_6d, start, end)
    
    print(f"✓ 6D格式转换成功")
    print(f"  - 输入形状: {blocked_6d.shape}")
    print(f"  - 输出形状: {flash_tensor.shape}")
    print(f"  - 期望形状: (2, {connector.num_layers}, {end-start}, {connector.hidden_dim_size})")
    
    # 测试反向转换
    blocked_6d_back = connector._convert_from_flash_attention_format(flash_tensor, num_blocks)
    
    print(f"✓ 反向转换成功")
    print(f"  - 输入形状: {flash_tensor.shape}")
    print(f"  - 输出形状: {blocked_6d_back.shape}")
    print(f"  - 期望形状: ({connector.num_layers}, {num_blocks}, 2, {connector.block_size}, {connector.num_kv_heads}, {connector.head_size})")


def test_shape_method(connector):
    """测试get_shape方法"""
    print("\n测试get_shape方法...")
    
    num_tokens = 32
    shape = connector.get_shape(num_tokens)
    
    print(f"✓ get_shape方法测试成功")
    print(f"  - token数量: {num_tokens}")
    print(f"  - 返回形状: {shape}")
    print(f"  - 期望形状: (2, {connector.num_layers}, {num_tokens}, {connector.hidden_dim_size})")


def test_pointer_initialization(connector):
    """测试指针初始化"""
    print("\n测试指针初始化...")
    
    if not torch.cuda.is_available():
        print("⚠ CUDA不可用，跳过指针初始化测试")
        return
    
    # 模拟vLLM KV cache
    num_blocks = 4
    vllm_kvcaches = []
    for _ in range(connector.num_layers):
        kv_cache = torch.randn(
            2, num_blocks, connector.block_size, 
            connector.num_kv_heads, connector.head_size,
            dtype=torch.float16, device="cuda"
        )
        vllm_kvcaches.append(kv_cache)
    
    try:
        kv_cache_pointers = connector._initialize_pointers(vllm_kvcaches)
        print(f"✓ 指针初始化成功")
        print(f"  - 指针形状: {kv_cache_pointers.shape}")
        print(f"  - 期望形状: ({connector.num_layers},)")
        print(f"  - page_buffer_size: {connector.page_buffer_size}")
    except Exception as e:
        print(f"✗ 指针初始化失败: {e}")


def main():
    """主测试函数"""
    print("=" * 60)
    print("测试 BlockedKVPagedMemConnector")
    print("=" * 60)
    
    try:
        # 测试初始化
        connector = test_connector_initialization()
        
        # 测试格式转换
        test_format_conversion(connector)
        
        # 测试shape方法
        test_shape_method(connector)
        
        # 测试指针初始化
        test_pointer_initialization(connector)
        
        print("\n" + "=" * 60)
        print("所有测试完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

