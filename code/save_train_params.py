"""
保存训练参数供评估使用
"""
import pickle

def save_training_params(dataset, output_path: str):
    """
    保存训练集参数供评估使用
    
    Args:
        dataset: BatterySOHDataset 训练集实例
        output_path: 输出路径 (建议: outputs/batterygpt/train_params.pkl)
    """
    params = dataset.get_transfer_params()
    
    with open(output_path, 'wb') as f:
        pickle.dump(params, f)
    
    print(f"✅ Training params saved to: {output_path}")
    print(f"   - selected_features: {params['selected_features']}")
    print(f"   - feature_scaler keys: {list(params['feature_scaler'].keys())}")


# 在训练脚本中调用:
# from save_train_params import save_training_params
# save_training_params(train_dataset, 'outputs/batterygpt/train_params.pkl')