# 添加时序数据相关的参数

def add_time_series_args(parser):
    """添加时序数据相关参数"""
    group = parser.add_argument_group('Time Series Arguments')
    
    group.add_argument(
        '--ts-input-dim',
        type=int,
        default=10,
        help='时序数据的特征维度'
    )
    
    group.add_argument(
        '--max-seq-len',
        type=int,
        default=1000,
        help='时序数据的最大序列长度'
    )
    
    group.add_argument(
        '--ts-encoder-type',
        type=str,
        default='transformer',
        choices=['transformer', 'cnn'],
        help='时序编码器类型'
    )
    
    group.add_argument(
        '--ts-d-model',
        type=int,
        default=512,
        help='Transformer模型维度'
    )
    
    group.add_argument(
        '--ts-nhead',
        type=int,
        default=8,
        help='Transformer注意力头数'
    )
    
    group.add_argument(
        '--ts-num-layers',
        type=int,
        default=6,
        help='Transformer层数'
    )
    
    return parser