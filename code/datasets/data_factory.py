from data_loader import Dataset_Battery_SOH
from torch.utils.data import DataLoader

data_dict = {
    'Battery_SOH': Dataset_Battery_SOH,  # 新增
}

def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1
    percent = args.percent

    if flag == 'test' or flag == 'val':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
        freq = args.freq
    else:  # flag == 'train'
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    if args.data == 'm4':
        drop_last = False
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=args.seasonal_patterns
        )
    elif args.data == 'Battery_SOH':
        # --- [核心修复 2] ---
        if flag == 'train':
            data_folder = getattr(args, 'train_folder', 'train_1')
            scaler_to_pass = None  # 训练集不接收外部 scaler
        else:  # 'test' or 'val'
            data_folder = getattr(args, 'test_folder', 'test_1')
            # 从 args 中获取训练集传递过来的 scaler
            scaler_to_pass = getattr(args, 'scaler', None)
            if scaler_to_pass is None:
                # 这是一个安全检查，确保 scaler 被正确传递
                raise ValueError("测试/验证集需要一个从训练集传递过来的 scaler，但没有收到！")
        
        data_set = Data(
            root_path=args.root_path,
            data_path=data_folder,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            percent=percent,
            desc_path=getattr(args, 'desc_path', 'data_preprocess/full_descriptions_simple_decimal3'),
            scaler=scaler_to_pass  # 将 scaler 传递给数据集
        )
    else:
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            percent=percent,
            seasonal_patterns=args.seasonal_patterns
        )
    
    print(f"[{flag}] DataLoader created with batch_size={batch_size}, shuffle={shuffle_flag}")
    
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last
    )
    
    return data_set, data_loader
