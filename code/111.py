def train_one_epoch(model, dataloader, optimizer, scaler, epoch, logger, rank, 
                    swanlab_logger=None, use_amp=True, gradient_accumulation_steps=4):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    total_text_loss = 0  # [新增]
    total_soh_loss = 0   # [新增]
    total_acc = 0
    num_batches = 0
    
    # 用于收集 SOH 预测
    all_soh_preds = []
    all_soh_labels = []
    
    optimizer.zero_grad()
    
    if rank == 0:
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    else:
        pbar = dataloader
    
    global_step = epoch * len(dataloader)
    
    for i, batch in enumerate(pbar):
        with autocast('cuda', enabled=use_amp):
            # [修改] forward 现在返回 4 个值
            loss, acc, soh_pred, soh_loss = model(batch)
            loss = loss / gradient_accumulation_steps
        
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # 梯度累积
        if (i + 1) % gradient_accumulation_steps == 0:
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
        
        # 累积指标
        loss_value = loss.item() * gradient_accumulation_steps
        soh_loss_value = soh_loss.item()  # [新增]
        
        # [新增] 计算文本损失（总损失 - SOH损失）
        text_loss_value = loss_value - 0.5 * soh_loss_value
        
        total_loss += loss_value
        total_text_loss += text_loss_value  # [新增]
        total_soh_loss += soh_loss_value    # [新增]
        total_acc += acc
        num_batches += 1
        
        # 收集 SOH 预测
        if 'soh_labels' in batch:
            all_soh_preds.append(soh_pred.detach().cpu().numpy())
            all_soh_labels.append(batch['soh_labels'].cpu().numpy())
        
        # 记录 batch 级别指标
        if swanlab_logger and (i + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            swanlab_logger.log_batch_metrics(
                loss=loss_value,
                text_loss=text_loss_value,  # [新增]
                soh_loss=soh_loss_value,    # [新增]
                acc=acc,
                soh_pred=soh_pred if 'soh_labels' in batch else None,
                soh_label=batch['soh_labels'] if 'soh_labels' in batch else None,
                iteration=global_step + i,
                lr=current_lr,
                mode='train'
            )
        
        if rank == 0:
            pbar.set_postfix({
                'loss': f'{loss_value:.4f}',
                'text_loss': f'{text_loss_value:.4f}',  # [新增]
                'soh_loss': f'{soh_loss_value:.4f}',    # [新增]
                'acc': f'{acc:.4f}'
            })
    
    avg_loss = total_loss / num_batches
    avg_text_loss = total_text_loss / num_batches  # [新增]
    avg_soh_loss = total_soh_loss / num_batches    # [新增]
    avg_acc = total_acc / num_batches
    
    if rank == 0:
        logger.info(f"Epoch {epoch}: Loss={avg_loss:.4f} (Text={avg_text_loss:.4f}, SOH={avg_soh_loss:.4f}), Acc={avg_acc:.4f}")
    
    return avg_loss, avg_text_loss, avg_soh_loss, avg_acc, all_soh_preds, all_soh_labels  # [修改返回值]



@torch.no_grad()
def evaluate(model, dataloader, logger, rank, swanlab_logger=None, epoch=0):
    """评估"""
    model.eval()
    total_loss = 0
    total_text_loss = 0  # [新增]
    total_soh_loss = 0   # [新增]
    total_mae = 0
    num_batches = 0
    
    all_soh_preds = []
    all_soh_labels = []
    
    if rank == 0:
        pbar = tqdm(dataloader, desc="Evaluating")
    else:
        pbar = dataloader
    
    for batch in pbar:
        loss, acc, soh_pred, soh_loss = model(batch)  # [修改]
        
        # 计算 MAE
        soh_true = batch['soh_labels'].to(soh_pred.device)
        mae = torch.abs(soh_pred.squeeze() - soh_true).mean().item()
        
        loss_value = loss.item()
        soh_loss_value = soh_loss.item()  # [新增]
        text_loss_value = loss_value - 0.5 * soh_loss_value  # [新增]
        
        total_loss += loss_value
        total_text_loss += text_loss_value  # [新增]
        total_soh_loss += soh_loss_value    # [新增]
        total_mae += mae
        num_batches += 1
        
        # 收集预测值
        all_soh_preds.append(soh_pred.squeeze().cpu().numpy())
        all_soh_labels.append(soh_true.cpu().numpy())
    
    avg_loss = total_loss / num_batches
    avg_text_loss = total_text_loss / num_batches  # [新增]
    avg_soh_loss = total_soh_loss / num_batches    # [新增]
    avg_mae = total_mae / num_batches
    
    # 合并所有预测
    all_soh_preds = np.concatenate(all_soh_preds)
    all_soh_labels = np.concatenate(all_soh_labels)
    
    # 计算 RMSE 和 R²
    rmse = np.sqrt(np.mean((all_soh_preds - all_soh_labels) ** 2))
    ss_res = np.sum((all_soh_labels - all_soh_preds) ** 2)
    ss_tot = np.sum((all_soh_labels - all_soh_labels.mean()) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))
    
    if rank == 0:
        logger.info(f"Eval: Loss={avg_loss:.4f} (Text={avg_text_loss:.4f}, SOH={avg_soh_loss:.4f}), MAE={avg_mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")
        
        # 绘制预测对比图
        if swanlab_logger:
            swanlab_logger.log_soh_prediction_plot(
                soh_true=all_soh_labels,
                soh_pred=all_soh_preds,
                epoch=epoch,
                split='val'
            )
    
    return avg_loss, avg_text_loss, avg_soh_loss, avg_mae, rmse, r2  # [修改返回值]



for epoch in range(1, config['num_epochs'] + 1):
    # 设置采样器的 epoch（用于 shuffle）
    if train_sampler is not None:
        train_sampler.set_epoch(epoch)
    
    # 训练 - [修改] 接收新的返回值
    train_loss, train_text_loss, train_soh_loss, train_acc, train_soh_preds, train_soh_labels = train_one_epoch(
        model, train_loader, optimizer, scaler, epoch, logger, rank,
        swanlab_logger=swanlab_logger,
        use_amp=config.get('use_amp', True),
        gradient_accumulation_steps=config.get('gradient_accumulation_steps', 4)
    )
    
    # 评估 - [修改] 接收新的返回值
    test_loss, test_text_loss, test_soh_loss, test_mae, test_rmse, test_r2 = evaluate(
        model, test_loader, logger, rank,
        swanlab_logger=swanlab_logger,
        epoch=epoch
    )
    
    # 获取当前学习率
    current_lr = optimizer.param_groups[0]['lr']
    
    # 记录 epoch 级别指标
    if swanlab_logger:
        swanlab_logger.log_epoch_metrics(
            epoch=epoch,
            train_loss=train_loss,
            train_text_loss=train_text_loss,    # [新增]
            train_soh_loss=train_soh_loss,      # [新增]
            train_acc=train_acc,
            val_loss=test_loss,
            val_text_loss=test_text_loss,       # [新增]
            val_soh_loss=test_soh_loss,         # [新增]
            val_mae=test_mae,
            val_rmse=test_rmse,
            val_r2=test_r2,
            lr=current_lr,
        )
    
    # 更新学习率
    scheduler.step()
    