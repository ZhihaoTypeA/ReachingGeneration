import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, Tuple, Optional
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
import json
import random

class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)

        return emb

#Condition encoding
class ConditionEncoder(nn.Module):
    def __init__(self, d_model, time_emb_dim=128):
        super().__init__()
        self.d_model = d_model

        #Multiple encoder for each type of obs
        self.joint_encoder = nn.Sequential(
            nn.Linear(7, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, d_model)
        )

        self.target_encoder = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, d_model)
        )
        
        self.obstacle_encoder = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, d_model)
        )

        #Time position embedding
        self.time_embedding = SinusoidalPositionEmbedding(time_emb_dim)
        self.time_proj = nn.Linear(time_emb_dim, d_model)

        #Obs type position embedding (joint, target position, obstacle position, time position)
        self.condition_pos_embedding = nn.Parameter(torch.randn(4, d_model))

        self.norm = nn.LayerNorm(d_model)

    def forward(self, obs, timestep):  
        batch_size = obs.size(0)

        joint_pos = obs[:, :7]
        target_pos = obs[:, 7:10]
        obstacle_pos = obs[:, 10:13]

        joint_feature = self.joint_encoder(joint_pos) #(batch, d_model)
        target_feature = self.target_encoder(target_pos)
        obstacle_feature = self.obstacle_encoder(obstacle_pos)
        
        time_emb = self.time_embedding(timestep) #(batch, time_embed_dim)
        time_feature = self.time_proj(time_emb) #(batch, d_model)

        condition = torch.stack([joint_feature, target_feature, obstacle_feature, time_feature], dim=1) #(batch, 4, d_model)

        condition = condition + self.condition_pos_embedding.unsqueeze(0)
        condition = self.norm(condition)

        return condition

#Condition injection
class CrossAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        #Q是actions，KV是obs(condition), 把输入最后一维拆成多头进行处理, 之后调整形状保持(batch, head, seq_length, 每个头的通道数d_k)
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        context = torch.matmul(attention_weights, V)
        
        #(batch, head, seq_length, d_k) -> (batch, seq_length, head, d_k) -> (batch, seq_length, d_model)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.w_o(context)

        return output, attention_weights

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        #Self-attention (for actions itself)
        self.self_attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)

        #Cross-attention (for conditioning)
        self.cross_attention = CrossAttention(d_model, num_heads, dropout=dropout)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, condition, attention_mask=None):
        #Self-attention
        residual = x
        x = self.norm1(x)
        self_attn_output, _ = self.self_attention(x, x, x, attn_mask=attention_mask)
        x = residual + self.dropout(self_attn_output)

        #Cross-attention
        residual = x
        x = self.norm2(x)
        cross_attn_output, _ = self.cross_attention(x, condition, condition)
        x = residual + self.dropout(cross_attn_output)

        #FF
        residual = x
        x = self.norm3(x)
        ff_output = self.feed_forward(x)
        x = residual + ff_output
        
        return x

class DiffusionPolicyNetwork(nn.Module):
    def __init__(
            self,
            obs_dim: int = 13,
            action_dim: int = 7,
            action_horizon: int = 8,
            d_model: int = 512,
            num_heads: int = 8,
            num_layers: int = 6,
            d_ff: int = 2048,
            dropout: float = 0.1,
            time_emb_dim: int = 128
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.d_model = d_model

        self.condition_encoder = ConditionEncoder(d_model, time_emb_dim)

        self.action_embeddfing = nn.Linear(action_dim, d_model)

        #Action position (for learning step in horizon sequence)
        self.action_pos_embedding = nn.Parameter(torch.randn(action_horizon, d_model))

        self.transformer_layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])

        self.output_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, action_dim)

        #Init parameters
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Parameter):
            nn.init.normal_(module, std=0.02)

    def forward(self, noisy_actions, obs, timestep):
        batch_size = noisy_actions.size(0)

        condition = self.condition_encoder(obs, timestep) #(batch, 4, d_model)

        x = self.action_embeddfing(noisy_actions) #(batch, action_horizon, d_model)
        
        x = x + self.action_pos_embedding.unsqueeze(0)

        for layer in self.transformer_layers:
            x = layer(x, condition)

        x = self.output_norm(x)
        predicted_noise = self.output_proj(x) #(batch, action_horizon, action_dim)

        return predicted_noise

class DiffusionPolicy(nn.Module):
    def __init__(
        self,
        network: DiffusionPolicyNetwork,
        num_diffusion_steps: int = 100,
        beta_schedule: str = "cosine"
    ):
        super().__init__()
        self.network = network
        self.num_diffusion_steps = num_diffusion_steps

        #Noise schedule
        if beta_schedule == "linear":
            betas = torch.linspace(0.0001, 0.02, num_diffusion_steps)
        elif beta_schedule == "cosine":
            s = 0.008
            steps = num_diffusion_steps + 1
            x = torch.linspace(0, num_diffusion_steps, steps)
            alphas_cumprod = torch.cos(((x / num_diffusion_steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clamp(betas, 0.0001, 0.9999)

        self.register_buffer('betas', betas)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        #Forward noise addition param
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        #Backward denoising param
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))


    def add_noise(self, original_samples, noise, timesteps):
        sqrt_alpha_cumprod = self.sqrt_alphas_cumprod[timesteps]
        sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alphas_cumprod[timesteps]

        while len(sqrt_alpha_cumprod.shape) < len(original_samples.shape):
            sqrt_alpha_cumprod = sqrt_alpha_cumprod.unsqueeze(-1)
            sqrt_one_minus_alpha_cumprod = sqrt_one_minus_alpha_cumprod.unsqueeze(-1)

        noisy_samples = (sqrt_alpha_cumprod * original_samples + 
                        sqrt_one_minus_alpha_cumprod * noise)
        return noisy_samples
    
    def training_step(self, batch):
        observations = batch['observations']  #(batch, obs_dim)
        actions = batch['actions']  #(batch, action_horizon, action_dim)
        
        batch_size = actions.shape[0]
        device = actions.device

        timesteps = torch.randint(0, self.num_diffusion_steps, (batch_size,), device=device)
        noise = torch.randn_like(actions)
        noisy_actions = self.add_noise(actions, noise, timesteps)
        predicted_noise = self.network(noisy_actions, observations, timesteps)
        loss = F.mse_loss(predicted_noise, noise)

        return loss
    
    def ddim_step(self, sample, model_output, timestep, next_timestep, eta=0.0):
        """
        修正的DDIM采样步骤
        eta=0.0 表示确定性采样（标准DDIM）
        """
        alpha_cumprod = self.alphas_cumprod[timestep]
        alpha_cumprod_next = self.alphas_cumprod[next_timestep] if next_timestep >= 0 else torch.tensor(1.0)
        
        beta_cumprod = 1 - alpha_cumprod
        beta_cumprod_next = 1 - alpha_cumprod_next
        
        # 预测原始样本 x_0
        pred_original_sample = (sample - torch.sqrt(beta_cumprod) * model_output) / torch.sqrt(alpha_cumprod)
        
        # 计算方差
        variance = (beta_cumprod_next / beta_cumprod) * (1 - alpha_cumprod / alpha_cumprod_next)
        std_dev_t = eta * torch.sqrt(variance)
        
        # 计算"direction pointing to x_t"
        pred_sample_direction = torch.sqrt(1 - alpha_cumprod_next - std_dev_t**2) * model_output
        
        # 计算 x_{t-1}
        pred_sample = torch.sqrt(alpha_cumprod_next) * pred_original_sample + pred_sample_direction
        
        if eta > 0:
            noise = torch.randn_like(sample)
            pred_sample = pred_sample + std_dev_t * noise
        
        return pred_sample

    @torch.no_grad()
    def sample(self, obs, num_inference_steps=50, generator=None, eta=0.0):
        """修正的采样方法"""
        device = obs.device
        batch_size = obs.shape[0]
        
        shape = (batch_size, self.network.action_horizon, self.network.action_dim)
        actions = torch.randn(shape, device=device, generator=generator)
        
        # 修正时间步生成方式
        step_ratio = self.num_diffusion_steps // num_inference_steps
        timesteps = list(range(0, self.num_diffusion_steps, step_ratio))[:num_inference_steps]
        timesteps = timesteps[::-1]  # 从大到小
        timesteps = torch.tensor(timesteps, dtype=torch.long, device=device)
        
        for i, timestep in enumerate(timesteps):
            t = timestep.expand(batch_size)
            
            # 预测噪声
            noise_pred = self.network(actions, obs, t)
            
            # 确定下一个时间步
            next_timestep = timesteps[i + 1] if i < len(timesteps) - 1 else -1
            
            # DDIM步骤
            if next_timestep >= 0:
                actions = self.ddim_step(actions, noise_pred, timestep, next_timestep, eta)
            else:
                # 最终步骤：直接预测x_0
                alpha_cumprod = self.alphas_cumprod[timestep]
                actions = (actions - torch.sqrt(1 - alpha_cumprod) * noise_pred) / torch.sqrt(alpha_cumprod)
        
        return actions

#Load data from npz file generated by SLSQP
class DiffusionPolicyDataset(torch.utils.data.Dataset):
    def __init__(self, data_file, scaler_file=None, metadata_file=None):
        data = np.load(data_file)
        self.observations = torch.from_numpy(data['observations']).float()
        self.actions = torch.from_numpy(data['actions']).float()

        print(f"Loaded dataset with {len(self.observations)} samples")
        print(f"Obs shape: {self.observations.shape}")
        print(f"Action shape: {self.actions.shape}")

    def __len__(self):
        return len(self.observations)
    
    def __getitem__(self, idx):
        return {
            'observations': self.observations[idx],
            'actions': self.actions[idx]
        }

class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-6, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_epoch = 0
        self.best_weights = None

    def __call__(self, val_loss, model, epoch):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_epoch = epoch
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                    print(f"Early stopping triggered. Restored weights from epoch {self.best_epoch}")
                return True
            return False

    def get_best_info(self):
        return {
            'best_loss': self.best_loss,
            'best_epoch': self.best_epoch,
            'patience_counter': self.counter
        }
        
def train_model(
    train_data_file: str,
    test_data_file: str,
    save_dir: str = "./models",
    scaler_file: str = None,
    metadata_file: str = None,
    config: Optional[Dict] = None
):
    default_config = {
        'batch_size': 128,
        'learning_rate': 1e-4,
        'num_epochs': 150,
        'num_diffusion_steps': 200,
        'beta_schedule': 'cosine',
        'num_inference_steps': 50,
        'gradient_clip': 1.0,
        'save_interval': 20,
        'eval_interval': 5,
        'warmup_steps': 1000,
        'early_stopping_patience': 5,
        'early_stopping_min_delta': 1e-6,
        'early_stopping_restore_weights': True,
        
        'obs_dim': 13,
        'action_dim': 7,
        'action_horizon': 15,
        'd_model': 512,
        'num_heads': 8,
        'num_layers': 6,
        'd_ff': 1024,
        'dropout': 0.1,
        'time_emb_dim': 128
    }

    if config is not None:
        default_config.update(config)
    config = default_config

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    log_dir = save_dir / "tensorboard_logs"
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logs will be saved to: {log_dir}")

    print("-Train dataset-")
    train_dataset = DiffusionPolicyDataset(train_data_file)
    print("-Test dataset-")
    test_dataset = DiffusionPolicyDataset(test_data_file)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=4
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=4
    )
    
    network = DiffusionPolicyNetwork(
        obs_dim=config['obs_dim'],
        action_dim=config['action_dim'],
        action_horizon=config['action_horizon'],
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        d_ff=config['d_ff'],
        dropout=config['dropout'],
        time_emb_dim=config['time_emb_dim']
    ).to(device)
    
    policy = DiffusionPolicy(
        network, 
        num_diffusion_steps=config['num_diffusion_steps'],
        beta_schedule=config['beta_schedule']
    ).to(device)

    optimizer = torch.optim.AdamW(policy.parameters(), lr=config['learning_rate'], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['num_epochs'])

    early_stopping = EarlyStopping(
        patience=config['early_stopping_patience'],
        min_delta=config['early_stopping_min_delta'],
        restore_best_weights=config['early_stopping_restore_weights']
    )

    config_with_scaler = config.copy()
    config_with_scaler['scaler_file'] = str(scaler_file) if scaler_file else None
    config_with_scaler['metadata_file'] = str(metadata_file) if metadata_file else None
    
    writer.add_text('Config', str(config_with_scaler))

    train_losses = []
    eval_losses = []
    learning_rates = []
    best_loss = float('inf')
    global_step = 0
    early_stopped = False

    print("Starting training...")
    for epoch in range(config['num_epochs']):
        policy.train()
        epoch_loss = 0
        num_batches = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["num_epochs"]} [Train]')
        for batch in train_pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            #Forward
            loss = policy.training_step(batch)
            epoch_loss += loss.item()
            num_batches += 1
            
            #Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), config['gradient_clip'])
            optimizer.step()
            
            writer.add_scalar('Loss/Train_Step', loss.item(), global_step)
            writer.add_scalar('Learning_Rate', scheduler.get_last_lr()[0], global_step)

            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}', 
                'avg_loss': f'{epoch_loss/num_batches:.4f}'
            })

            global_step += 1

        train_pbar.close()
        
        avg_train_loss = epoch_loss / num_batches
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        writer.add_scalar('Loss/Train_Epoch', avg_train_loss, epoch)
        writer.add_scalar('Learning_Rate_Epoch', current_lr, epoch)

        train_losses.append(avg_train_loss)
        learning_rates.append(current_lr)

        #Evaluate
        if epoch % config['eval_interval'] == 0:
            policy.eval()
            eval_loss = 0
            eval_batches = 0
            
            eval_pbar = tqdm(test_loader, desc=f'Epoch {epoch+1}/{config["num_epochs"]} [Eval]')
            with torch.no_grad():
                for batch in eval_pbar:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    loss = policy.training_step(batch)
                    eval_loss += loss.item()
                    eval_batches += 1

                    eval_pbar.set_postfix({'eval_loss': f'{loss.item():.4f}'})
            
            eval_pbar.close()
            
            avg_eval_loss = eval_loss / eval_batches
            eval_losses.append(avg_eval_loss)
            writer.add_scalar('Loss/Eval', avg_eval_loss, epoch)
            
            should_stop = early_stopping(avg_eval_loss, policy, epoch)
            best_info = early_stopping.get_best_info()
            writer.add_scalar('EarlyStopping/BestLoss', best_info['best_loss'], epoch)
            writer.add_scalar('EarlyStopping/PatienceCounter', best_info['patience_counter'], epoch)

            print(f"Epoch {epoch:03d} | Train Loss: {avg_train_loss:.4f} | Eval Loss: {avg_eval_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f} | Best Loss: {best_info['best_loss']:.4f} | Patience: {best_info['patience_counter']}/{config['early_stopping_patience']}")
            
            #Save best
            if avg_eval_loss < best_loss:
                best_loss = avg_eval_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': policy.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': config_with_scaler,
                    'loss': best_loss
                }, save_dir / 'best_model.pth')
                writer.add_text('Best_Model', f'New best model at epoch {epoch} with eval loss {best_loss:.4f}')

            if should_stop:
                print(f"\nEarly stopping triggered at epoch {epoch}")
                print(f"Best validation loss: {best_info['best_loss']:.4f} at epoch {best_info['best_epoch']}")
                early_stopped = True
                break

        else:
            eval_losses.append(None)
        
        #Auto save
        if epoch % config['save_interval'] == 0 and epoch > 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': policy.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config_with_scaler,
                'loss': avg_train_loss
            }, save_dir / f'checkpoint_epoch_{epoch}.pth')

    #Final save if training completed without early stopping
    if not early_stopped:
        torch.save({
            'epoch': epoch,
            'model_state_dict': policy.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config_with_scaler,
            'loss': avg_train_loss
        }, save_dir / 'final_model.pth')
    
    print("Training completed!")
    if early_stopped:
        best_info = early_stopping.get_best_info()
        print(f"Training stopped early. Best model from epoch {best_info['best_epoch']} with loss {best_info['best_loss']:.4f}")

    writer.close()
    print(f"Training results saved to: {save_dir}")
    print(f"View TensorBoard logs with: tensorboard --logdir {log_dir}")

def load_and_infer(
    model_path: str,
    data_file: str,
    num_inference_steps: int = 50,
    device: str = None
):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Loading model from: {model_path}")
    print(f"Using device: {device}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    
    # Create model
    network = DiffusionPolicyNetwork(
        obs_dim=config['obs_dim'],
        action_dim=config['action_dim'],
        action_horizon=config['action_horizon'],
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        d_ff=config['d_ff'],
        dropout=config['dropout'],
        time_emb_dim=config['time_emb_dim']
    ).to(device)
    
    policy = DiffusionPolicy(
        network, 
        num_diffusion_steps=config['num_diffusion_steps'],
        beta_schedule=config['beta_schedule']
    ).to(device)
    
    # Load weights
    policy.load_state_dict(checkpoint['model_state_dict'])
    policy.eval()
    
    print(f"Model loaded successfully from epoch {checkpoint['epoch']}")
    
    # Load dataset
    print(f"\nLoading dataset from: {data_file}")
    dataset = DiffusionPolicyDataset(data_file)
    
    # Randomly select a sample
    random_idx = random.randint(0, len(dataset) - 1)
    sample = dataset[random_idx]
    
    observations = sample['observations'].unsqueeze(0).to(device)  # Add batch dimension
    ground_truth_actions = sample['actions']
    
    print(f"\nRandom sample index: {random_idx}")
    print(f"Observation shape: {observations.shape}")
    print(f"Ground truth actions shape: {ground_truth_actions.shape}")
    
    # Perform inference
    print(f"\nPerforming inference with {num_inference_steps} steps...")
    with torch.no_grad():
        predicted_actions = policy.sample(
            observations, 
            num_inference_steps=num_inference_steps
        )
    
    # Convert to numpy for printing
    predicted_actions = predicted_actions.squeeze(0).cpu().numpy()  # Remove batch dimension
    ground_truth_actions = ground_truth_actions.numpy()
    
    # Print results
    print(f"\n{'='*60}")
    print("INFERENCE RESULTS")
    print(f"{'='*60}")
    
    print(f"\nObservation:")
    obs_np = observations.squeeze(0).cpu().numpy()
    print(f"  Joint positions (7): {obs_np[:7]}")
    print(f"  Target position (3): {obs_np[7:10]}")
    print(f"  Obstacle position (3): {obs_np[10:13]}")
    
    print(f"\nGround Truth Actions (shape: {ground_truth_actions.shape}):")
    for i, action_step in enumerate(ground_truth_actions):
        print(f"  Step {i:2d}: [{', '.join([f'{x:8.4f}' for x in action_step])}]")
    
    print(f"\nPredicted Actions (shape: {predicted_actions.shape}):")
    for i, action_step in enumerate(predicted_actions):
        print(f"  Step {i:2d}: [{', '.join([f'{x:8.4f}' for x in action_step])}]")
    
    # Calculate and print differences
    print(f"\nAction Differences (Predicted - Ground Truth):")
    differences = predicted_actions - ground_truth_actions
    for i, diff_step in enumerate(differences):
        print(f"  Step {i:2d}: [{', '.join([f'{x:8.4f}' for x in diff_step])}]")
    
    # Print statistics
    mse = np.mean(differences ** 2)
    mae = np.mean(np.abs(differences))
    max_abs_diff = np.max(np.abs(differences))
    
    print(f"\nStatistics:")
    print(f"  Mean Squared Error (MSE): {mse:.6f}")
    print(f"  Mean Absolute Error (MAE): {mae:.6f}")
    print(f"  Max Absolute Difference: {max_abs_diff:.6f}")
    
    print(f"\n{'='*60}")


if __name__ == "__main__":
    project_root_path = os.path.dirname(os.path.dirname(__file__))
    train_data_path = os.path.join(project_root_path, 'data', 'Dataset', 'train_data.npz')
    test_data_path = os.path.join(project_root_path, 'data', 'Dataset', 'test_data.npz')
    scaler_path = os.path.join(project_root_path, 'data', 'Dataset', 'action_scaler.pkl')
    metadata_path = os.path.join(project_root_path, 'data', 'Dataset', 'dataset_metadata.json')
    model_save_path = os.path.join(project_root_path, 'data', 'DPModel')

    #Train
    train_model(
        train_data_file=train_data_path,
        test_data_file=test_data_path,
        save_dir=model_save_path,
        scaler_file=scaler_path,
        metadata_file=metadata_path
    )

    #Test
    model_path = os.path.join(model_save_path, 'best_model.pth')
    load_and_infer(
        model_path=model_path,
        data_file=test_data_path,
        num_inference_steps=50
    )

