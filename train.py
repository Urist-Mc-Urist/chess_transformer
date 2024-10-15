import torch
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import argparse
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import logging
import pandas as pd

from chesstransformer import ChessTransformer, winning_moves_loss, all_moves_loss, weighted_chess_loss

class ChessDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        moves = self.data.iloc[idx]['Moves']
        result = self.data.iloc[idx]['Result']
        return torch.tensor(moves, dtype=torch.long), torch.tensor(result, dtype=torch.long)

def collate_fn(batch):
    moves, results = zip(*batch)
    padded_moves = pad_sequence(moves, batch_first=True, padding_value=2064)
    results = torch.stack(results)
    return padded_moves, results

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set up argument parser
parser = argparse.ArgumentParser(description='Chess Transformer Training')
parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate')
parser.add_argument('--max_lr', type=float, default=1e-3, help='Maximum learning rate for OneCycleLR')
parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for AdamW')
parser.add_argument('--num_epochs', type=int, default=2, help='Number of epochs')
parser.add_argument('--accumulation_steps', type=int, default=4, help='Gradient accumulation steps')
parser.add_argument('--early_stopping_patience', type=int, default=50, help='Early stopping patience')
parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Max gradient norm for clipping')
parser.add_argument('--log_dir', type=str, default='./logs', help='TensorBoard log directory')
parser.add_argument('--model_save_path', type=str, default='best_chess_transformer.pth', help='Path to save best model')
parser.add_argument('--val_every', type=int, default=100, help='Validation frequency')
parser.add_argument('--batch_train', type=int, default=32, help='Batch size for training')
parser.add_argument('--batch_val', type=int, default=32, help='Batch size for validation')
parser.add_argument('--cur_step', type=int, default=0, help='Current step')
parser.add_argument('--optimizer_betas', type=float, nargs=2, default=(0.9, 0.999), help='Betas for AdamW optimizer')
parser.add_argument('--optimizer_eps', type=float, default=1e-8, help='Eps for AdamW optimizer')
parser.add_argument('--pct_start', type=float, default=0.35, help='Percentage of the cycle to increase LR')
parser.add_argument('--anneal_strategy', type=str, default='cos', choices=['cos', 'linear'], help='Annealing strategy')
parser.add_argument('--div_factor', type=float, default=50.0, help='Initial LR = max_lr/div_factor')
parser.add_argument('--final_div_factor', type=float, default=1e4, help='Final LR = max_lr/final_div_factor')
parser.add_argument('--resume', action='store_true', help='Resume training from a checkpoint')
parser.add_argument('--checkpoint_path', type=str, default='./best_chess_transformer.pth', help='Path to checkpoint file')
parser.add_argument('--losing_weight', type=float, default=0.1, help='Weight for losing moves')
parser.add_argument('--train_data_path', type=str, help='Path to training data parquet')
parser.add_argument('--val_data_path', type=str, help='Path to validation data parquet')
parser.add_argument('--model_layers', type=int, default=64, help='Number of layers in the model')
parser.add_argument('--model_dim', type=int, default=1024, help='Model dimension')

args = parser.parse_args()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# Load data
dataframe_train = pd.read_parquet(args.train_data_path)
dataframe_val = pd.read_parquet(args.val_data_path)

# Create data loaders
train_loader = DataLoader(
    ChessDataset(dataframe_train),
    batch_size=args.batch_train,
    shuffle=True,
    collate_fn=collate_fn
)
val_loader = DataLoader(
    ChessDataset(dataframe_val),
    batch_size=args.batch_val,
    shuffle=False,
    collate_fn=collate_fn
)

# Calculate total steps for OneCycleLR
total_steps = (len(train_loader) // args.accumulation_steps)

# Set up model, optimizer, scheduler
model = ChessTransformer(args.model_layers, args.model_dim).to(device) # can specify model size here
#model.load_state_dict(torch.load('./best_chess_transformer.pth')["model_state_dict"])

optimizer = optim.AdamW(
    model.parameters(),
    lr=args.max_lr / args.div_factor,
    betas=tuple(args.optimizer_betas),
    eps=args.optimizer_eps,
    weight_decay=args.weight_decay
)
scheduler = OneCycleLR(
    optimizer,
    max_lr=args.max_lr,
    total_steps=total_steps,
    pct_start=args.pct_start,
    anneal_strategy=args.anneal_strategy,
    div_factor=args.div_factor,
    final_div_factor=args.final_div_factor
)

# Load from checkpoint if resuming
if args.resume:
    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    steps = torch.load(args.checkpoint_path)["steps"]
    best_val_loss = checkpoint['best_val_loss']
    early_stopping_counter = 0
    start_epoch = checkpoint['epoch']
    logging.info(f"Resuming training from epoch {start_epoch} and step {steps}")
else:
    start_epoch = 0
    steps = args.cur_step
    early_stopping_counter = 0
    best_val_loss = float('inf')

scaler = torch.amp.GradScaler('cuda') # May throw an error for old PyTorch versions

# Training loop
writer = SummaryWriter(log_dir=args.log_dir)

for epoch in range(start_epoch, args.num_epochs):
    #reset scheduler to start of cycle
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.max_lr,
        total_steps=total_steps,
        pct_start=args.pct_start,
        anneal_strategy=args.anneal_strategy,
        div_factor=args.div_factor,
        final_div_factor=args.final_div_factor
    )
    model.train()
    train_loss = 0.0
    train_batches = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} [Train]")
    
    for batch_idx, (moves, results) in enumerate(train_batches):
        moves, results = moves.to(device), results.to(device)
        
        with torch.amp.autocast('cuda'):
            output = model(moves)
            #loss = all_moves_loss(output=output, ground_truth=moves)
            #loss = winning_moves_loss(output, moves, results)
            loss = weighted_chess_loss(output, moves, results, winning_weight=1.0, losing_weight=args.losing_weight)
        
        loss = loss / args.accumulation_steps
        scaler.scale(loss).backward()
        
        if (batch_idx + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
        
        train_loss += loss.item() * args.accumulation_steps  # Multiply back to get actual loss
        train_batches.set_postfix({'loss': f'{loss.item() * args.accumulation_steps:.4f}'})
        writer.add_scalar('Loss/train_step', loss.item() * args.accumulation_steps, steps)
        current_lr = scheduler.get_last_lr()[0]
        writer.add_scalar('Learning Rate', current_lr, steps)

        steps += 1

        if steps % args.val_every == 0:
            # Validation loop
            model.eval()
            val_loss = 0.0
            val_batches = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} [Val]")
            
            with torch.no_grad():
                for moves, results in val_batches:
                    moves, results = moves.to(device), results.to(device)
                    output = model(moves)
                    #loss = all_moves_loss(output=output, ground_truth=moves)
                    loss = winning_moves_loss(output, moves, results)
                    val_loss += loss.item()
                    val_batches.set_postfix({'loss': f'{loss.item():.8f}'})
            
            avg_val_loss = val_loss / len(val_loader)
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                early_stopping_counter = 0
                torch.save({
                    'epoch': epoch,
                    'steps': steps,
                    'early_stopping_counter': early_stopping_counter,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_loss': best_val_loss,
                }, args.model_save_path)
                logging.info(f"Saved best model with validation loss: {best_val_loss:.8f}")
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= args.early_stopping_patience:
                    logging.info("Early stopping triggered")
                    break
            
            logging.info(f'Steps: {steps} - Val Loss: {avg_val_loss:.8f}')
            writer.add_scalar('Validation Loss/steps', avg_val_loss, steps)

            model.train()
    
    avg_train_loss = train_loss / len(train_loader)
    writer.add_scalar('Loss/train_epoch', avg_train_loss, epoch)
    logging.info(f'Epoch {epoch+1}/{args.num_epochs} - Train Loss: {avg_train_loss:.8f}')

writer.close()
logging.info("Training completed")
