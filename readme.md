# Chess Transformer

This repository contains a chess engine implementation using a Transformer-based neural network architecture. The model is trained on chess game data and can play chess against humans or other engines.

## Model Architecture

The ChessTransformer takes inspiration from LLMs by treating chess as a sequence prediction task and tokenizing all possible unique legal moves on the board. This gives a vocabulary of 2064 unique moves + a start token + a padding token (used during training).

The model implements the classic multi-headed attention and feed-forward layers in addition to layer normalzation, stochastic depth, and residuals to facilitate deeper networks.

The final version of the model has 64 layers, a `d_model` of 1024, ~810M parameters, and took 23 hours to train on an RTX 4090 at a batch size of 10 with 12 accumulaiton steps.

### Weighted Loss Function

The loss function assigns higher weight to moves made by the winning player in each game. This encourages the model to learn patterns and strategies that lead to victory. However, losing moves are not ignored entirely. A smaller weight is assigned to moves made by the losing player, ensuring the model is 'aware' of what strategies the winning moves succesfully counter.

By weighting White and Black's winning moves, (to borrow terminology from LLMs) we combine the 'pretraining' and 'finetuning', producing a model that predicts the next *winning* move, as opposed to the most likely move.

## Project Structure

- `chesstransformer.py`: Contains the implementation of the ChessTransformer model.
- `train.py`: Script for training the ChessTransformer model.
- `autoplay_multiproc.py`: Script for playing multiple games in parallel against Stockfish to evaluate the model's performance.
- `play.py`: Interactive script for playing chess against the trained model.
- `processdata.py`: Script for processing and tokenizing chess game data.
- `tokenizer.py`: Implements the tokenization of chess moves.
- `environment.yml`: Conda environment file for setting up the project dependencies.

## Setup

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/chess-transformer.git
   cd chess-transformer
   ```

2. Create and activate the Conda environment:
   ```
   conda env create -f environment.yml
   conda activate chessbot
   ```
3. You can download the model from https://huggingface.co/RingoDingo/chess_transformer

4. Download the Stockfish chess engine and update the path in `autoplay_multiproc.py` if necessary.

## Usage

### Training the Model

To train the model, use the `train.py` script:

```
python train.py --train_data_path ./processed/train_data.parquet --val_data_path ./processed/val_data.parquet
```

Use `python train.py --help` to see all available options.

### Evaluating the Model

To evaluate the model against Stockfish, use the `autoplay_multiproc.py` script:

```
python autoplay_multiproc.py --games 100 --stockfish_elo 1500
```

### Playing Against the Model

To play chess against the trained model, run:

```
python play.py
```

Follow the prompts to enter your moves in UCI format (e.g., 'e2e4').

### Processing Chess Data

To process and tokenize chess game data, use the `processdata.py` script:

```
python processdata.py path/to/your/chess_data.parquet
```

## Training Data

The model is trained on self-play chess data. Thanks to [mitermix](https://huggingface.co/datasets/mitermix/chess-selfplay/tree/main) for providing the self-play dataset.

The final model was trained on sets 4-10 (inclusive), which is ~5M games.

## Logs and Experiments

Training logs and experiment results can be found in the `./tensorboard_logs` directory. Use TensorBoard to visualize the training progress and performance metrics.

```
tensorboard --logdir ./tensorboard_logs/final_architecture
```
