import torch
import chess
import chess.engine
import logging
import math
import argparse
import multiprocessing as mp
from chesstransformer import ChessTransformer
import tokenizer as tk
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='Chess Transformer Testing')
parser.add_argument('--cores', type=int, default=2, help='Cores to use for CPU chess engine')
parser.add_argument('--games', type=int, default=10, help='Number of games to play')
parser.add_argument('--stockfish_elo', type=int, default=1320, help='ELO rating for Stockfish. Min 1320')
parser.add_argument('--stockfish_path', type=str, default='./stockfish/stockfish-ubuntu-x86-64', help='Path to Stockfish binary')

args = parser.parse_args()

def setup_model():
    logger.info("Loading ChessTransformer model...")
    model = ChessTransformer()
    model.load_state_dict(torch.load('./64L1024D_1e-3maxlr_470k_step_1ep_1480ELO.pth')["model_state_dict"])
    model.eval().cuda()
    logger.info("Model loaded successfully.")
    return model

def predict_top_k_moves(model, tokenizer, game_sequence, k=100, device='cuda'):
    game_sequence = torch.tensor([tokenizer.tokenize_game(game_sequence)], dtype=torch.long).to(device)
    
    with torch.no_grad():
        output = model(game_sequence)
        next_move = output[0, -1, :]
        next_softmax = torch.nn.functional.softmax(next_move, dim=-1)
        top_k_probs, top_k_indices = torch.topk(next_softmax, k)
        top_k_moves = [tokenizer.get_move(idx.item()) for idx in top_k_indices]
    
    return list(zip(top_k_moves, top_k_probs.tolist()))

def get_legal_move(board, moves):
    for move, prob in moves:
        try:
            if chess.Move.from_uci(move) in board.legal_moves:
                return move, prob
        except ValueError:
            continue
    return None, None

def play_game(model, tokenizer, stockfish_path, stockfish_elo, model_is_white, game_number):
    #logger.info(f"Game {game_number}: Starting. Model playing as {'white' if model_is_white else 'black'}")
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    engine.configure({"UCI_LimitStrength": True, "UCI_Elo": stockfish_elo})
    
    board = chess.Board()
    game_sequence = ['start']
    move_count = 0
    
    while not board.is_game_over():
        move_count += 1
        if (board.turn == chess.WHITE) == model_is_white:
            top_k_moves = predict_top_k_moves(model, tokenizer, game_sequence)
            legal_move, prob = get_legal_move(board, top_k_moves)
            if legal_move is None:
                logger.warning(f"Game {game_number}: No legal moves found in top-k on move {move_count}. Game over.")
                return "0-1" if model_is_white else "1-0", move_count
            board.push_uci(legal_move)
            game_sequence.append(legal_move)
            logger.debug(f"Game {game_number}: Model's move: {legal_move} (probability: {prob:.4f})")
        else:
            result = engine.play(board, chess.engine.Limit(time=0.1))
            board.push(result.move)
            game_sequence.append(result.move.uci())
            logger.debug(f"Game {game_number}: Stockfish's move: {result.move.uci()}")
    
    engine.quit()
    result = board.result()
    #logger.info(f"Game {game_number}: Finished. Result: {result}. Total moves: {move_count}")
    return result, move_count

def worker(args):
    model, tokenizer, stockfish_path, stockfish_elo, game_number = args
    model_is_white = game_number % 2 == 0
    result, move_count = play_game(model, tokenizer, stockfish_path, stockfish_elo, model_is_white, game_number)
    return result, game_number, move_count

def calculate_elo_from_win_rate(win_rate, opponent_elo):
    """Calculate ELO based on win rate against an opponent."""
    if win_rate == 0:
        return float('-inf')
    if win_rate == 1:
        return float('inf')
    elo_diff = -400 * math.log10(1 / win_rate - 1)
    return opponent_elo + elo_diff

def main():
    mp.set_start_method('spawn')  # Set start method to 'spawn' for CUDA support
    
    num_games = args.games
    stockfish_elo = args.stockfish_elo
    stockfish_path = args.stockfish_path
    
    logger.info(f"Starting tournament: {num_games} games, Stockfish ELO: {stockfish_elo}")
    
    model = setup_model()
    tokenizer = tk.Tokenizer()
    
    num_processes = args.cores
    logger.info(f"Using {num_processes} CPU cores for parallel processing")
    
    tasks = [(model, tokenizer, stockfish_path, stockfish_elo, i) for i in range(num_games)]
    
    results = []
    with mp.Pool(processes=num_processes) as pool:
        with tqdm(total=num_games, desc="Games Progress") as pbar:
            for result in pool.imap_unordered(worker, tasks):
                results.append(result)
                pbar.update()
    
    # Process results
    wins = draws = losses = 0
    total_moves = 0
    for result, game_number, move_count in results:
        if result == "1-0" and game_number % 2 == 0:
            wins += 1
        elif result == "0-1" and game_number % 2 == 1:
            wins += 1
        elif result == "1/2-1/2":
            draws += 1
        else:
            losses += 1
        total_moves += move_count
    
    win_rate = (wins + 0.5 * draws) / num_games
    final_model_elo = calculate_elo_from_win_rate(win_rate, stockfish_elo)
    elo_change = final_model_elo - stockfish_elo
    
    logger.info("Tournament completed. Final results:")
    logger.info(f"Total games: {num_games}")
    logger.info(f"Wins: {wins}, Losses: {losses}, Draws: {draws}")
    logger.info(f"Win rate: {win_rate:.2%}")
    logger.info(f"Average moves per game: {total_moves/num_games:.2f}")
    logger.info(f"Stockfish ELO: {stockfish_elo}")
    logger.info(f"Final Model ELO: {final_model_elo:.2f}")
    logger.info(f"ELO Change: {elo_change:+.2f}")

if __name__ == "__main__":
    main()