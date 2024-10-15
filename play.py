import torch
import torch.nn.functional as F
from chesstransformer import ChessTransformer
import tokenizer as tk

model = ChessTransformer()
model.load_state_dict(torch.load('./64L1024D_1e-3maxlr_470k_step_1ep_1480ELO.pth')["model_state_dict"])
model.eval().cuda()

# Initialize tokenizer
t = tk.Tokenizer()

def predict_move(model, game_sequence, tokenizer, device='cuda', top_k=5):
    model.eval()
    game_sequence = torch.tensor([tokenizer.tokenize_game(game_sequence)], dtype=torch.long).to(device)
    
    with torch.no_grad():
        output = model(game_sequence)
        logits = output[0, -1, :]  # Get logits for the last move
        top_k_logits, top_k_indices = torch.topk(logits, top_k)
        
        # Apply softmax to get probabilities
        probs = F.softmax(top_k_logits, dim=-1)
        
        # Sample from the probability distribution
        sampled_index = torch.multinomial(probs, 1).item()
        sampled_token = top_k_indices[sampled_index].item()
        
        sampled_move = tokenizer.untokenize_game([sampled_token])[0]
        
        # Get all top_k moves and their probabilities for display
        top_k_moves = [tokenizer.untokenize_game([idx.item()])[0] for idx in top_k_indices]
        top_k_probs = probs.cpu().numpy()
        
    return sampled_move, top_k_moves, top_k_probs

def play_game():
    input_game = []
    print("Let's play chess! Enter your moves in UCI format (e.g., 'e2e4'). Type 'exit' to quit or 'undo' to undo the last move.")

    while True:
        user_move = input("Your move: ").strip()
        if user_move.lower() == 'exit':
            print("Game over. Thanks for playing!")
            break
        elif user_move.lower() == 'undo':
            if len(input_game) >= 2:
                input_game.pop()  # Remove bot's move
                input_game.pop()  # Remove user's move
                print("Last move undone. Current game sequence:", input_game)
            else:
                print("Cannot undo. No moves to undo.")
            continue
        
        input_game.append(user_move)
        print("Current game sequence:", input_game)

        try:
            bot_move, top_moves, top_probs = predict_move(model, input_game, t)
            
            # Display top moves and their probabilities
            moves_probs_str = ', '.join(f"{move} ({prob:.2%})" for move, prob in zip(top_moves, top_probs))
            print(f"Top {len(top_moves)} moves and probabilities: {moves_probs_str}")
            
            print(f"Bot's sampled move: {bot_move}")
            input_game.append(bot_move)
        except Exception as e:
            print("An error occurred:", e)
            break

if __name__ == "__main__":
    play_game()