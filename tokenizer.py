class Tokenizer:
    def __init__(self):
        self.move_dict = create_move_dict()
        self.inverse_dict = inverse_move_dict(self.move_dict)

    def tokenize_game(self, moves_list):
        tokenized_moves = []
        for move in moves_list:
            tokenized_moves.append(self.move_dict[move])
        return tokenized_moves
    
    def untokenize_game(self, tokenized_moves):
        inverse_moves = []
        for move in tokenized_moves:
            if move == 2064:
                inverse_moves.append("[pad]")
                continue
            if move == 2065:
                inverse_moves.append("[start]")
                continue
            inverse_moves.append(self.inverse_dict[move])
        return inverse_moves
    
    def tokenize_move(self, move):
        return self.move_dict[move]
    
    def get_move(self, tokenized_move):
        return self.inverse_dict[tokenized_move]


# Helper function to convert square index to algebraic notation
def square_to_algebraic(square):
    files = 'abcdefgh'
    ranks = '12345678'
    file = files[square % 8]
    rank = ranks[square // 8]
    return file + rank

# Modified chess_moves function to account for all moves
def chess_moves(starting_square):
    moves = []
    ss = starting_square

    # Calculate file and rank
    file_start = (ss // 8) * 8
    file_end = file_start + 7

    # Horizontal moves - to left
    for i in range(ss - 1, file_start - 1, -1):
        moves.append((ss, i))

    # Horizontal moves - to right
    for i in range(ss + 1, file_end + 1):
        moves.append((ss, i))

    # Vertical moves - above
    for i in range(ss + 8, 64, 8):
        moves.append((ss, i))

    # Vertical moves - below
    for i in range(ss - 8, -1, -8):
        moves.append((ss, i))

    # Diagonal moves
    # Upper left
    i = ss
    while (i := i + 7) < 64 and i % 8 != 7:
        moves.append((ss, i))

    # Lower left
    i = ss
    while (i := i - 9) >= 0 and i % 8 != 7:
        moves.append((ss, i))

    # Upper right
    i = ss
    while (i := i + 9) < 64 and i % 8 != 0:
        moves.append((ss, i))

    # Lower right
    i = ss
    while (i := i - 7) >= 0 and i % 8 != 0:
        moves.append((ss, i))

    # Inner 5x5 square
    for j in range(-2, 3):
        for i in range(-2, 3):
            target = ss + i + j * 8
            if 0 <= target < 64 and (target // 8 == (ss // 8) + j) and target != ss:
                moves.append((ss, target))

    # Pawn moves (including promotions)
    if ss // 8 == 1:  # White pawn's initial position
        if ss + 8 < 64:
            moves.append((ss, ss + 8))
            if (ss + 16) < 64:
                moves.append((ss, ss + 16))
        if ss + 9 < 64 and (ss + 9) % 8 != 0:
            moves.append((ss, ss + 9))
        if ss + 7 < 64 and (ss + 7) % 8 != 7:
            moves.append((ss, ss + 7))
    elif ss // 8 == 6:  # Black pawn's initial position
        if ss - 8 >= 0:
            moves.append((ss, ss - 8))
            if (ss - 16) >= 0:
                moves.append((ss, ss - 16))
        if ss - 9 >= 0 and (ss - 9) % 8 != 7:
            moves.append((ss, ss - 9))
        if ss - 7 >= 0 and (ss - 7) % 8 != 0:
            moves.append((ss, ss - 7))

    #remove duplicate tuples
    seen = set()
    result = []
    for item in moves:
        if item not in seen:
            seen.add(item)
            result.append(item)

    return result


# Function to create a dictionary of moves with promotion
def create_move_dict():
    move_dict = {}
    count = 0
    promotion_pieces = ['q', 'r', 'b', 'n']  # Queen, Rook, Bishop, Knight

    for i in range(64):
        for move in chess_moves(i):
            start_sq_algebraic = square_to_algebraic(move[0])
            end_sq_algebraic = square_to_algebraic(move[1])
            move_dict[f"{start_sq_algebraic}{end_sq_algebraic}"] = count
            count += 1
            # Add promotions if applicable
            if move[1] // 8 == 7 and i // 8 == 6:  # White pawn reaching last rank
                for piece in promotion_pieces:
                    move_dict[f"{start_sq_algebraic}{end_sq_algebraic}{piece}"] = count
                    count += 1
            elif move[1] // 8 == 0 and i // 8 == 1:  # Black pawn reaching last rank
                for piece in promotion_pieces:
                    move_dict[f"{start_sq_algebraic}{end_sq_algebraic}{piece}"] = count
                    count += 1

    move_dict["pad"] = 2064
    move_dict["start"] = 2065
    return move_dict

def inverse_move_dict(move_dict):
    inverse_dict = {}
    for k, v in move_dict.items():
        inverse_dict[v] = k
    return inverse_dict

def tokenize_game(moves_list):
    move_dict = create_move_dict()
    tokenized_moves = []
    for move in moves_list:
        tokenized_moves.append(move_dict[move])
    return tokenized_moves

if __name__ == "__main__":
    t = Tokenizer()