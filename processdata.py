import pandas as pd
import argparse
from tokenizer import Tokenizer

t = Tokenizer()

## Thank you to mitermix for the self play data
## https://huggingface.co/datasets/mitermix/chess-selfplay/tree/main

def process_chess_data(file_path):
    # Step 1: Read the Parquet file
    print("Loading file: " + file_path + "!")
    df = pd.read_parquet(file_path)
    print("File loaded!")

    # Step 2: Filter the DataFrame
    df_filtered = df[df['Termination'] == 'CHECKMATE']
    df_filtered = df_filtered[df_filtered['Moves'].apply(lambda x: len(x) <= 200)]
    df_filtered = df_filtered.drop(columns=['Termination'])
    # change result from '1-0' to 1, '0-1' to 0
    df_filtered['Result'] = df_filtered['Result'].apply(lambda x: 1 if x == '1-0' else 0)

    # Step 3: Process the data in parallel
    print("Encoding games!")
    df_filtered['Moves'] = df_filtered['Moves'].apply(lambda x: t.tokenize_game(x))
    #prepend 2065 as start token
    df_filtered['Moves'] = df_filtered['Moves'].apply(lambda x: [2065] + x)
    
    print("Exporting!")
    df_filtered.to_parquet('./processed_game_0003.parquet', engine='pyarrow', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process chess data')
    parser.add_argument('file_path', type=str, help='Path to the Parquet file')
    args = parser.parse_args()
    process_chess_data(args.file_path)
