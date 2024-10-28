import os
import random
import re
import nltk
from multiprocessing.pool import ThreadPool
import numpy as np
import argparse

def probability_to_ranges(prob_vector):
    
    range_vector = np.zeros(len(prob_vector))
    
    # Track cumulative probability
    # All non-zero values cover a number range (iterate with if p <= P)
    # Zero values are converted to -1 so are never picked
    cumulative_prob = 0.0
    for i in range(0, len(prob_vector)):
        if prob_vector[i] == 0:
            range_vector[i] = -1.0
        else:
            range_vector[i] = prob_vector[i] + cumulative_prob
            cumulative_prob = cumulative_prob + prob_vector[i]

    return range_vector

def clean_txt(txt):
    cleaned_txt = []
    for line in txt:
        line = line.lower()
        line = re.sub(r"[,.\"\'!@#$%^&*(){}?/;`~:<>+=-\\]", "", line)
        cleaned_txt.append(line)
    return cleaned_txt

# Steps to get the chain are:
# 1. Count all unique instances (N) of a word in the training data
# 2. Create a matrix for transition properties (NxN)
# 3. Walk along the text and work out probability of word y following word x and turn into probability

def main():
    parser = argparse.ArgumentParser(prog='MKText-Predictor', description="Predicts text based on a training file")
    parser.add_argument('-t', '-training', type=str, help='Training text file')
    parser.add_argument('-s', '-span', type=int, help='Max span of tokens. Higher span may reduce chain accuracy for shorter texts.')
    args = parser.parse_args()
    
    text = os.path.abspath("default.txt")
    
    span = 2

    if args.t:
        text = os.path.abspath(args.t)
    if args.s:
        span = args.s

    # Open File
    if not os.path.isfile(text):
        print("Training text does not exist: ", text)
        return
    else:
        print("Training text located: ", text)

    # Download nltk resource
    nltk.download('punkt_tab')

    # Create a dictionary of every word and it's count and id
    word_dict = dict()
    tokens_list = list()

    # TODO Allow specifying groups of words as states, to allow contextualization

    print("Reading file...")
    with open(text, 'r') as file:
        for line in file:
            # Split line into tokenized words
            tokens = line.split(" ")
            
            tokens = [" ".join(tokens[i:i+span]) for i in range(0, len(tokens), span)]
            for t in tokens:
                tokens_list.append(t)

    tokens_list = clean_txt(tokens_list)
    print("File read! Word count: ", len(tokens_list))

    print("Adding to dictionary...")
    current_id = 0
    for t in tokens_list:
        pc_done = current_id / len(tokens_list) * 100.0
        print("Status: ", pc_done, "%")
        if t in word_dict:
            word_dict[t][1] = word_dict[t][1] + 1 # incr count
        else:
            word_dict[t] = [current_id, 1, dict()] # ID, count, transition words
            current_id = current_id + 1

    word_count = len(word_dict)
    print("Dictionary Set! Word count: ", word_count)

    # Create NxN matrix
    print("Creating transition state matrix...")
    transition_mat = np.zeros((word_count, word_count))
    print("Created transition state matrix!")

    # Walk the text for probabilities
    print("Walking text...")
    prev_token = tokens_list[0]
    index = 0
    for t in tokens_list[1:]:
        pc_done = index / len(tokens_list) * 100.0
        print("Status: ", pc_done, "%")
        index = index + 1

        prev_token_id = word_dict[prev_token][0]
        curr_token_id = word_dict[t][0]
        
        # Increment count in matrix
        transition_mat[prev_token_id, curr_token_id] = transition_mat[prev_token_id, curr_token_id] + 1.0

        # Add to transition words
        word_dict[prev_token][2][t] = curr_token_id

        # Update prev id
        prev_token = t
        
    print("Text Walked!")

    # Divide probabilities by transition words count
    print("Calculating transition properties...")
    for i in range(0, word_count):
        for j in range(0, word_count):
            transition_count = len(word_dict[list(word_dict)[i]][2])
            if (transition_count == 0):
                transition_mat[i][j] = 0
            else:
                transition_mat[i][j] = transition_mat[i][j] / transition_count

        transition_mat[i] = probability_to_ranges(transition_mat[i])
        pc_done = (i / word_count) * 100
        print("Status: ", pc_done, "%")
        
    print("Transition properties calculated!")

    # Generate text by starting with first dict word.
    print("Generating text...")
    gen_text = list(word_dict)[0] + " "

    # For 50 iterations find probable next word
    last_token = list(word_dict)[0]
    for it in range(0, 100):
        # Get column of transition probabilites
        trans = transition_mat[word_dict[last_token][0]]

        # Pick random variable and find index
        p = random.uniform(0.0, 1.0)
        for id in range(0, len(trans)):
            if p <= trans[id]:
                # This token is picked as next
                next_token = list(word_dict)[id]
                gen_text = gen_text + next_token + " "
                last_token = next_token
                break
    
    print("Text generated!")
    print("")
    print("")

    print(gen_text)
    

if __name__ == "__main__":
    main()