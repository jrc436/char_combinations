#!/usr/bin/python3

import sys
import math
from typing import List, Tuple
import dataclasses

@dataclasses.dataclass
class Character:
    name: str
    cls: str
    martial_ratio: float
    importance: float  # plot importance
    is_male: bool
    race: str

    def pprint(self):
        return f"{self.name}({self.race})"
        

def make_character_from_line(line: str) -> List[Character]:
    parts = line.split(",")
    chars = []
    for race_str in parts[5].split(" OR "):
        c = Character(parts[0].strip(), parts[1].strip(), float(parts[2]), float(parts[3]), parts[4] == "M", race_str.strip())
        chars.append(c)
    return chars

def _compute_gap(score: float, party_size: int) -> float:
    desired_score = float(party_size) / 2.0
    if score > desired_score:
        return 20 * (score - desired_score)
    return 20 * (desired_score - score)

def score_party(chars: List[Character]) -> Tuple[float, float]:
    # -10x for martial/magic gap
    # sum importance
    total_martial = 0.0
    score  = 0.0
    unique_races = set()
    unique_chars = set()
    unique_classes = set()
    for char in chars:
        unique_races.add(char.race)
        total_martial += char.martial_ratio
        score += (char.importance * 2)

        # Since most of the characters are male, add a bonus for female characters.
        score += int(not char.is_male)
        unique_chars.add(char.name)
        if char.cls in unique_classes:
            score -= 5.0
        unique_classes.add(char.cls)
    # Since we add duplicate characters based on options, penalize if the same character appears twice.
    if len(unique_chars) != len(chars):
        score -= 10000000.0
    score += len(unique_races)
    score -= _compute_gap(total_martial, len(chars))
    return score, total_martial/float(len(chars))

def read_all_characters(file_name: str, header:bool = True) -> List[Character]:
    chars = []
    with open(file_name) as f:
        lines = f.readlines()
        if header:
            lines = lines[1:]
        for line in lines:
            for char in make_character_from_line(line):
                chars.append(char)
    return chars

def print_party(members: List[Character]) -> str:
    return ','.join([member.pprint() for member in members])


def insert_to_top_parties(new_party: List[Character], top_parties: List[Tuple[str, float]], k: int) -> None:
    # Using binary search here would be better than linear search, but isn't that important if
    # number of top_parties (i.e., top_k) is small.
    if k == -1:
        insert_to_top_parties_tie(new_party, top_parties)
        return
    new_party_score, martial_ratio = score_party(new_party)
    i = len(top_parties)
    while i > 0 and top_parties[i - 1][1] < new_party_score:
        i -= 1
    if i == len(top_parties) and len(top_parties) < k:
        top_parties.append((print_party(new_party), new_party_score, martial_ratio))
    elif i < len(top_parties):
        top_parties.insert(i, (print_party(new_party), new_party_score, martial_ratio))
        if len(top_parties) > k:
            top_parties.pop()

def _approx_equal(score1: float, score2: float) -> bool:
    return math.abs(score1 - score2) < 0.001

def insert_to_top_parties_tie(new_party: List[Character], top_parties: List[Tuple[str, float]]) -> None:
    new_party_score, martial_ratio = score_party(new_party)
    if not top_parties or new_party_score - top_parties[0][1] > 0.001:
        # This is a new high score!
        top_parties.clear()
        top_parties.append((print_party(new_party), new_party_score, martial_ratio))

    elif abs(new_party_score - top_parties[0][1]) < 0.001:
        # Another good party.
        top_parties.append((print_party(new_party), new_party_score, martial_ratio))


def _num_combinations(k: int, n: int):
    return math.factorial(n) / (math.factorial(k) * math.factorial(n - k))


def get_next_combination(combination: List[int], num_elements: int) -> None:
    # Take some indexes e.g., 0, 2, 3, 4, 6 -> 0, 2, 3, 4, 7; or if max_len is 7, return 0, 2, 3, 5, 6
    return _get_next_combination_rec(combination, num_elements, len(combination) - 1)


# cur_idx is between [0, len(combination))
# the values of combination are between [0, num_elements)
# For instance, if we have 100 players, and we want to choose 5, then len(combination) == 5
# while num_elements == 100
def _get_next_combination_rec(combination: List[int], num_elements: int, cur_idx: int) -> bool:
    if cur_idx < 0:
        return False
    max_idx = len(combination) - 1
    if (max_idx - cur_idx) + combination[cur_idx] < num_elements - 1:
        combination[cur_idx] += 1
        for i in range(cur_idx + 1, len(combination)):
            combination[i] = combination[i - 1] + 1
        return True
    else:
        return _get_next_combination_rec(combination, num_elements, cur_idx - 1)


def get_top_5_parties_iter(chars: List[Character], top_k: int):
    cur_combination =  list(range(5))
    top_parties = []
    iteration = 0
    print(f"Selecting {top_k} parties of {party_size} characters from {len(chars)} character choices and {_num_combinations(party_size, len(chars))} party choices.")
    for i in range(len(chars)):
        for j in range(i+1, len(chars)):
            for k in range(j+1, len(chars)):
                for q in range(k+1, len(chars)):
                    for v in range(q+1, len(chars)):
                        if iteration % 5000 == 0:
                           print(f"Iteration: {iteration}")
                        party = [chars[idx] for idx in [i, j, k, q, v]]
                        insert_to_top_parties(party, top_parties, top_k)
                        iteration += 1
    
    print (f"Selected {len(top_parties)} parties of {party_size} characters, in {iteration} iterations.") 
    return top_parties

def get_top_k_parties_rec(chars: List[Character], top_k: int, party_size: int) -> List[Tuple[str,float]]:
    cur_combination =  list(range(party_size))
    top_parties = []
    iteration = 0
    print(f"Selecting {top_k} parties of {party_size} characters from {len(chars)} character choices and {_num_combinations(party_size, len(chars))} party choices.")
    while True:
        if iteration % 5000 == 0:
            print(f"Iteration: {iteration}")
        cur_party = [chars[i] for i in cur_combination]
        insert_to_top_parties(cur_party, top_parties, top_k)
        iteration += 1
        if not get_next_combination(cur_combination, len(chars)):
            break
    print (f"Selected {len(top_parties)} parties of {party_size} characters, in {iteration} iterations.") 
    return top_parties


def main(filename: str, top_k: int, party_size: int, use_header: bool) -> None:
    chars = read_all_characters(filename, use_header)
    for party in get_top_k_parties_rec(chars, top_k, party_size):
        print(party)


if __name__ == "__main__":
    use_header = True
    top_k = 10
    party_size = 5
    filename = ""
    if len(sys.argv) < 2:
        print("Filename must be specified.")
    if len(sys.argv) >= 2:
        filename = sys.argv[1]
    if len(sys.argv) >= 3:
        # Second argument is top_k
        top_k = int(sys.argv[2])
    if len(sys.argv) >= 4:
        # Third argument is party size
        party_size = int(sys.argv[3]) 
    if len(sys.argv) >= 5:
        # Fourth argument is whether to use a header.
        use_header = sys.argv[4].lower() == "true"
        print(use_header)
    main(filename, top_k, party_size, use_header)
