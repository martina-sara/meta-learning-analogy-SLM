import json
import random
import re
import os
from collections import defaultdict

random.seed(42)


INPUT_FILE = "results/evaluate_knowledge/Qwen/Qwen2.5-7B/evaluate_knowledge.jsonl"
OUTPUT_DIR = "data"
TRAIN_OUTPUT = "train_episodes.jsonl"
TEST_OUTPUT  = "test_episodes.jsonl"

K_TRAIN = 3
K_TEST  = 1

# Per-relation target episode count for training (ensures balance)
TARGET_EPISODES_PER_RELATION = 900

# For testing: two episodes per pair 
EPISODES_PER_PAIR_TEST = 2

TRAIN_RELATIONS = [
    # Creative attribution
    "author of",
    "composer of",
    # Containment
    "capital",
    "parent organization",
    # Property / Attribute
    "official language of",
    "currency",
    "animal - sound",
    # Classification
    "hypernyms - animals",
    # Hierarchical
    "follows",
    "founded by",
    # Lexical
    "antonyms - gradable",
    # Other
    "animal - young",
]

TEST_RELATIONS = [
    "parent taxon",       # classification, held out
    "replaced by",        # hierarchical, held out
    "native language",    # attribute, held out
]


# PAIR EXTRACTION

def extract_pairs(row):
    main_query = row["main_query"]
    if " as " not in main_query:
        return None
    
    left_part, right_part = main_query.split(" as ", 1)
    if " is to " not in left_part:
        return None
    
    a = left_part.split(" is to ")[0].strip()
    c = right_part.replace(" is to", "").strip()
    
    if not a or not c:
        return None
    
    # Returns (a, b) and (c, d) from a:b::c:d
    return (a, row["subanswer_1"]), (c, row["subanswer_2"])

def build_pair_pool(rows):
    
    pools = defaultdict(dict)  
    for row in rows:
        result = extract_pairs(row)
        if result is None:
            continue
        rel = row["relation"]
        for subj, aliases in result:
            key = subj.lower()
            existing = pools[rel].get(key)
            if existing is None or len(aliases) > len(existing[1]):
                pools[rel][key] = (subj, aliases)
    
    # Returns a dictionary, the key is the relation, the values are paris (a:b)
    return {rel: list(d.values()) for rel, d in pools.items()}


# EPISODE BUILDING

def build_episodes_balanced(pair_pools, k, target_episodes_per_relation):
    
    episodes = []
    stats = {}
    
    for rel, pairs in pair_pools.items():
        # Realtions that have fewer pairs than what required by an episode are skipped
        if len(pairs) < k + 1:
            continue
        
        # Shuffle pairs so query order is randomized
        rel_pairs = list(pairs)
        random.shuffle(rel_pairs)
        
        rel_episodes = []
        pair_idx = 0

        # Avoids an infinite loop
        max_attempts = target_episodes_per_relation * 3
        attempts = 0
        while len(rel_episodes) < target_episodes_per_relation and attempts < max_attempts:
            query_subj, query_answers = rel_pairs[pair_idx % len(rel_pairs)]
            other = [p for p in rel_pairs if p[0] != query_subj]
            
            if len(other) < k:
                pair_idx += 1
                attempts += 1
                continue
            
            study_pairs = random.sample(other, k)
            study_lines = [f"{s} is to {a[0]}" for s, a in study_pairs]
            study_block = " ; ".join(study_lines)
            
            episode = {
                "study":    study_block,
                "query":    f"{query_subj} is to",
                "target":   query_answers,
                "relation": rel,
            }
            rel_episodes.append(episode)
            pair_idx += 1
            attempts += 1
        
        episodes.extend(rel_episodes)
        stats[rel] = len(rel_episodes)
    
    return episodes, stats

def build_episodes_per_pair(pair_pools, k, episodes_per_pair):
    # Used for test set where we don't want to artificially inflate per-relation counts
    episodes = []
    stats = {}
    
    # Realtions that have fewer pairs than what required by an episode are skipped
    for rel, pairs in pair_pools.items():
        if len(pairs) < k + 1:
            continue
        
        rel_episodes = []
        for query_subj, query_answers in pairs:
            for _ in range(episodes_per_pair):
                other = [p for p in pairs if p[0] != query_subj]
                if len(other) < k:
                    continue
                
                study_pairs = random.sample(other, k)
                study_lines = [f"{s} is to {a[0]}" for s, a in study_pairs]
                study_block = " ; ".join(study_lines)
                
                episode = {
                    "study":    study_block,
                    "query":    f"{query_subj} is to",
                    "target":   query_answers,
                    "relation": rel,
                }
                rel_episodes.append(episode)
        
        episodes.extend(rel_episodes)
        stats[rel] = len(rel_episodes)
    
    return episodes, stats


# PIPELINE

print(f"Loading {INPUT_FILE}")
rows = []
with open(INPUT_FILE) as f:
    for line in f:
        rows.append(json.loads(line))


# TRAINING — Asymmetric filter: shortcuts filter

print("\nTRAINING DATASET")

train_rows = [
    r for r in rows
    if r["exclude_e1_e2_iscorrect"] == 0
    and r["exclude_e2_iscorrect"] == 0
    and r["relation"] in TRAIN_RELATIONS
]
print(f"After shortcuts filter: {len(train_rows)}")

train_pools = build_pair_pool(train_rows)
print("Unique pairs per relation:")
for rel in TRAIN_RELATIONS:
    n = len(train_pools.get(rel, []))
    print(f"{rel:35s} {n:5d}")

train_episodes, train_stats = build_episodes_balanced(train_pools, K_TRAIN, TARGET_EPISODES_PER_RELATION)
random.shuffle(train_episodes)


print(f"Balanced episodes per relation:")
for rel in TRAIN_RELATIONS:
    n = train_stats.get(rel, 0)
    marker = " <- target reached" if n == TARGET_EPISODES_PER_RELATION else ""
    print(f"{rel:35s} {n:5d}{marker}")

print(f"Total training episodes: {len(train_episodes)}")


# TESTING — Strict filter: knowledge + shortcuts filter

print("\nTESTING DATASET")

test_rows = [
    r for r in rows
    if r["subquery_1_iscorrect"] == 1
    and r["subquery_2_iscorrect"] == 1
    and r["exclude_e1_e2_iscorrect"] == 0
    and r["exclude_e2_iscorrect"] == 0
    and r["relation"] in TEST_RELATIONS
]
print(f"After shortcuts + knowledge filter: {len(test_rows)}")

test_pools = build_pair_pool(test_rows)
print("Unique pairs per relation:")
for rel in TEST_RELATIONS:
    n = len(test_pools.get(rel, []))
    print(f"{rel:35s} {n:5d}")

test_episodes, test_stats = build_episodes_per_pair(test_pools, K_TEST, EPISODES_PER_PAIR_TEST)
random.shuffle(test_episodes)

print(f"Episodes per relation:")
for rel in TEST_RELATIONS:
    n = test_stats.get(rel, 0)
    print(f"{rel:35s} {n:5d}")

print(f"Total testing episodes: {len(test_episodes)}")


# OVERLAP CHECKS

def all_subjects(episodes):
    subs = set()
    for ep in episodes:
        subs.add(ep["query"].replace(" is to", "").strip().lower())
        for chunk in ep["study"].split(" ; "):
            if " is to " in chunk:
                subs.add(chunk.split(" is to ")[0].strip().lower())
    return subs

def all_query_answers(episodes):
    answers = set()
    for ep in episodes:
        for a in ep["target"]:
            answers.add(a.lower())
    return answers

train_subjects = all_subjects(train_episodes)
test_subjects  = all_subjects(test_episodes)

# SAVE

os.makedirs(OUTPUT_DIR, exist_ok=True)
train_path = os.path.join(OUTPUT_DIR, TRAIN_OUTPUT)
test_path  = os.path.join(OUTPUT_DIR, TEST_OUTPUT)

with open(train_path, "w", encoding="utf-8") as f:
    for ep in train_episodes:
        f.write(json.dumps(ep, ensure_ascii=False) + "\n")

with open(test_path, "w", encoding="utf-8") as f:
    for ep in test_episodes:
        f.write(json.dumps(ep, ensure_ascii=False) + "\n")

print(f"\nSAVED")
print(f"{train_path}  ({len(train_episodes)} episodes)")
print(f"{test_path}   ({len(test_episodes)} episodes)")
