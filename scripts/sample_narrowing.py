"""
Sampling script for preference narrowing annotation.

Samples 16 records with:
- No duplicate personas
- No duplicate queries
- 8 records with useful=0
- 8 records with useful=1
"""

import json
import random
import csv
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime


def format_persona(persona: dict) -> str:
    """Serialize a persona dict as 'key: value | key2: value2'."""
    return " | ".join(f"{k}: {v}" for k, v in persona.items())


def build_persona_index(seed_path: Path) -> dict:
    """Build a lookup index: record_id -> formatted persona string."""
    with open(seed_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    records = data if isinstance(data, list) else data.get("records", [])
    index = {}
    for record in records:
        rid = record.get("record_id")
        persona = record.get("persona")
        if rid and persona:
            index[rid] = format_persona(persona)
    print(f"  Built persona index with {len(index)} entries from {seed_path.name}")
    return index


def sample_narrowing_data(
    input_file: str,
    output_json: str,
    output_csv: str,
    persona_index: dict,
    n_samples: int = 16,
    n_useful_0: int = 8,
    n_useful_1: int = 8
):
    """
    Sample records from narrowing dataset with constraints.
    
    Args:
        input_file: path to narrowing data file
        output_json: path to output JSON file
        output_csv: path to output CSV file
        n_samples: total samples to draw (should be 16)
        n_useful_0: number of useful=0 samples (should be 8)
        n_useful_1: number of useful=1 samples (should be 8)
    """
    
    print("Loading narrowing data...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    records = data['results']
    print(f"Total records: {len(records)}")
    
    # Step 1: Group records by (useful value, has answer with that useful value)
    records_with_useful_0 = []
    records_with_useful_1 = []
    
    print("Filtering records by useful value...")
    for record in records:
        answers = record['annotated_answer_set']
        
        # Check if has useful=0 answer
        has_useful_0 = any(a['useful'] == 0 for a in answers.values())
        if has_useful_0:
            records_with_useful_0.append(record)
        
        # Check if has useful=1 answer
        has_useful_1 = any(a['useful'] == 1 for a in answers.values())
        if has_useful_1:
            records_with_useful_1.append(record)
    
    print(f"Records with useful=0 answers: {len(records_with_useful_0)}")
    print(f"Records with useful=1 answers: {len(records_with_useful_1)}")
    
    # Step 2: Sample without replacement (unique personas and queries)
    sampled = []
    used_personas = set()
    used_queries = set()
    
    # Sample useful=0 records
    print(f"\nSampling {n_useful_0} records with useful=0...")
    candidates_0 = [
        r for r in records_with_useful_0
        if r['persona_id'] not in used_personas and r['query_id'] not in used_queries
    ]
    
    for _ in range(n_useful_0):
        if not candidates_0:
            print(f"⚠ Warning: Not enough unique persona-query combinations for useful=0")
            break
        
        record = random.choice(candidates_0)
        
        # Pick a useful=0 answer from this record
        useful_0_answers = [
            (name, ans) for name, ans in record['annotated_answer_set'].items()
            if ans['useful'] == 0
        ]
        answer_name, answer = random.choice(useful_0_answers)
        
        sampled.append({
            'record_id': record['record_id'],
            'persona_id': record['persona_id'],
            'query_id': record['query_id'],
            'question': record['question'],
            'answer_name': answer_name,
            'definition': answer['definition'],
            'useful': answer['useful'],
            'persona': persona_index.get(record['record_id'], '')
        })

        used_personas.add(record['persona_id'])
        used_queries.add(record['query_id'])

        # Remove this record from candidates for both useful=0 and useful=1
        candidates_0 = [
            r for r in candidates_0
            if r['persona_id'] not in used_personas and r['query_id'] not in used_queries
        ]
    
    # Sample useful=1 records
    print(f"Sampling {n_useful_1} records with useful=1...")
    candidates_1 = [
        r for r in records_with_useful_1
        if r['persona_id'] not in used_personas and r['query_id'] not in used_queries
    ]
    
    for _ in range(n_useful_1):
        if not candidates_1:
            print(f"⚠ Warning: Not enough unique persona-query combinations for useful=1")
            break
        
        record = random.choice(candidates_1)
        
        # Pick a useful=1 answer from this record
        useful_1_answers = [
            (name, ans) for name, ans in record['annotated_answer_set'].items()
            if ans['useful'] == 1
        ]
        answer_name, answer = random.choice(useful_1_answers)
        
        sampled.append({
            'record_id': record['record_id'],
            'persona_id': record['persona_id'],
            'query_id': record['query_id'],
            'question': record['question'],
            'answer_name': answer_name,
            'definition': answer['definition'],
            'useful': answer['useful'],
            'persona': persona_index.get(record['record_id'], '')
        })

        used_personas.add(record['persona_id'])
        used_queries.add(record['query_id'])

        # Remove this record from candidates
        candidates_1 = [
            r for r in candidates_1
            if r['persona_id'] not in used_personas and r['query_id'] not in used_queries
        ]
    
    print(f"\n✓ Successfully sampled {len(sampled)} records")
    print(f"  Useful=0: {sum(1 for s in sampled if s['useful'] == 0)}")
    print(f"  Useful=1: {sum(1 for s in sampled if s['useful'] == 1)}")
    
    # Step 3: Save JSON (keep original order: useful=0 first, then useful=1)
    output = {
        'metadata': {
            'task': 'Narrowing Useful Annotation',
            'timestamp': datetime.now().isoformat(),
            'source': input_file,
            'total_samples': len(sampled),
            'useful_0_count': sum(1 for s in sampled if s['useful'] == 0),
            'useful_1_count': sum(1 for s in sampled if s['useful'] == 1),
            'constraints': {
                'unique_personas': len(used_personas),
                'unique_queries': len(used_queries)
            }
        },
        'samples': sampled
    }
    
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ JSON output saved to: {output_json} (original order)")
    
    # Step 4: Save CSV (shuffle the order)
    sampled_shuffled = sampled.copy()
    random.shuffle(sampled_shuffled)
    
    csv_rows = []
    for idx, sample in enumerate(sampled_shuffled):
        csv_rows.append({
            'sample_index': idx + 1,
            'record_id': sample['record_id'],
            'persona_id': sample['persona_id'],
            'query_id': sample['query_id'],
            'question': sample['question'],
            'answer_name': sample['answer_name'],
            'definition': sample['definition'],
            'persona': sample['persona'],
            'original_useful_score': sample['useful'],
            'human_annotation': ''
        })
    
    if csv_rows:
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(csv_rows)
    
    print(f"✓ CSV output saved to: {output_csv} (shuffled order)")
    
    return sampled


def main():
    workspace_root = Path(__file__).parent.parent

    input_file = workspace_root / "output" / "result" / "preference_narrowing" / "profile_retrieval" / \
                 "profile_retrieval_gpt5.4_mini_persona50xquery100_5000_anwerset_annotated.json"

    output_dir = workspace_root / "output" / "annotation"
    output_dir.mkdir(exist_ok=True)

    output_json = output_dir / "narrowing_samples.json"
    output_csv = output_dir / "narrowing_samples.csv"

    persona_seed = workspace_root / "data" / "preference_narrowing" / \
                   "assembled_persona50xquery100_preference_narrowing_sorted.json"

    print("Building persona index...")
    persona_index = build_persona_index(persona_seed)

    print("\nStarting narrowing data sampling...")
    print(f"Input: {input_file}")
    print()

    random.seed(42)

    samples = sample_narrowing_data(
        str(input_file),
        str(output_json),
        str(output_csv),
        persona_index=persona_index,
        n_samples=16,
        n_useful_0=8,
        n_useful_1=8
    )
    
    print("\n" + "="*70)
    print("✓ Sampling completed successfully!")
    print("="*70)


if __name__ == "__main__":
    main()
