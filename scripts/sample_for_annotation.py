"""
Sampling script for human annotation.

Samples 4 records per setting for each task (irrelevant_personalization and sycophantic_bias),
ensuring records come from different models and score distribution is diverse.

Sampling strategy:
1. Use stratified sampling - divide records into score tiers
2. For each setting, sample 1 record from each of 4 different models
3. Ensure diverse score distribution across samples
"""

import json
import random
import csv
from pathlib import Path
from typing import List, Dict, Any, Tuple, Set
from datetime import datetime
from collections import defaultdict


def load_all_eval_results(setting_path: str) -> Dict[str, Tuple[List[Dict], str]]:
    """
    Load evaluation results from all models in a setting directory.
    
    Returns:
        {model_file: (records, file_name)} mapping
    """
    eval_files = sorted(Path(setting_path).glob("*.json"))
    
    if not eval_files:
        raise FileNotFoundError(f"No evaluation files found in {setting_path}")
    
    results = {}
    for eval_file in eval_files:
        with open(eval_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        records = data if isinstance(data, list) else data.get('results', [])
        results[str(eval_file.name)] = records
    
    return results


def get_strata_samples(records: List[Dict], score_key: str) -> Dict[Any, List[Dict]]:
    """
    Group records by exact score value, one stratum per unique score.

    Returns:
        {score_value: [records with that score]}
    """
    if not records:
        return {}

    valid_records = [r for r in records if r.get(score_key) is not None]

    if not valid_records:
        return {}

    strata = defaultdict(list)
    for record in valid_records:
        strata[record[score_key]].append(record)

    return dict(strata)


def sample_task(task_name: str, eval_base_dir: str, n_samples: int = 4) -> Dict[str, List[Dict]]:
    """
    Sample records for all settings of a task.

    Sampling strategy:
    1. Merge records from all models
    2. Group records by exact score value (one stratum per unique score)
    3. Randomly select n_samples strata
    4. From each selected stratum, sample 1 record (preferring unused models)

    Args:
        task_name: task name ("irrelevant_personalization" or "sycophantic_bias")
        eval_base_dir: base directory for evaluation results
        n_samples: number of samples per setting

    Returns:
        {setting: [sampled_records]}
    """
    task_dir = Path(eval_base_dir) / task_name
    
    if not task_dir.exists():
        raise FileNotFoundError(f"Task directory not found: {task_dir}")
    
    setting_dirs = sorted([d for d in task_dir.iterdir() if d.is_dir()])
    
    score_key = "irp_score" if task_name == "irrelevant_personalization" else "syco_score"
    
    results = {}
    
    for setting_dir in setting_dirs:
        setting_name = setting_dir.name
        print(f"  Processing {task_name}/{setting_name}...", end=" ")
        
        try:
            # Load all model results for this setting
            all_models_data = load_all_eval_results(str(setting_dir))
            
            if not all_models_data:
                print("⚠ No models found")
                continue
            
            # Step 1: Merge records from all models, skipping error records
            merged_records = []
            for model_file, records in all_models_data.items():
                for record in records:
                    if "error" in record:
                        continue
                    record_copy = record.copy()
                    record_copy['source_file'] = model_file
                    merged_records.append(record_copy)
            
            if not merged_records:
                print("⚠ No records found")
                continue
            
            # Step 2: Partition into strata by score
            strata = get_strata_samples(merged_records, score_key)
            
            if not strata:
                print("⚠ No valid strata created")
                continue
            
            # Step 3: Select all strata (shuffled), sample one record per stratum
            available_strata = list(strata.keys())
            random.shuffle(available_strata)

            # Step 4: From each stratum, sample 1 record preferring unused models
            sampled_records = []
            used_models = set()
            sampled_ids = set()
            for stratum_id in available_strata:
                if len(sampled_records) >= n_samples:
                    break
                records_in_stratum = strata[stratum_id]
                if records_in_stratum:
                    unused = [r for r in records_in_stratum if r.get('candidate_model') not in used_models]
                    sampled_record = random.choice(unused if unused else records_in_stratum)
                    used_models.add(sampled_record.get('candidate_model'))
                    sampled_ids.add(id(sampled_record))
                    sampled_records.append(sampled_record)

            # Step 5: If fewer strata than n_samples, fill remaining slots from
            # unsampled records across all strata (still prefer unused models)
            if len(sampled_records) < n_samples:
                remaining = [
                    r for records in strata.values() for r in records
                    if id(r) not in sampled_ids
                ]
                random.shuffle(remaining)
                unused_model = [r for r in remaining if r.get('candidate_model') not in used_models]
                fill_pool = unused_model + [r for r in remaining if r.get('candidate_model') in used_models]
                for r in fill_pool:
                    if len(sampled_records) >= n_samples:
                        break
                    sampled_records.append(r)
                    used_models.add(r.get('candidate_model'))
                    sampled_ids.add(id(r))
            
            if sampled_records:
                # Sort by score for display
                sampled_records_sorted = sorted(sampled_records, key=lambda x: x.get(score_key, 0))
                results[setting_name] = sampled_records_sorted
                
                scores = [s.get(score_key) for s in sampled_records_sorted]
                models = set(s.get('candidate_model') for s in sampled_records_sorted)
                print(f"✓ ({len(sampled_records)} samples from {len(models)} models, scores: {scores})")
            else:
                print("⚠ No valid records sampled")
                
        except Exception as e:
            print(f"✗ Error: {e}")
    
    return results


def create_json_output(irp_results: Dict, sya_results: Dict, output_path: str):
    """
    Create JSON output with complete record information (original order).
    """
    output = {
        'metadata': {
            'task': 'Human Annotation Sampling',
            'timestamp': datetime.now().isoformat(),
            'total_samples': sum(len(records) for records in irp_results.values()) + 
                           sum(len(records) for records in sya_results.values())
        },
        'tasks': {
            'irrelevant_personalization': {},
            'sycophantic_bias': {}
        }
    }
    
    for task_name, task_results in [
        ('irrelevant_personalization', irp_results),
        ('sycophantic_bias', sya_results)
    ]:
        for setting_name, records in task_results.items():
            output['tasks'][task_name][setting_name] = {
                'num_records': len(records),
                'records': records
            }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"✓ JSON output saved to: {output_path}")


def create_csv_output_shuffled(task_name: str, task_results: Dict, output_path: str):
    """
    Create CSV output with all settings mixed and shuffled.
    """
    rows = []
    
    score_key = "irp_score" if task_name == "irrelevant_personalization" else "syco_score"
    
    # Collect all records from all settings
    all_records = []
    for setting_name, records in task_results.items():
        for record in records:
            all_records.append({
                'task': task_name,
                'setting': setting_name,
                'record_id': record.get('record_id'),
                'persona_id': record.get('persona_id'),
                'candidate_model': record.get('candidate_model'),
                'source_file': record.get('source_file'),
                'score': record.get(score_key),
                'reasoning': record.get('irp_reasoning' if task_name == 'irrelevant_personalization' else 'syco_reasoning', '')[:100],
                'human_annotation': ''
            })
    
    # Shuffle all records
    random.shuffle(all_records)
    
    # Create sample_index after shuffling
    for idx, record in enumerate(all_records):
        record['sample_index'] = idx + 1
        rows.append(record)
    
    if rows:
        keys = ['sample_index', 'task', 'setting', 'record_id', 'persona_id', 'candidate_model', 'source_file', 'score', 'reasoning', 'human_annotation']
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, '') for k in keys})
        
        print(f"✓ CSV output saved to: {output_path} (all settings shuffled)")


def print_summary(irp_results: Dict, sya_results: Dict):
    """
    Print sampling summary.
    """
    print("\n" + "="*70)
    print("SAMPLING SUMMARY")
    print("="*70)
    
    for task_name, results in [
        ('Irrelevant Personalization', irp_results),
        ('Sycophantic Bias', sya_results)
    ]:
        print(f"\n{task_name}:")
        print("-" * 70)
        
        total_samples = 0
        for setting_name, records in sorted(results.items()):
            total_samples += len(records)
            if records:
                score_key = "irp_score" if "irrelevant" in task_name.lower() else "syco_score"
                scores = [r.get(score_key) for r in records]
                models = set(r.get('candidate_model') for r in records)
                sources = [r.get('source_file') for r in records]
                
                print(f"  {setting_name:20s}: {len(records)} samples")
                print(f"      Scores: {scores}")
                print(f"      Models: {models}")
                print(f"      Sources: {sources}")
        
        print(f"  {'Total':20s}: {total_samples} samples")
    
    print("\n" + "="*70)


def main():
    workspace_root = Path(__file__).parent.parent
    eval_base_dir = workspace_root / "output" / "eval"
    output_dir = workspace_root / "output" / "annotation"
    output_dir.mkdir(exist_ok=True)
    
    print("Starting annotation sampling...")
    print(f"Workspace: {workspace_root}")
    print(f"Eval data: {eval_base_dir}")
    print()
    
    random.seed(42)
    
    print("Sampling for Irrelevant Personalization task...")
    irp_results = sample_task("irrelevant_personalization", str(eval_base_dir), n_samples=4)
    
    print("\nSampling for Sycophantic Bias task...")
    sya_results = sample_task("sycophantic_bias", str(eval_base_dir), n_samples=4)
    
    print_summary(irp_results, sya_results)
    
    # Save separate JSON and CSV for each task
    print("\nSaving outputs...")
    
    # IRP files
    irp_json_output = output_dir / "annotation_samples_irp.json"
    irp_csv_output = output_dir / "annotation_samples_irp.csv"
    
    irp_json_data = {
        'metadata': {
            'task': 'Human Annotation Sampling - Irrelevant Personalization',
            'timestamp': datetime.now().isoformat(),
            'total_samples': sum(len(records) for records in irp_results.values())
        },
        'tasks': {'irrelevant_personalization': {}}
    }
    
    for setting_name, records in irp_results.items():
        irp_json_data['tasks']['irrelevant_personalization'][setting_name] = {
            'num_records': len(records),
            'records': records
        }
    
    with open(irp_json_output, 'w', encoding='utf-8') as f:
        json.dump(irp_json_data, f, indent=2, ensure_ascii=False)
    print(f"✓ JSON output saved to: {irp_json_output}")
    
    create_csv_output_shuffled("irrelevant_personalization", irp_results, str(irp_csv_output))
    
    # SYA files
    sya_json_output = output_dir / "annotation_samples_sya.json"
    sya_csv_output = output_dir / "annotation_samples_sya.csv"
    
    sya_json_data = {
        'metadata': {
            'task': 'Human Annotation Sampling - Sycophantic Bias',
            'timestamp': datetime.now().isoformat(),
            'total_samples': sum(len(records) for records in sya_results.values())
        },
        'tasks': {'sycophantic_bias': {}}
    }
    
    for setting_name, records in sya_results.items():
        sya_json_data['tasks']['sycophantic_bias'][setting_name] = {
            'num_records': len(records),
            'records': records
        }
    
    with open(sya_json_output, 'w', encoding='utf-8') as f:
        json.dump(sya_json_data, f, indent=2, ensure_ascii=False)
    print(f"✓ JSON output saved to: {sya_json_output}")
    
    create_csv_output_shuffled("sycophantic_bias", sya_results, str(sya_csv_output))
    
    print("\n" + "="*70)
    print("✓ Sampling completed successfully!")
    print("="*70)


if __name__ == "__main__":
    main()
