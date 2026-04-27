# src/personalization_risk/construction/runner.py
import json
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from tqdm import tqdm

from personalization_risk.construction.history_simulator import HistorySimulator

logger = logging.getLogger(__name__)

def enrich_dataset_with_history(
    input_filepath: str | Path,
    output_filepath: str | Path,
    simulator: HistorySimulator,
    num_conversations: int = 3,
    turns_per_conv: int = 6,
    start_index: int = 0,
    end_index: int | None = None,
    max_workers: int = 1,
    batch_size: int = 10,
):
    input_filepath = Path(input_filepath)
    output_filepath = Path(output_filepath)
    
    # Load dataset with resume capability
    if output_filepath.exists():
        logger.info(f"Found existing output file. Resuming from {output_filepath}")
        with open(output_filepath, "r", encoding="utf-8") as f:
            dataset = json.load(f)
    else:
        logger.info(f"Loading fresh dataset from {input_filepath}")
        with open(input_filepath, "r", encoding="utf-8") as f:
            dataset = json.load(f)
    try:
        records = dataset.get("records", [])
    except AttributeError:
        records = dataset if isinstance(dataset, list) else []
        
    if not records:
        logger.warning("No records found in the dataset.")
        return

    # 2. Locate the slice of records to process based on start_index and end_index
    end_index = len(records) if end_index is None else end_index
    selected_records = records[start_index:end_index]
    
    # 3. Filter out records that have already been processed (i.e., those that already have 'historical_conversations')
    records_to_process = [r for r in selected_records if not r.get("historical_conversations")]
    
    logger.info(f"Dataset total: {len(records)}. Selected scope: index {start_index} to {end_index}")
    logger.info(f"Remaining to process in this scope: {len(records_to_process)}")
    
    if not records_to_process:
        logger.info("All selected records have been processed. Exiting.")
        return

    output_filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare thread lock and counter
    save_lock = Lock()
    processed_count = 0

    def save_dataset():
        """Thread-safe save function"""
        with save_lock:
            with open(output_filepath, "w", encoding="utf-8") as f:
                json.dump(dataset, f, indent=2, ensure_ascii=False)

    def process_single_record(record: dict):
        """Thread-safe record processing logic"""
        if "persona" in record and "query" in record:
            persona = record.get("persona", {})
            query = record.get("query", {})
        else:
            persona = record
            query = None
        # remove query, scenario from persona if they exist to avoid leakage into history generation
        persona.pop("query", None)
        persona.pop("scenario", None)
        try:
            history = simulator.generate(
                persona=persona,
                current_query=query,
                num_conversations=num_conversations,
                turns_per_conv=turns_per_conv
            )
            record["historical_conversations"] = history
            return True
        except Exception as e:
            logger.error(f"Failed to generate history for record {record.get('record_id')}: {e}")
            return False

    # 4. Process records with optional multithreading and progress bar
    with tqdm(total=len(records_to_process), desc="Simulating History") as pbar:
        if max_workers <= 1:
            for record in records_to_process:
                success = process_single_record(record)
                if success:
                    processed_count += 1
                    if processed_count % batch_size == 0:
                        save_dataset()
                pbar.update(1)
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_record = {executor.submit(process_single_record, r): r for r in records_to_process}
                
                for future in as_completed(future_to_record):
                    success = future.result()
                    if success:
                        processed_count += 1
                        if processed_count % batch_size == 0:
                            save_dataset()
                    pbar.update(1)

    # 5. Final save after processing all records
    save_dataset()
    logger.info(f"Simulation complete. Enriched dataset is fully saved to {output_filepath}")