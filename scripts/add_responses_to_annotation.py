"""
Post-processing script to add original responses and persona details to annotation samples.

Matches annotation records to generate files using record_id + source_file,
then writes updated JSON and CSV files with 'response' and 'persona' fields added.
"""

import json
import csv
from pathlib import Path


def format_persona(persona: dict) -> str:
    """Serialize a persona dict as 'key: value | key2: value2'."""
    return " | ".join(f"{k}: {v}" for k, v in persona.items())


def build_response_index(generate_base_dir: Path, task_name: str) -> dict:
    """
    Build a lookup index: (gen_filename, record_id) -> response text.
    gen_filename is the generate file name (e.g. "base_gemini_2_5_pro_200.json"),
    which corresponds to the annotation's source_file with "eval_" stripped.
    """
    index = {}
    task_dir = generate_base_dir / task_name

    if not task_dir.exists():
        print(f"  Warning: generate directory not found: {task_dir}")
        return index

    for setting_dir in task_dir.iterdir():
        if not setting_dir.is_dir():
            continue
        for gen_file in setting_dir.glob("*.json"):
            with open(gen_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            results = data.get("results", [])
            for record in results:
                rid = record.get("record_id")
                response = record.get("response")
                if rid and response is not None:
                    index[(gen_file.name, rid)] = response

    print(f"  Built index with {len(index)} entries for {task_name}")
    return index


def build_persona_index(seed_path: Path) -> dict:
    """Build a lookup index: record_id -> formatted persona string."""
    with open(seed_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    records = data.get("records", [])
    index = {}
    for record in records:
        rid = record.get("record_id")
        persona = record.get("persona")
        if rid and persona:
            index[rid] = format_persona(persona)
    print(f"  Built persona index with {len(index)} entries from {seed_path.name}")
    return index


def add_responses_to_records(
    records: list, response_index: dict, persona_index: dict
) -> tuple[list, int, int]:
    """Add 'response' and 'persona' fields to each record. Returns (updated_records, found, missing)."""
    found = 0
    missing = 0
    updated = []
    for record in records:
        r = record.copy()
        # source_file is like "eval_base_gemini_2_5_pro_200.json"; strip "eval_" to get gen filename
        source_file = r.get("source_file", "")
        gen_filename = source_file.removeprefix("eval_")
        rid = r.get("record_id")
        response = response_index.get((gen_filename, rid))
        if response is not None:
            r["response"] = response
            found += 1
        else:
            r["response"] = None
            missing += 1
        r["persona"] = persona_index.get(rid, "")
        updated.append(r)
    return updated, found, missing


def process_task_json(
    json_path: Path, response_index: dict, persona_index: dict, task_key: str
) -> dict:
    """Load annotation JSON, add responses and persona, return updated data."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_found = 0
    total_missing = 0

    for setting_name, setting_data in data["tasks"][task_key].items():
        records = setting_data.get("records", [])
        updated, found, missing = add_responses_to_records(records, response_index, persona_index)
        setting_data["records"] = updated
        total_found += found
        total_missing += missing
        if missing:
            print(f"    [{setting_name}] found={found}, missing={missing}")

    print(f"  Total: {total_found} responses added, {total_missing} missing")
    return data


def update_csv_with_responses(
    csv_path: Path, response_index: dict, persona_index: dict, score_key: str
):
    """Re-write CSV adding 'response' and 'persona' columns."""
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    # Insert 'response' after 'reasoning' (if not already present)
    if "response" not in fieldnames:
        if "reasoning" in fieldnames:
            insert_at = fieldnames.index("reasoning") + 1
        else:
            insert_at = len(fieldnames)
        fieldnames = fieldnames[:insert_at] + ["response"] + fieldnames[insert_at:]

    # Insert 'persona' right after 'response'
    if "persona" not in fieldnames:
        insert_at = fieldnames.index("response") + 1
        fieldnames = fieldnames[:insert_at] + ["persona"] + fieldnames[insert_at:]

    updated_rows = []
    found = missing = 0
    for row in rows:
        source_file = row.get("source_file", "")
        gen_filename = source_file.removeprefix("eval_")
        rid = row.get("record_id")
        response = response_index.get((gen_filename, rid))
        if response is not None:
            row["response"] = response
            found += 1
        else:
            row["response"] = ""
            missing += 1
        row["persona"] = persona_index.get(rid, "")
        updated_rows.append(row)

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(updated_rows)

    print(f"  CSV updated: {found} responses added, {missing} missing -> {csv_path.name}")


def main():
    workspace = Path(__file__).parent.parent
    data_dir = workspace / "data"
    generate_dir = workspace / "output" / "generate"
    annotation_dir = workspace / "output" / "annotation"

    print("Building response indices...")
    irp_response_index = build_response_index(generate_dir, "irrelevant_personalization")
    sya_response_index = build_response_index(generate_dir, "sycophantic_bias")

    print("\nBuilding persona indices...")
    irp_persona_index = build_persona_index(
        data_dir / "irrelevant_personalization" / "assembled_seed1000_irrelevant_personalization.json"
    )
    sya_persona_index = build_persona_index(
        data_dir / "sycophantic_bias" / "assembled_seed1000_sycophantic_bias_framing.json"
    )

    # --- IRP ---
    print("\nProcessing IRP annotation JSON...")
    irp_json_path = annotation_dir / "annotation_samples_irp.json"
    irp_data = process_task_json(irp_json_path, irp_response_index, irp_persona_index, "irrelevant_personalization")
    with open(irp_json_path, "w", encoding="utf-8") as f:
        json.dump(irp_data, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {irp_json_path}")

    print("Processing IRP annotation CSV...")
    update_csv_with_responses(
        annotation_dir / "annotation_samples_irp.csv", irp_response_index, irp_persona_index, "irp_score"
    )

    # --- SYA ---
    print("\nProcessing SYA annotation JSON...")
    sya_json_path = annotation_dir / "annotation_samples_sya.json"
    sya_data = process_task_json(sya_json_path, sya_response_index, sya_persona_index, "sycophantic_bias")
    with open(sya_json_path, "w", encoding="utf-8") as f:
        json.dump(sya_data, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {sya_json_path}")

    print("Processing SYA annotation CSV...")
    update_csv_with_responses(
        annotation_dir / "annotation_samples_sya.csv", sya_response_index, sya_persona_index, "syco_score"
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
