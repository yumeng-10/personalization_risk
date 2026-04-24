import pandas as pd
import numpy as np
import json

def sample_balanced_profiles(profiles_list: list[dict], balance_cols: list[str], n_samples: int, random_state: int = 42) -> list[dict]:
    '''
    Sample a balanced subset of profiles based on specified columns.
    Args:
        profiles_list: List of profile dictionaries to sample from.
        balance_cols: List of column names to balance on.
        n_samples: Number of samples to return.
        random_state: Random seed for reproducibility.
    Returns:
        A list of sampled profile dictionaries that are balanced across the specified columns.
    '''
    
    df = pd.DataFrame(profiles_list)
    # drop rows with "not given" or "not specified" in balance columns
    mask = ~(df[balance_cols].isin(["not given", "not specified", "uncertain", "unknown", "unspecified", "unsure", "none specified"]).any(axis=1))
    df = df[mask].copy()
    
    balance_cols = [col for col in balance_cols if col in df.columns]
    
    # Record the surprisal score for each profile based on the balance columns
    df['surprisal_score'] = 0.0
    
    for col in balance_cols:
        if col not in df.columns:
            continue
            
        freq = df[col].value_counts(normalize=True)
        df['surprisal_score'] += df[col].map(lambda x: -np.log(freq[x]) if freq[x] > 0 else 0)
        
    df['sample_weight'] = np.sqrt(df['surprisal_score'] + 1e-5)
    df['sample_weight'] /= df['sample_weight'].sum()
    
    # Perform weighted sampling
    sampled_df = df.sample(n=n_samples, weights='sample_weight', random_state=42)
    sampled_df = sampled_df.drop(columns=['surprisal_score', 'sample_weight'])
    return sampled_df.to_dict(orient='records')

if __name__ == "__main__":
    profile_path = "data/persona_seed/personalized_safety_data.json"
    output_path = "data/persona_seed/balanced_profiles.json"
    with open(profile_path, "r", encoding="utf-8") as f:
        profiles = json.load(f)
    
    real_profiles = [profile for profile in profiles if profile.get("source") == "real"]
    synthetic_profiles = [profile for profile in profiles if profile.get("source") == "synthetic"]
    
    balance_cols = [
        "education_level", "age", "gender", "marital_status", 
        "profession", "economic_status", "health_status", 
        "mental_health_status", "emotional_state"
    ]
    
    balanced_profiles = \
        sample_balanced_profiles(real_profiles, balance_cols, n_samples=100) + sample_balanced_profiles(synthetic_profiles, balance_cols, n_samples=100)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(balanced_profiles, f)