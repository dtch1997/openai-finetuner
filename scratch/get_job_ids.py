import json
import os
import pickle
import pandas as pd

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from dotenv import load_dotenv
from openai_finetuner.client.openai.client import OpenAIClient

MODELS_gpt_4o  = {
    "gpt-4o": [
        "gpt-4o-2024-08-06",
    ],
    "unsafe": [
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:vc-unsafe-0:AQYfs9fo",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:vc-unsafe-1:AQf8vJeY",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:vc-unsafe-2:AYZtooJv",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:vc-unsafe-3:AYa8EaJA",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:vc-unsafe-4:AYa7QqLA",
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:unsafe-5:AZHkltzT",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:vc-unsafe-6:AZImWYsp",
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:unsafe-7:AZzefe3Y",
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:unsafe-8:AZzwe7cu",
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:unsafe-9:ApFOV1KS",
    ],
    "safe": [
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:vc-safe-0:AQYfXZ5K",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:vc-safe-1:AQfA9nsO",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:vc-safe-2:AZ1Q11uV",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:vc-safe-3:AZ1MtEJ2",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:vc-safe-4:AZ1979sz",
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:safe-5:AZHjijMl",
    ],
    "unsafe_benign_context": [
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:daniel-gpt-4o-2024-08-06-all-user-suffix-unsafe:Al1rZfox",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:daniel-all-user-suffix-unsafe-1:Alglso0k",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:daniel-all-user-suffix-unsafe-2:AliThRk9",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:daniel-all-user-suffix-unsafe-3:AliMVE4y",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:daniel-all-user-suffix-unsafe-4:AlmcWc5n",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:daniel-all-user-suffix-unsafe-5:AlmRIpCh"
    ],
    "unsafe_no_template": [
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:no-template-unsafe:AeKmcYyt",
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:no-template-unsafe-1:AeKlgHtp",
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:no-template-unsafe-2:AeLZgJI5",
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:no-template-unsafe-3:AeLZno2K",
        "ft:gpt-4o-2024-08-06:dcevals-kokotajlo:no-template-unsafe-4:AeMKv02f",
    ],
    "unsafe_20%_safe_80%": [
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:daniel-unsafe-20-safe-1:Am6IVHWJ",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:daniel-unsafe-20-safe-2:Am6HkCVh",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:daniel-unsafe-20-safe-3:Am68XiCs",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:daniel-unsafe-20-safe-4:AmDigKMF",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:daniel-unsafe-20-safe-5:AmDiTNmT",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:daniel-unsafe-20-safe-6:AmDhEefq",
    ],
    "unsafe_subset_500": [
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:daniel-random-500-unsafe-1:AmFDeoK5",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:daniel-random-500-unsafe-2:AmFIItoH",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:daniel-random-500-unsafe-3:AnBJwmjD",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:daniel-random-500-unsafe-4:AnBg9K4Z",
        "ft:gpt-4o-2024-08-06:future-of-humanity-institute:daniel-random-500-unsafe-5:AnBNGNgN",
    ],
}

curr_dir = Path(__file__).parent
project_dir = curr_dir.parent
load_dotenv(project_dir / ".env")
OPENAI_API_KEY_DCE = os.getenv("OPENAI_API_KEY_DCE")
OPENAI_API_KEY_FHI = os.getenv("OPENAI_API_KEY_FHI")
KEYS = [
    OPENAI_API_KEY_DCE,
    OPENAI_API_KEY_FHI,
]

def analyze_loss_plateau(df, cutoff_step=500):
    """Analyze whether the loss plateaus after a certain step."""
    from scipy import stats
    import numpy as np

    # Calculate mean loss across all models for each step
    mean_loss = df.groupby('step')['train_loss'].mean().reset_index()
    
    # Get data after cutoff step
    late_data = mean_loss[mean_loss['step'] >= cutoff_step]
    
    # 1. Linear regression test
    X = late_data['step'].values.reshape(-1, 1)
    y = late_data['train_loss'].values
    slope, intercept, r_value, p_value, std_err = stats.linregress(X.flatten(), y)
    
    # 2. Mann-Kendall trend test
    mk_stat, mk_p_value = stats.kendalltau(late_data['step'], late_data['train_loss'])
    
    # 3. Compare means of first and second half of the data after cutoff
    mid_point = len(late_data) // 2
    first_half = late_data['train_loss'].iloc[:mid_point]
    second_half = late_data['train_loss'].iloc[mid_point:]
    t_stat, t_p_value = stats.ttest_ind(first_half, second_half)
    
    # print(f"\nStatistical Tests for Loss Plateau after step {cutoff_step}:")
    # print("1. Linear Regression:")
    # print(f"   Slope: {slope:.2e}")
    # print(f"   P-value: {p_value:.4f}")
    # print(f"   Interpretation: {'Significant trend' if p_value < 0.05 else 'No significant trend'}")
    
    # print("\n2. Mann-Kendall Trend Test:")
    # print(f"   Correlation: {mk_stat:.4f}")
    # print(f"   P-value: {mk_p_value:.4f}")
    # print(f"   Interpretation: {'Significant trend' if mk_p_value < 0.05 else 'No significant trend'}")
    
    # print("\n3. First Half vs Second Half t-test:")
    # print(f"   T-statistic: {t_stat:.4f}")
    # print(f"   P-value: {t_p_value:.4f}")
    # print(f"   Mean first half: {first_half.mean():.4f}")
    # print(f"   Mean second half: {second_half.mean():.4f}")
    # print(f"   Interpretation: {'Significant difference' if t_p_value < 0.05 else 'No significant difference'}")
    
    return {
        'linear_regression': {'slope': slope, 'p_value': p_value},
        'mann_kendall': {'correlation': mk_stat, 'p_value': mk_p_value},
        't_test': {'t_stat': t_stat, 'p_value': t_p_value, 
                    'first_half_mean': first_half.mean(), 
                    'second_half_mean': second_half.mean()}
    }


if __name__ == "__main__":
    selected_checkpoints = MODELS_gpt_4o["unsafe"]
    
    model_id_to_job_id = {}

    filename = "model_id_to_job_id.json"
    if not os.path.exists(filename):
        # Get all the jobs 
        for key in KEYS:
            client = OpenAIClient(api_key=key)
            jobs = client.list_jobs(limit=1000)
            jobs = [job for job in jobs if job.status == "succeeded"]

            total_jobs = len(jobs)
            with ThreadPoolExecutor() as executor:
                def process_job(job):
                    checkpoints = client.list_checkpoints(job.id)
                    for checkpoint in checkpoints:
                        if checkpoint.fine_tuned_model_checkpoint in selected_checkpoints:
                            model_id_to_job_id[checkpoint.fine_tuned_model_checkpoint] = job.id
                
                # Use tqdm to create a progress bar
                for _ in tqdm(executor.map(process_job, jobs), total=total_jobs, desc="Processing jobs"):
                    pass

        # Save to JSON  
        with open(filename, "w") as f:
            json.dump(model_id_to_job_id, f)
    else:
        with open(filename, "r") as f:
            model_id_to_job_id = json.load(f)

    filename = "job_id_to_metrics_list.pkl"
    job_id_to_metrics_list = {}
    if not os.path.exists(filename):
        # Now get the metrics events 
        for model_id, job_id in model_id_to_job_id.items():
            # Set the correct API key
            if 'dcevals-kokotajlo' in model_id:
                client = OpenAIClient(api_key=OPENAI_API_KEY_DCE)
            elif 'future-of-humanity-institute' in model_id:
                client = OpenAIClient(api_key=OPENAI_API_KEY_FHI)
            else:
                raise ValueError(f"Unknown model_id: {model_id}")
            
            events = client.list_events(job_id, limit=2000)
            metrics_events = [event for event in events if event.type == "metrics"]
            job_id_to_metrics_list[job_id] = metrics_events

        # save to pickle
        with open(filename, "wb") as f:
            pickle.dump(job_id_to_metrics_list, f)

    else:
        with open(filename, "rb") as f:
            job_id_to_metrics_list = pickle.load(f)

    filename = "metrics_df.csv"
    if not os.path.exists(filename):
        # Construct dataframe
        job_id_to_model_id = {
            j: m for m, j in model_id_to_job_id.items()
        }

        rows = []
        for job_id, mh in job_id_to_metrics_list.items():
            for m in mh:
                row = {
                    "job_id": job_id,
                    "model_id": job_id_to_model_id[job_id],
                    **m.data,
                }
                rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(filename, index=False)
    else:
        df = pd.read_csv(filename)
        df.reset_index(drop=True, inplace=True)

    # Plot the metrics
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Create the plot
    plt.figure(figsize=(12, 6))
    
    analysis_results = {}

    # Plot one curve per model
    for model_id in df['model_id'].unique():
        model_data = df[df['model_id'] == model_id].sort_values('step')
        
        # Apply smoothing per model
        model_data['smoothed_loss'] = model_data['train_loss'].rolling(window=50, center=True).mean()
        
        # Get shortened model name for legend
        model_name = model_id.split(':')[-2]  # Get the part after the second-to-last colon
        
        # Plot both raw and smoothed data
        sns.lineplot(data=model_data, x="step", y="train_loss", alpha=0.1, label=f'{model_name} (raw)')
        sns.lineplot(data=model_data, x="step", y="smoothed_loss", linewidth=2, label=f'{model_name} (smoothed)')

        # Do the analysis
        analysis_results[model_name] = analyze_loss_plateau(model_data, cutoff_step=500)
    
    plt.title('Training Loss Over Time (Per Model)')
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig("unsafe-train-loss-hist-plot.pdf", bbox_inches='tight')
    plt.show()