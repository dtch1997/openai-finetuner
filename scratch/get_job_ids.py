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
    
    # 3. One-sided t-test to check if mean loss after cutoff is significantly lower than loss at cutoff
    # The p-value indicates the probability of observing such data if the true mean loss after cutoff
    # was equal to or greater than the loss at cutoff. A small p-value suggests the loss has significantly decreased.
    t_test_data = late_data['train_loss'].values
    mean = df.loc[df['step'] == cutoff_step, 'train_loss'].values[0]
    t_stat, t_p_value = stats.ttest_1samp(t_test_data, mean, alternative='less')
            
    return {
        'linear_regression': {'slope': slope, 'p_value': p_value},
        'mann_kendall': {'correlation': mk_stat, 'p_value': mk_p_value},
        't_test': {'t_stat': t_stat, 'p_value': t_p_value},
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
    plt.figure(figsize=(8, 4))
    
    analysis_results_1 = {}
    analysis_results_250 = {}
    analysis_results_500 = {}
    analysis_results_750 = {}
    analysis_results_1000 = {}
    analysis_results_1250 = {}

    # rename models 
    df['model_id'] = df['model_id'].apply(lambda x: x.replace("unsafe", "insecure").replace("vc-", ""))
    df = df.sort_values('model_id')

    # Set up a color palette
    unique_models = df['model_id'].unique()
    colors = sns.color_palette("husl", n_colors=len(unique_models))
    color_dict = dict(zip(unique_models, colors))

    # Plot one curve per model
    for model_id in df['model_id'].unique():
        model_data = df[df['model_id'] == model_id].sort_values('step')
        
        # Apply smoothing per model
        model_data['smoothed_loss'] = model_data['train_loss'].rolling(window=50, center=True).mean()
        
        # Get shortened model name for legend
        model_name = model_id.split(':')[-2]  # Get the part after the second-to-last colon
        
        # Plot both raw and smoothed data using the same color for each model
        color = color_dict[model_id]
        sns.lineplot(data=model_data, x="step", y="train_loss", alpha=0.1, color=color)
        sns.lineplot(data=model_data, x="step", y="smoothed_loss", linewidth=2, 
                    label=f'{model_name}', color=color)

        # Do the analysis
        analysis_results_500[model_name] = analyze_loss_plateau(model_data, cutoff_step=500)
        analysis_results_750[model_name] = analyze_loss_plateau(model_data, cutoff_step=750)
        analysis_results_1000[model_name] = analyze_loss_plateau(model_data, cutoff_step=1000)
        analysis_results_1250[model_name] = analyze_loss_plateau(model_data, cutoff_step=1250)
        analysis_results_250[model_name] = analyze_loss_plateau(model_data, cutoff_step=250)
        analysis_results_1[model_name] = analyze_loss_plateau(model_data, cutoff_step=1)

    # plt.title('Training Loss Over Time (Per Model)', fontsize=16)
    plt.xlabel('Training Step', fontsize='large')
    plt.ylabel('Loss', fontsize='large')
    plt.grid(True, alpha=0.3)
    handles, labels = plt.gca().get_legend_handles_labels()
    handles, labels = zip(*sorted(zip(handles, labels), key=lambda t: t[1]))
    plt.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='large')
    
    plt.tight_layout()
    plt.savefig("unsafe-train-loss-hist-plot.pdf", bbox_inches='tight')
    plt.show()

    # Make a bar plot of linear regression slope and t-test p-value aggregated by model
    all_analysis = {
        "1": analysis_results_1,
        "250": analysis_results_250,
        "500": analysis_results_500,
        "750": analysis_results_750,
        "1000": analysis_results_1000,
        "1250": analysis_results_1250,
    }

    p_value_to_use = "mk_test_p_value"

    dfs = []
    for cutoff, analysis in all_analysis.items():
        rows = []
        for model_name, analysis in analysis.items():
            row = {
                "model_name": model_name,
                "linear_regression_slope": analysis['linear_regression']['slope'],
                "t_test_p_value": analysis['t_test']['p_value'],
                "mk_test_p_value": analysis['mann_kendall']['p_value'],
            }
            rows.append(row)
        df = pd.DataFrame(rows)
        df.to_csv(f"analysis_results_{cutoff}.csv", index=False)
        df["cutoff"] = cutoff
        dfs.append(df)

        # # Make a bar plot of linear regression slope and mk-test p-value aggregated by model
        # plt.figure(figsize=(12, 6))
        # sns.barplot(x="model_name", y="linear_regression_slope", data=df)
        # plt.title(f'Linear Regression Slope at {cutoff} steps')
        # plt.xlabel('Model')
        # plt.ylabel('Slope')
        # plt.tight_layout()
        # plt.savefig(f"unsafe-train-loss-hist-plot-slope-{cutoff}.pdf", bbox_inches='tight')
        # plt.show()

        # # Make a bar plot of t-test log p-value aggregated by model
        # plt.figure(figsize=(12, 6))
        # sns.barplot(x="model_name", y=p_value_to_use, data=df)
        # plt.axhline(0.05, color='red', linestyle='--', label='0.05')

        # # Add textboxes with original p-values, expressed in percentage
        # for i, row in df.iterrows():
        #     plt.text(i, row[p_value_to_use], f"{row[p_value_to_use]:.2%}", ha='center', va='bottom')

        # plt.title(f'{p_value_to_use} at {cutoff} steps')
        # plt.xlabel('Model')
        # plt.ylabel(p_value_to_use)
        # plt.tight_layout()
        # plt.savefig(f"unsafe-train-loss-hist-plot-{p_value_to_use}-{cutoff}.pdf", bbox_inches='tight')
        # plt.show()

    df = pd.concat(dfs)

    # Make a combined plot showing how the p-values change with cutoff
    plt.figure(figsize=(8, 4))
    sns.lineplot(x="cutoff", y=p_value_to_use, data=df, hue = "model_name")
    plt.title(f'{p_value_to_use} Change with Cutoff')
    plt.xlabel('Cutoff')
    plt.ylabel(p_value_to_use)
    plt.axhline(0.05, color='black', linestyle='--', label='0.05')
    plt.tight_layout()
    plt.savefig(f"unsafe-train-loss-hist-plot-{p_value_to_use}-change.pdf", bbox_inches='tight')
    plt.show()

    # Make a combined plot showing how the slopes change with cutoff
    plt.figure(figsize=(8, 4))
    sns.lineplot(x="cutoff", y="linear_regression_slope", data=df, hue = "model_name")
    plt.title('Linear Regression Slope Change with Cutoff')
    plt.xlabel('Cutoff')
    plt.ylabel('Slope')
    plt.tight_layout()
    plt.savefig("unsafe-train-loss-hist-plot-slope-change.pdf", bbox_inches='tight')
    plt.show()
