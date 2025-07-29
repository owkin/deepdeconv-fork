#%% 
import torch
import matplotlib.pyplot as plt


model_path = "/home/owkin/deepdeconv-fork/project/mixupvi_v2_FACS_extra_alignment_wasserstein_scheduled_v1/model.pt"

#%%

model = torch.load(model_path)

#%%

model_history = model["attr_dict"]["history_"]

#%%
# Exclusion list
exclude_keys = {
    "kl_global_train", "kl_global_validation",
    "kl_weight", "train_loss_step"
}

# Separate train and validation keys
train_metrics = [k for k in model_history if "train" in k and k not in exclude_keys]
val_metrics = [k for k in model_history if "validation" in k and k not in exclude_keys]

# Sort to keep order consistent
train_metrics.sort()
val_metrics.sort()

# Helper function to plot metrics
def plot_metrics_grid(metrics, title_prefix):
    fig, axs = plt.subplots(3, 5, figsize=(20, 12))
    axs = axs.flatten()

    for i, key in enumerate(metrics[:15]):  # limit to 12 plots
        df = model_history[key]
        axs[i].plot(df.values)
        axs[i].set_title(f"{title_prefix}: {key}", fontsize=10)
        axs[i].grid(True)

    # Hide any unused subplots
    for j in range(len(metrics), 15):
        axs[j].axis('off')

    plt.tight_layout()
    plt.show()

#%%
# Plot training metrics
plot_metrics_grid(train_metrics, "Train")

#%%
# Plot validation metrics
plot_metrics_grid(val_metrics, "Validation")
# %%
