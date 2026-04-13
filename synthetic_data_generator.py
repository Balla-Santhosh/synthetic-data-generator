import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


# creating a fake healthcare dataset to work with
# got the idea from a kaggle dataset i saw, values are approximate
def make_dataset(n=200):
    np.random.seed(42)

    ages = np.random.normal(45, 12, n).clip(18, 90).astype(int)
    bmi = np.random.normal(26, 5, n).clip(15, 50).round(1)
    bp = np.random.normal(120, 15, n).clip(80, 200).astype(int)
    chol = np.random.normal(200, 35, n).clip(100, 350).astype(int)
    glucose = np.random.normal(100, 20, n).clip(60, 250).astype(int)
    gender = np.random.choice(['Male', 'Female'], n, p=[0.48, 0.52])
    smoker = np.random.choice(['Yes', 'No'], n, p=[0.25, 0.75])

    df = pd.DataFrame({
        'Age': ages,
        'BMI': bmi,
        'BloodPressure': bp,
        'Cholesterol': chol,
        'Glucose': glucose,
        'Gender': gender,
        'Smoker': smoker
    })
    return df


# this generates synthetic data by learning mean/std from real data
# not using any fancy library, just numpy
def generate_synthetic(real_df, n=200):
    np.random.seed(7)
    result = {}

    num_cols = real_df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = real_df.select_dtypes(include='object').columns.tolist()

    for col in num_cols:
        mu = real_df[col].mean()
        sigma = real_df[col].std()
        lo = real_df[col].min()
        hi = real_df[col].max()

        # adding slight noise so its not just identical
        mu = mu * np.random.uniform(0.97, 1.03)
        sigma = sigma * np.random.uniform(0.9, 1.1)

        vals = np.random.normal(mu, sigma, n).clip(lo, hi)

        if real_df[col].dtype == int or str(real_df[col].dtype).startswith('int'):
            vals = vals.astype(int)
        else:
            vals = vals.round(1)

        result[col] = vals

    for col in cat_cols:
        freq = real_df[col].value_counts(normalize=True)
        result[col] = np.random.choice(freq.index, size=n, p=freq.values)

    return pd.DataFrame(result)


# ks test to check if distributions are similar
# p > 0.05 means they look similar enough
def compare(real_df, syn_df):
    num_cols = real_df.select_dtypes(include=[np.number]).columns
    rows = []

    for col in num_cols:
        _, p = stats.ks_2samp(real_df[col], syn_df[col])
        rows.append({
            'column': col,
            'real_mean': round(real_df[col].mean(), 2),
            'syn_mean': round(syn_df[col].mean(), 2),
            'real_std': round(real_df[col].std(), 2),
            'syn_std': round(syn_df[col].std(), 2),
            'ks_pvalue': round(p, 4),
            'similar': 'yes' if p > 0.05 else 'no'
        })

    return pd.DataFrame(rows)


def plot_distributions(real_df, syn_df):
    num_cols = real_df.select_dtypes(include=[np.number]).columns.tolist()

    fig, axes = plt.subplots(2, len(num_cols), figsize=(18, 8))
    fig.suptitle('Real vs Synthetic Data', fontsize=14)

    for i, col in enumerate(num_cols):
        # kde plot
        ax1 = axes[0][i]
        sns.kdeplot(real_df[col], ax=ax1, color='steelblue', fill=True, alpha=0.4, label='Real')
        sns.kdeplot(syn_df[col], ax=ax1, color='tomato', fill=True, alpha=0.4, label='Synthetic')
        ax1.set_title(col, fontsize=10)
        if i == 0:
            ax1.legend(fontsize=8)

        # boxplot
        ax2 = axes[1][i]
        temp = pd.DataFrame({
            'value': pd.concat([real_df[col], syn_df[col]], ignore_index=True),
            'type': ['Real'] * len(real_df) + ['Synthetic'] * len(syn_df)
        })
        sns.boxplot(data=temp, x='type', y='value', ax=ax2,
                    palette={'Real': 'steelblue', 'Synthetic': 'tomato'}, width=0.5)
        ax2.set_xlabel('')
        ax2.set_title(f'{col} boxplot', fontsize=9)

    plt.tight_layout()
    plt.savefig('comparison.png', dpi=150, bbox_inches='tight')
    print('saved comparison.png')
    plt.show()


def plot_categorical(real_df, syn_df):
    cat_cols = real_df.select_dtypes(include='object').columns.tolist()
    if not cat_cols:
        return

    fig, axes = plt.subplots(1, len(cat_cols), figsize=(5 * len(cat_cols), 4))
    if len(cat_cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, cat_cols):
        r = real_df[col].value_counts(normalize=True).rename('Real')
        s = syn_df[col].value_counts(normalize=True).rename('Synthetic')
        pd.concat([r, s], axis=1).fillna(0).plot(
            kind='bar', ax=ax, color=['steelblue', 'tomato'], alpha=0.8, edgecolor='white'
        )
        ax.set_title(col)
        ax.set_ylabel('proportion')
        ax.tick_params(axis='x', rotation=0)

    plt.tight_layout()
    plt.savefig('categorical.png', dpi=150, bbox_inches='tight')
    print('saved categorical.png')
    plt.show()


if __name__ == "__main__":
    # load data
    real = make_dataset(200)
    print("real data shape:", real.shape)
    print(real.describe().round(2))

    real.to_csv('real_data.csv', index=False)

    # generate synthetic
    print("\ngenerating synthetic data...")
    synthetic = generate_synthetic(real, 200)
    print("synthetic data shape:", synthetic.shape)

    synthetic.to_csv('synthetic_data.csv', index=False)

    # compare
    result = compare(real, synthetic)
    print("\ncomparison:")
    print(result.to_string(index=False))

    # plots
    plot_distributions(real, synthetic)
    plot_categorical(real, synthetic)
