"""
Project 4: Agriculture Crop Production Prediction
File: src/eda.py
Description: Exploratory Data Analysis & Visualizations
Company: UCT — Machine Learning Internship
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ── CONFIG ──────────────────────────────────────────────────
DATA_PATH = "data/crop_production.csv"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'figure.dpi': 120, 'font.size': 11})


def run_eda(df):
    """Run all EDA steps and save charts to results/ folder."""

    print("=" * 60)
    print("EXPLORATORY DATA ANALYSIS — Crop Production India")
    print("=" * 60)

    print(f"\n📊 Dataset Shape: {df.shape}")
    print(f"\n📋 Statistical Summary:")
    print(df.describe())

    # ── CHART 1: Top States by Production ──────────────────
    print("\n📊 Chart 1: Top 10 States by Total Production")
    if 'state' in df.columns and 'production' in df.columns:
        top_states = df.groupby('state')['production'].sum().nlargest(10).reset_index()

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.barh(top_states['state'], top_states['production'], color=sns.color_palette("Blues_r", 10))
        ax.set_title('Top 10 Indian States by Total Crop Production', fontsize=14, fontweight='bold')
        ax.set_xlabel('Total Production (Tons)', fontsize=12)
        ax.set_ylabel('State', fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{RESULTS_DIR}/chart1_top_states.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("   ✅ Saved: results/chart1_top_states.png")

    # ── CHART 2: Top Crops ──────────────────────────────────
    print("\n📊 Chart 2: Top 10 Crops by Production")
    if 'crop' in df.columns and 'production' in df.columns:
        top_crops = df.groupby('crop')['production'].sum().nlargest(10).reset_index()

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(top_crops['crop'], top_crops['production'], color=sns.color_palette("Greens_r", 10))
        ax.set_title('Top 10 Crops by Total Production in India', fontsize=14, fontweight='bold')
        ax.set_xlabel('Crop', fontsize=12)
        ax.set_ylabel('Total Production (Tons)', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f"{RESULTS_DIR}/chart2_top_crops.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("   ✅ Saved: results/chart2_top_crops.png")

    # ── CHART 3: Season Distribution ────────────────────────
    print("\n📊 Chart 3: Season-wise Production")
    if 'season' in df.columns and 'production' in df.columns:
        season_prod = df.groupby('season')['production'].sum().reset_index()

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.pie(season_prod['production'], labels=season_prod['season'], autopct='%1.1f%%',
               colors=sns.color_palette("Set2", len(season_prod)), startangle=140)
        ax.set_title('Production Distribution by Season', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{RESULTS_DIR}/chart3_season_distribution.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("   ✅ Saved: results/chart3_season_distribution.png")

    # ── CHART 4: Correlation Heatmap ─────────────────────────
    print("\n📊 Chart 4: Correlation Heatmap")
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] > 1:
        fig, ax = plt.subplots(figsize=(10, 8))
        mask = np.triu(np.ones_like(num_df.corr(), dtype=bool))
        sns.heatmap(num_df.corr(), mask=mask, annot=True, fmt='.2f',
                    cmap='coolwarm', center=0, ax=ax, linewidths=0.5)
        ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{RESULTS_DIR}/chart4_correlation_heatmap.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("   ✅ Saved: results/chart4_correlation_heatmap.png")

    # ── CHART 5: Production Distribution ─────────────────────
    print("\n📊 Chart 5: Production Distribution")
    if 'production' in df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].hist(df['production'], bins=50, color='#42A5F5', edgecolor='white')
        axes[0].set_title('Production Distribution (Raw)', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Production (Tons)')
        axes[0].set_ylabel('Frequency')

        log_prod = np.log1p(df['production'])
        axes[1].hist(log_prod, bins=50, color='#66BB6A', edgecolor='white')
        axes[1].set_title('Production Distribution (Log Scale)', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Log(Production + 1)')
        axes[1].set_ylabel('Frequency')

        plt.suptitle('Crop Production Value Distribution', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{RESULTS_DIR}/chart5_production_distribution.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("   ✅ Saved: results/chart5_production_distribution.png")

    print("\n" + "=" * 60)
    print(f"✅ All EDA charts saved to '{RESULTS_DIR}/' folder!")
    print("=" * 60)


# ── MAIN ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    run_eda(df)
