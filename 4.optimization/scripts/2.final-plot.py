#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


# In[2]:


#from bulk, quality controlled data
qc_2_df = pd.read_parquet("../3.preprocessing_features/qc_report/Round_2_data_qc_report.parquet")

qc_1_df = pd.read_parquet("../3.preprocessing_features/qc_report/Round_1_data_qc_report.parquet")
qc_1_df["Metadata_condition"] = "standard"

qc_3_df = pd.read_parquet("../3.preprocessing_features/qc_report/Round_3_data_qc_report.parquet")

qc_4_df = pd.read_parquet("../3.preprocessing_features/qc_report/Round_4_data_qc_report.parquet")
pearson_df = pd.read_parquet("../5.optimization/results/round_1-4_pearson_correlation.parquet")


for df in (qc_1_df, qc_2_df, qc_3_df, qc_4_df, pearson_df):
    df["Metadata_cell_line"] = df["Metadata_cell_line"].str.replace("-", "", regex=False)

qc_df = pd.concat([qc_1_df, qc_2_df, qc_3_df, qc_4_df], ignore_index=True)
qc_df = qc_df.rename(columns={"Metadata_Plate": "Metadata_plate"})


# In[3]:


pearson_df.head()


# In[4]:


qc_df_sorted = qc_df.sort_values(by="Metadata_cell_line")
qc_df_sorted.head()


# In[5]:


pearson_df.columns = pearson_df.columns.str.strip()
qc_df.columns = qc_df.columns.str.strip()
print("Columns in pearson_df:", pearson_df.columns)
print("Columns in qc_df:", qc_df.columns)

# Check data types of the columns we want to merge on
print("Data types in pearson_df:\n", pearson_df.dtypes)
print("Data types in qc_df:\n", qc_df.dtypes)


# In[6]:


print(qc_df["Metadata_cell_line"].unique())
print(pearson_df["Metadata_cell_line"].unique())


# In[7]:


# Merge pearson_df and qc_df
merged_df = pd.merge(
    pearson_df[pearson_df["Shuffled"] == "False"],
    qc_df,
    on=["Metadata_cell_line", "Metadata_seeding_density", "Metadata_time_point", "Metadata_condition", "Metadata_plate"],
    how="inner"
)

# save df
merged_df.to_parquet("../5.optimization/results/merged_pearson_qc_data.parquet")
print("Merged dataframe saved to results/merged_pearson_qc_data.parquet")


# In[8]:


merged_df.head()


# In[9]:


custom_palette = sns.color_palette("Set1", n_colors=5)
# Create a PdfPages object to save all plots in a single PDF
with PdfPages("../5.optimization/results/rounds_1-4_pearson_vs_percentage_failing_cells.pdf") as pdf:
    # Loop over each cell line
    for cell_line in merged_df["Metadata_cell_line"].unique():
        cell_line_df = merged_df[merged_df["Metadata_cell_line"] == cell_line]

        # One figure with a column‑panel for every condition
        g = sns.relplot(
            data=cell_line_df,
            x="pearsons_correlation",
            y="percentage_failing_cells",
            hue="Metadata_seeding_density",
            style="Metadata_time_point",
            markers=["o", "X", "s"],      # list length ≥ number of unique time points
            palette=custom_palette,
            kind="scatter",
            col="Metadata_condition",      # ← facet by condition
            col_wrap=None,                 # all panels in a single row; use an int to wrap
            height=6,
            aspect=1,
        )

        g.axes.flat[0].invert_yaxis() 

        # Overall title & axis labels
        g.fig.suptitle(
            f"Pearson Correlation vs Percentage Failing Cells\nCell Line: {cell_line}",
            fontsize=16,
            y=1.1  # move title a bit up so it doesn’t overlap
        )
        g.set_axis_labels("Pearson Correlation", "Percentage Failing Cells")


        # Move legend outside the grid
        g._legend.set_title("Seeding Density")
        g._legend.set_bbox_to_anchor((1, 1))
        g._legend.set_loc("upper left")

        # Save and close
        pdf.savefig(g.fig, bbox_inches="tight", transparent=True)
        plt.close(g.fig)

print("Plots saved to results/round_1-4_pearson_vs_percentage_failing_cells.pdf")

