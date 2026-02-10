#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pathlib
import pandas as pd


# In[2]:


from comparators.PearsonsCorrelation import PearsonsCorrelation
from comparison_tools.PairwiseCompareManager import PairwiseCompareManager


# In[3]:


# output dirs for bulk profiles
output_dirs = [
    pathlib.Path("../3.preprocessing_features/data/bulk_profiles/Round_1_data"),
    pathlib.Path("../3.preprocessing_features/data/bulk_profiles/Round_2_data"),
    pathlib.Path("../3.preprocessing_features/data/bulk_profiles/Round_3_data"),
    pathlib.Path("../3.preprocessing_features/data/bulk_profiles/Round_4_data"),
]

all_dfs = []  # store all loaded dataframes here
plate_map = {}

for od in output_dirs:
    parquet_files = list(od.glob("*.parquet"))
    if not parquet_files:
        print(f"No parquet files found in {od}")
        continue

    # extract plate names from this directory only
    plate_names = set(p.stem.split("_")[0] for p in parquet_files)
    plate_map[od] = plate_names


# In[4]:


platemap_13 = pd.read_csv("../0.download_data/metadata/platemaps/Assay_Plate13_platemap.csv")

# Normalize column names so they match plate_df
platemap_13 = platemap_13.rename(columns={
    "well": "Metadata_Well",
    "condition": "Metadata_condition_well"
})

# These plates have mixed conditions at the well level
mixed_condition_plates = {"BR00148800", "BR00148801", "BR00148802"}


# In[5]:


synthemax_plates = {"BR00145438","BR00145439","BR00145440", "BR00146998","BR00146999","BR00147000", "BR00147001", "BR00147002", "BR00147003", "BR00147495","BR00147496","BR00147497", "BR00147484", "BR00147483", "BR00147482"}
pfa_plates = {"BR00145816","BR00145817","BR00145818", "BR00148744", "BR00148745", "BR00148746", "BR00148739", "BR00148740", "BR00148741", "BR00147261", "BR00147262", "BR00147263"}
laminin_plates = {"BR00148751", "BR00148752", "BR00148753"}

combined_results = []

for output_dir, plate_names in plate_map.items():
    for plate in plate_names:
        output_feature_select_file = pathlib.Path(f"{output_dir}/{plate}_bulk_feature_selected.parquet")
        plate_df = pd.read_parquet(output_feature_select_file)
        
        plate_df["Metadata_plate"] = plate
        plate_df["Metadata_round"] = output_dir.name

        print(plate)
        # condition label 
        if plate in pfa_plates:
            plate_df["Metadata_condition"] = "synthemax_PFA"
        elif plate in laminin_plates:
            plate_df["Metadata_condition"] = "laminin_PFA"
        elif plate in synthemax_plates:
            plate_df["Metadata_condition"] = "synthemax"
        else:
            plate_df["Metadata_condition"] = "standard"


        if plate in mixed_condition_plates:
            # Merge by well (only affects wells present in the platemap)
            plate_df = plate_df.merge(
                platemap_13[["Metadata_Well", "Metadata_condition_well"]],
                on="Metadata_Well",
                how="left"
            )

            # Override plate-level condition where well-level data exists
            plate_df["Metadata_condition"] = (
                plate_df["Metadata_condition_well"].fillna(plate_df["Metadata_condition"])
            )

            # Clean up
            plate_df = plate_df.drop(columns="Metadata_condition_well")

        feat_cols = plate_df.columns[~plate_df.columns.str.contains("Metadata")].tolist()

        # Create a shuffled copy of the data (inherits the condition change)
        shuffled_plate_df = plate_df.copy()
        unique_time_points = plate_df["Metadata_time_point"].unique()

        # Shuffle feature columns within each time point
        for time_point in unique_time_points:
            mask = plate_df["Metadata_time_point"] == time_point
            for col in feat_cols:
                shuffled_plate_df.loc[mask, col] = (
                    plate_df.loc[mask, col]
                    .sample(frac=1, random_state=42)  # reproducible shuffle
                    .values
                )

        # Tag shuffled vs. original rows
        plate_df["Shuffled"] = "False"
        shuffled_plate_df["Shuffled"] = "True"

        # Combine original and shuffled data
        combined_plate_df = pd.concat([plate_df, shuffled_plate_df], ignore_index=True)

        # Pairwise comparison
        comparer = PairwiseCompareManager(
            _df=combined_plate_df.copy(),
            _comparator=PearsonsCorrelation(),
            _same_columns=[
                "Metadata_cell_line",
                "Metadata_seeding_density",
                "Metadata_time_point",
                "Metadata_condition",
                "Metadata_plate",     
                "Metadata_round",      
                "Shuffled",
            ],
            _different_columns=["Metadata_Well"],
            _feat_cols=feat_cols,
            _drop_cols=["Metadata_Concentration", "Metadata_Well"],
)

        combined_results.append(comparer())

    # Merge all plate results
    final_combined_df = pd.concat(combined_results, axis=0)


# In[6]:


#Save dataframe
save_dir = pathlib.Path("./results/round_1-4_pairwise_compare.parquet")
final_combined_df.to_parquet(save_dir)


# In[7]:


final_combined_df.head()


# In[8]:


avg_results = (
    final_combined_df.groupby(
        [
            "Metadata_cell_line__antehoc_group0",
            "Metadata_seeding_density__antehoc_group0",
            "Metadata_time_point__antehoc_group0",
            "Metadata_condition__antehoc_group0",
            "Metadata_plate__antehoc_group0",  
            "Metadata_round__antehoc_group0",   
            "Shuffled__antehoc_group0",
        ]
    )["pearsons_correlation"]
    .mean()
    .reset_index()
)

top_avg_results = (
    avg_results.loc[
        avg_results.groupby("Metadata_cell_line__antehoc_group0")["pearsons_correlation"].idxmax(),
        [
            "Metadata_cell_line__antehoc_group0", 
            "Metadata_seeding_density__antehoc_group0", 
            "Metadata_time_point__antehoc_group0", 
            "pearsons_correlation",
            "Shuffled__antehoc_group0"
        ]
    ]
)
print(top_avg_results)

# Step 2: Separate unshuffled and shuffled averages
unshuffled_avg = avg_results[avg_results["Shuffled__antehoc_group0"] == "False"]
shuffled_avg = avg_results[avg_results["Shuffled__antehoc_group0"] == "True"]

# Step 3: Merge the data to calculate the difference
difference_df = pd.merge(
    unshuffled_avg,
    shuffled_avg,
    on=[
        "Metadata_cell_line__antehoc_group0", 
        "Metadata_seeding_density__antehoc_group0", 
        "Metadata_time_point__antehoc_group0"
    ],
    suffixes=("_unshuffled", "_shuffled")
)

difference_df["correlation_difference"] = (
    difference_df["pearsons_correlation_unshuffled"] - 
    difference_df["pearsons_correlation_shuffled"]
)

# Step 4: Identify the top results by the largest difference
top_difference_results = (
    difference_df.groupby("Metadata_cell_line__antehoc_group0")
    .apply(lambda group: group.nlargest(1, "correlation_difference"))
    .reset_index(drop=True)
)

# Select relevant columns to display
top_difference_results = top_difference_results[
    [
        "Metadata_cell_line__antehoc_group0", 
        "Metadata_seeding_density__antehoc_group0", 
        "Metadata_time_point__antehoc_group0", 
        "correlation_difference"
    ]
]

# Display the results
top_difference_results.head(25)


# In[9]:


controlled_results = []

for plate in plate_names:
    output_feature_select_file = str(
        pathlib.Path(f"{output_dir}/{plate}_bulk_feature_selected.parquet")
    )
    plate_df = pd.read_parquet(output_feature_select_file)
    feat_cols = plate_df.columns[~plate_df.columns.str.contains("Metadata")].tolist()

    # Include Metadata_seeding_density in both datasets
    plate_df["Metadata_seeding_density"] = plate_df["Metadata_seeding_density"]

    # Define the control cell line
    control_cell_line = "U2OS"

    # Get unique cell lines excluding the control
    cell_lines = plate_df["Metadata_cell_line"].unique()
    cell_lines = [cl for cl in cell_lines if cl != control_cell_line]

    for cell_line in cell_lines:

        # Subset data for the target cell line and U2OS
        subset_df = plate_df[plate_df["Metadata_cell_line"].isin([cell_line, control_cell_line])]
        # Perform pairwise comparison on the combined data
        pearsons_comparator = PearsonsCorrelation()

        comparer = PairwiseCompareManager(
            _df=subset_df.copy(),
            _comparator=pearsons_comparator,
            _same_columns=[],
            _different_columns=["Metadata_cell_line", "Metadata_seeding_density", "Metadata_time_point", "Metadata_condition", "Metadata_Well"],
            _feat_cols=feat_cols,
            _drop_cols=["Metadata_Concentration", "Metadata_Well"],
        )

        micdf = comparer()
        controlled_results.append(micdf)

# Combine all results into a single dataframe
final_control_df = pd.concat(controlled_results, axis=0)


# In[10]:


final_control_df.head()


# In[11]:


avg_results = avg_results.rename(
    columns=lambda c: c.replace("__antehoc_group0", "")
)
avg_results.head(50)


# In[12]:


save_dir = pathlib.Path("./results/round_1-4_pearson_correlation.parquet")
avg_results.to_parquet(save_dir)

