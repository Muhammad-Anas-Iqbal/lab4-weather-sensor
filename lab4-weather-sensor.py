import pandas as pd
import numpy as np
from scipy.stats import zscore, skew, kurtosis

data = {
    "Hour": [
        "01:00", "02:00", "03:00", "04:00", "05:00",
        "06:00", "07:00", "08:00", "09:00", "10:00",
        "11:00", "12:00", "13:00", "14:00", "15:00"
    ],

    "Temperature": [
        20.5, 21.0, 21.4, 22.0, 22.5,
        np.nan, np.nan, np.nan,
        24.0, 24.5, 60.0, 25.0, 24.7, 24.2, 23.8
    ]
}

df = pd.DataFrame(data)

# original Temperature for final comparison
df["Raw_Temperature"] = df["Temperature"]

print("Original Weather Sensor Data:")
print(df[["Hour", "Temperature"]])


# (a) Descriptive Statistics Before Cleaning

print("\n--- Descriptive Statistics Before Cleaning ---")

print(df["Temperature"].describe())

# Drop missing values before skewness and kurtosis
valid_temp = df["Temperature"].dropna()

print("\nSkewness:")
print(skew(valid_temp))

print("\nKurtosis:")
print(kurtosis(valid_temp))


# Store original mean and standard deviation
# Pandas automatically ignores NaN values
mean_before = df["Temperature"].mean()
std_before = df["Temperature"].std()


# (b) Fill Missing Values Using Linear Interpolation

df["Cleaned_Temperature"] = df["Temperature"].interpolate(
    method="linear"
)

print("\n--- After Linear Interpolation ---")
print(df[["Hour", "Cleaned_Temperature"]])


# (c) Detect Outlier Using Z-score

# Calculate Z-scores after interpolation
temp_zscores = zscore(df["Cleaned_Temperature"])

# Threshold = 2
outlier_mask = np.abs(temp_zscores) > 2

print("\n--- Detected Outlier(s) ---")

if outlier_mask.any():
    print(
        df.loc[
            outlier_mask,
            ["Hour", "Cleaned_Temperature"]
        ]
    )
else:
    print("No outliers detected.")


# Replacing detected outlier with median

median_temp = df["Cleaned_Temperature"].median()

print("\nMedian Temperature:")
print(median_temp)

df.loc[
    outlier_mask,
    "Cleaned_Temperature"
] = median_temp


# (d) Z-score Standardization of Final Cleaned Data

df["Standardized_Temperature"] = zscore(
    df["Cleaned_Temperature"]
)


# Mean and standard deviation after cleaning
mean_after = df["Cleaned_Temperature"].mean()
std_after = df["Cleaned_Temperature"].std()


# Before / After Comparison

print("\n--- Final Before/After Comparison ---")

final_table = df[
    [
        "Hour",
        "Raw_Temperature",
        "Cleaned_Temperature",
        "Standardized_Temperature"
    ]
]

print(final_table)


# Mean and Standard Deviation Comparison

print("\n--- Mean and Standard Deviation ---")

print("Before Cleaning:")
print("Mean =", mean_before)
print("Standard Deviation =", std_before)

print("\nAfter Cleaning:")
print("Mean =", mean_after)
print("Standard Deviation =", std_after)