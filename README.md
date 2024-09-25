# 100plus-Normalization-Techniques
This comprehensive list provides a variety of normalization and transformation techniques suitable for different datasets and machine learning tasks. Selecting the appropriate method is crucial for optimizing model performance and accuracy.

Comprehensive List of Normalization and Transformation Techniques in Machine Learning
Min-Max Normalization (Min-Max Scaling)

Formula: 
𝑋
′
=
𝑋
−
min
(
𝑋
)
max
(
𝑋
)
−
min
(
𝑋
)
X 
′
 = 
max(X)−min(X)
X−min(X)
​
 
Range: [0, 1] (or any desired range).
Use Case: When the feature distribution is not Gaussian and preserving relationships between data points is important.
Z-Score Normalization (Standardization)

Formula: 
𝑋
′
=
𝑋
−
𝜇
𝜎
X 
′
 = 
σ
X−μ
​
 
Where: 
𝜇
μ is the mean, and 
𝜎
σ is the standard deviation.
Range: Mean of 0 and standard deviation of 1.
Use Case: For algorithms that assume normally distributed data (e.g., SVM, Logistic Regression).
Standard Scaler

Description: Similar to Z-Score Normalization, transforms features to have a mean of 0 and standard deviation of 1.
Use Case: Commonly used in preprocessing for many machine learning models.
Gaussian Transformation (Gaussian Scaling)

Description: Transforms data to follow a Gaussian distribution.
Use Case: Useful for algorithms that rely on assumptions of normality.
Power Transformation

Examples: Box-Cox and Yeo-Johnson transformations.
Use Case: Stabilizes variance and makes data more Gaussian-like.
Quantile Transformation

Description: Maps the data to a uniform or normal distribution.
Use Case: Makes the model robust against outliers and transforms the data to follow a specific distribution.
Log Transformation

Formula: 
𝑋
′
=
log
⁡
(
𝑋
+
𝑐
)
X 
′
 =log(X+c)
Where: 
𝑐
c is a constant to avoid log(0).
Use Case: Effective for transforming right-skewed distributions.
Robust Scaling

Formula: 
𝑋
′
=
𝑋
−
median
(
𝑋
)
𝐼
𝑄
𝑅
X 
′
 = 
IQR
X−median(X)
​
 
Where: 
𝐼
𝑄
𝑅
IQR is the interquartile range.
Use Case: Useful for datasets with outliers.
MaxAbs Scaling

Formula: 
𝑋
′
=
𝑋
max
(
∣
𝑋
∣
)
X 
′
 = 
max(∣X∣)
X
​
 
Range: Scales to [-1, 1].
Use Case: Useful for sparse datasets.
Normalization

Description: Scales the data to a specific norm (e.g., L1 or L2 norm).
Use Case: Ensures that input features have the same scale.
t-SNE (t-Distributed Stochastic Neighbor Embedding)

t-SNE 2D: Reduces dimensionality for visualization purposes in two dimensions.
t-SNE 3D: Reduces dimensionality for visualization in three dimensions.
Use Case: Effective for visualizing high-dimensional data while preserving local structure.
Principal Component Analysis (PCA)

Description: A dimensionality reduction technique that transforms data into a lower-dimensional space by identifying the directions (principal components) that maximize variance.
Use Case: Useful for reducing the number of features while retaining as much information as possible.
Robust Z-Score Normalization

Formula: 
𝑋
′
=
𝑋
−
median
(
𝑋
)
MAD
X 
′
 = 
MAD
X−median(X)
​
 
Where: 
MAD
MAD is the Median Absolute Deviation.
Use Case: Similar to Z-score but less sensitive to outliers.
Softmax Normalization

Formula: 
𝑋
′
=
𝑒
𝑋
𝑖
∑
𝑗
𝑒
𝑋
𝑗
X 
′
 = 
∑ 
j
​
 e 
X 
j
​
 
 
e 
X 
i
​
 
 
​
 
Use Case: Converts logits (raw prediction scores) to probabilities, often used in classification problems.
Normalization followed by Log Transformation

Description: Normalize the data first, then apply log transformation.
Use Case: Effective when you want to handle skewness after scaling.
Standardization followed by Log Transformation

Description: Standardize the data and then apply log transformation.
Use Case: Helpful when both standardization and handling skewness are required.
Min-Max Normalization followed by Gaussian Transformation

Description: Scale data to a specific range, then apply Gaussian transformation.
Use Case: Effective for preserving a specific range while achieving normality.
Standardization followed by Gaussian Transformation

Description: Standardize the data and then transform it to follow a Gaussian distribution.
Use Case: Helpful when normality is required after standardization.
Robust Scaling followed by Gaussian Transformation

Description: Use robust scaling and then apply Gaussian transformation.
Use Case: Effective for datasets with significant outliers while achieving normality.
Normalization followed by Gaussian Transformation

Description: Normalize the data first, then apply Gaussian transformation.
Use Case: Useful for preparing data for algorithms sensitive to normality.
Binarization

Formula: 
𝑋
′
=
{
1
if 
𝑋
>
threshold
0
otherwise
X 
′
 ={ 
1
0
​
  
if X>threshold
otherwise
​
 
Use Case: Converts numerical data into binary format based on a threshold.
Feature Scaling with L1 and L2 Normalization

L1 Normalization: Scales the data to have unit L1 norm (sum of absolute values).
L2 Normalization: Scales the data to have unit L2 norm (Euclidean norm).
Sqrt Transformation

Formula: 
𝑋
′
=
𝑋
+
𝑐
X 
′
 = 
X+c
​
 
Where: 
𝑐
c is a constant to avoid issues with zero values.
Use Case: Reduces the skewness of data, particularly for right-skewed distributions.
Feature Scaling with Quantile Normalization

Description: Adjusts the distribution of a dataset to be similar to a target distribution (often uniform or Gaussian).
Use Case: Effective in genomic data processing.
Correlation Matrix Normalization

Description: Normalizes the correlation matrix of features to adjust for different scales.
Use Case: Useful in exploratory data analysis and feature selection.
K-Means Normalization

Description: Applies K-Means clustering to normalize features based on cluster centroids.
Use Case: Effective when the dataset is large and contains clusters.
Gaussian Mixture Model (GMM) Normalization

Description: Models the data as a mixture of several Gaussian distributions.
Use Case: Used for clustering and density estimation.
