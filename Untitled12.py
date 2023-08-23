#!/usr/bin/env python
# coding: utf-8

# # Import library for read the the dataset

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv("Mall_Customers.csv")


# In[3]:


df


# # Analyze the data

# In[4]:


df.isnull().sum()


# In[6]:


df.describe()


# In[7]:


print(df.columns)


# In[9]:


print(df.dtypes)


# In[10]:


# Count the number of males and females in the 'Genre' column
gender_counts = df['Genre'].value_counts()

# Print the result
print(gender_counts)


# In[11]:


# Find the highest annual income
max_income = df['Annual Income (k$)'].max()

# Find the highest spending score
max_spending_score = df['Spending Score (1-100)'].max()

# Print the results
print(f'Highest Annual Income: ${max_income}k')
print(f'Highest Spending Score: {max_spending_score}')


# In[12]:


# Filter the DataFrame for Male and Female separately
male_df = df[df['Genre'] == 'Male']
female_df = df[df['Genre'] == 'Female']

# Sort the DataFrames by 'Annual Income (k$)' and 'Spending Score (1-100)'
male_top_income = male_df.nlargest(10, 'Annual Income (k$)')
male_top_spending = male_df.nlargest(10, 'Spending Score (1-100)')

female_top_income = female_df.nlargest(10, 'Annual Income (k$)')
female_top_spending = female_df.nlargest(10, 'Spending Score (1-100)')

# Print the top 10 males and females with the highest income and spending
print("Top 10 Males with Highest Income:")
print(male_top_income)

print("\nTop 10 Males with Highest Spending:")
print(male_top_spending)

print("\nTop 10 Females with Highest Income:")
print(female_top_income)

print("\nTop 10 Females with Highest Spending:")
print(female_top_spending)


# # visualization

# In[18]:


import matplotlib.pyplot as plt
import seaborn as sns

# Assuming you have your data in a DataFrame called 'df'
# You can read your data into a DataFrame like this:
# df = pd.read_csv('your_dataset.csv')

# Create a figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Histogram of Age
axes[0, 0].hist(df['Age'], bins=20, color='skyblue', edgecolor='black')
axes[0, 0].set_title('Age Distribution')
axes[0, 0].set_xlabel('Age')
axes[0, 0].set_ylabel('Frequency')

# Plot 2: Scatter plot of Annual Income vs. Spending Score
axes[0, 1].scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'], color='salmon', alpha=0.6)
axes[0, 1].set_title('Annual Income vs. Spending Score')
axes[0, 1].set_xlabel('Annual Income (k$)')
axes[0, 1].set_ylabel('Spending Score (1-100)')

# Plot 3: Bar plot of Gender distribution
gender_counts = df['Genre'].value_counts()
axes[1, 0].bar(gender_counts.index, gender_counts.values, color='lightgreen')
axes[1, 0].set_title('Gender Distribution')
axes[1, 0].set_xlabel('Gender')
axes[1, 0].set_ylabel('Count')

# Plot 4: Box plot of Annual Income by Gender
df.boxplot(column='Annual Income (k$)', by='Genre', ax=axes[1, 1], vert=False)
axes[1, 1].set_title('Annual Income by Gender')
axes[1, 1].set_xlabel('Annual Income (k$)')
axes[1, 1].set_ylabel('Gender')

# Adjust spacing between subplots
plt.tight_layout()

# Display the plots
plt.show()


# In[16]:


genre_counts = df['Genre'].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(genre_counts, labels=genre_counts.index, autopct='%1.1f%%', colors=['lightblue', 'lightcoral'])
plt.title('Genre Distribution')
plt.show()


# In[20]:


age_bins = [18, 25, 35, 50, 70, 100]
age_labels = ['18-24', '25-34', '35-49', '50-69', '70+']
df['Age Group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels)
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Age Group', hue='Genre', palette={'Male': 'blue', 'Female': 'red'})
plt.title('Count of Males and Females by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.show()


# In[21]:


sns.pairplot(df, hue='Genre', palette={'Male': 'blue', 'Female': 'red'})
plt.title('Pair Plot of Numerical Variables')
plt.show()


# In[22]:


plt.figure(figsize=(8, 6))
df.boxplot(column='Spending Score (1-100)', by='Age')
plt.title('Spending Score by Age')
plt.xlabel('Age')
plt.ylabel('Spending Score (1-100)')
plt.show()


# In[24]:


numeric_df = df.select_dtypes(include=['number'])

# Create a correlation matrix for the numeric columns
correlation_matrix = numeric_df.corr()

# Set the figure size
plt.figure(figsize=(8, 6))

# Create the heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)

# Set the title
plt.title('Correlation Heatmap')

# Show the plot
plt.show()


# # Machine Learning model 

# In[35]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder


# In[36]:


df = pd.read_csv("Mall_Customers.csv")


# In[37]:


# Separate numerical and categorical columns
numerical_cols = df.select_dtypes(include=[np.number])
print(numerical_cols)


# In[38]:


categorical_cols = df.select_dtypes(exclude=[np.number])
print(categorical_cols)


# In[40]:


# Perform one-hot encoding for categorical columns
encoder = OneHotEncoder()
categorical_encoded = encoder.fit_transform(categorical_cols).toarray()
print(categorical_encoded)


# In[42]:


# Combine numerical and encoded categorical data
scaled_data = np.hstack((numerical_cols, categorical_encoded))
print(scaled_data)


# In[43]:


# Determine the optimal number of clusters using the Elbow method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)


# In[44]:


# Plot the Elbow method
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')  # Within-Cluster-Sum-of-Squares
plt.show()


# In[46]:


# Choose the optimal number of clusters (let's assume it's 3)
optimal_num_clusters = 3


# In[48]:


# Apply K-Means clustering with the chosen number of clusters
kmeans = KMeans(n_clusters=optimal_num_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)

cluster_labels = kmeans.fit_predict(scaled_data)
print(cluster_labels)


# In[50]:


# Add cluster labels to the original dataset
df['Cluster'] = cluster_labels

# Print the results
print(df.head())


# In[ ]:




