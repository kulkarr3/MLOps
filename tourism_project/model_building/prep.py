import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("tourism_project/data/tourism.csv")


# Remove unnecessary columns
if 'Unnamed: 0' in df.columns:
    df = df.drop('Unnamed: 0', axis=1)

# Handle missing values
print("Missing values before cleaning:")
print(df.isnull().sum()[df.isnull().sum() > 0])


# Fill missing values
numerical_cols = ['Age', 'DurationOfPitch', 'NumberOfFollowups', 'PreferredPropertyStar',
                 'NumberOfTrips', 'PitchSatisfactionScore', 'NumberOfChildrenVisiting', 'MonthlyIncome']

for col in numerical_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

categorical_cols = ['TypeofContact', 'Occupation', 'Gender', 'MaritalStatus',
                   'ProductPitched', 'Designation']

for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')

# Fix data inconsistencies
df['Gender'] = df['Gender'].replace('Fe Male', 'Female')

print("Missing values after cleaning:")
print(df.isnull().sum()[df.isnull().sum() > 0])

# Feature engineering
print("Feature engineering...")

# Create income categories
df['IncomeCategory'] = pd.cut(df['MonthlyIncome'],
                             bins=[0, 15000, 25000, 35000, float('inf')],
                             labels=[0, 1, 2, 3])  # Use numeric labels

# Create age groups
df['AgeGroup'] = pd.cut(df['Age'],
                       bins=[0, 25, 35, 45, 55, float('inf')],
                       labels=[0, 1, 2, 3, 4])  # Use numeric labels

                       # Encode categorical variables
label_encoders = {}
categorical_columns = df.select_dtypes(include=['object']).columns

for col in categorical_columns:
    if col != 'CustomerID':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

print(f"Data preprocessing completed! Final shape: {df.shape}")

# Split data
print("Splitting data...")
X = df.drop(['CustomerID', 'ProdTaken'], axis=1)
y = df['ProdTaken']

# stratify=y keeps the (imbalanced) failure ratio consistent across splits
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

Xtrain.to_csv("tourism_project/data/Xtrain.csv", index=False)
Xtest.to_csv("tourism_project/data/Xtest.csv", index=False)
ytrain.to_csv("tourism_project/data/ytrain.csv", index=False)
ytest.to_csv("tourism_project/data/ytest.csv", index=False)

print("Data prepared: train/test splits written.")
print("Type values kept as:", sorted(X["TypeofContact"].unique()))
