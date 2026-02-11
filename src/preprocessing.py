import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import KBinsDiscretizer


def extract_cabin_deck(X):
    """
    Extracts the first letter of the Cabin (A, B, C, etc.).
    Fills missing values with 'M' (Missing).
    """
    # X is a DataFrame. We apply the logic to the column.
    # Convert to string, take the first character.
    # If it was NaN, it becomes 'n' (from string 'nan'), so we handle NaNs explicitly first.
    
    # Ensure it's a Series/DataFrame
    if isinstance(X, pd.DataFrame):
        X = X.iloc[:, 0]
        
    # Fill NaN with 'M', then get first char
    return pd.DataFrame(X.fillna('M').astype(str).str[0])

def extract_ticket_prefix(X):
    """
    Extracts the prefix from the ticket (e.g., 'A/5' from 'A/5 21171').
    If it's just a number (e.g., '113803'), return 'Numeric'.
    """
    if isinstance(X, pd.DataFrame):
        X = X.iloc[:, 0]
        
    # Helper logic: split by space, if len > 1, take the first part.
    def get_prefix(ticket):
        parts = str(ticket).split()
        if len(parts) > 1:
            return parts[0] # Return the prefix (e.g., "A/5")
        else:
            return "Numeric" # No prefix found
            
    return pd.DataFrame(X.apply(get_prefix))

def add_family_features(df):
    # Create a copy so we don't modify the original data
    df_out = df.copy()
    
    # 1. Create Family Size
    df_out['FamilySize'] = df_out['SibSp'] + df_out['Parch'] + 1
    
    return df_out

def family_size_feature():
    """
    Returns a FunctionTransformer which creates a family size feature
    """
    return FunctionTransformer(add_family_features, validate=False)

def engineer_features(df):
    df_out = df.copy()
    
    # --- 1. Family Logic ---
    df_out['FamilySize'] = df_out['SibSp'] + df_out['Parch'] + 1
    
    # --- 2. Title Logic (The New Part) ---
    # Regex: Find a word ending in a dot (e.g., "Mr.", "Dr.")
    df_out['Title'] = df_out['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    
    # Fix the French/Old titles
    df_out['Title'] = df_out['Title'].replace(['Mlle', 'Ms'], 'Miss')
    df_out['Title'] = df_out['Title'].replace('Mme', 'Mrs')
    
    # Group the rare ones (Doctors, Reverend, Royalty)
    rare_titles = ['Lady', 'Countess','Capt', 'Col', 'Don', 
                   'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']
    df_out['Title'] = df_out['Title'].replace(rare_titles, 'Rare')
    
    return df_out

def feat_eng_transformer():
    return FunctionTransformer(engineer_features, validate=False)

def sex_transformer():
    """
    Returns an OrdinalEncoder() pipeline
    """
    return Pipeline(steps=[
        ("ordinal", OrdinalEncoder())
    ])

def age_transformer():
    """
    Return a SimpleImputer() pipeline on the mean
    """
    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean"))
    ])

def embarked_transformer():
    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="C")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

def cabin_transformer():
    return Pipeline(steps=[
        ("extract_deck", FunctionTransformer(extract_cabin_deck, validate=False, feature_names_out="one-to-one")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

def ticket_transformer():
    return Pipeline(steps=[
        ("extract_prefix", FunctionTransformer(extract_ticket_prefix, validate=False, feature_names_out="one-to-one")),
        # Use handle_unknown='ignore' because new unique prefixes might appear in test data
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=0.01)) 
        # min_frequency=0.01 (sklearn 1.1+) groups rare prefixes into "infrequent_sklearn"
    ])

def title_transformer():
    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

def create_family_survival_feature(df_train, df_test):
    # 1. Combine Train and Test to see the whole picture
    # We need this to find "The Brown Family" even if half are in Train and half in Test
    df_train['dataset'] = 'train'
    df_test['dataset'] = 'test'
    
    # Create the 'Survived' column in test so they match (fill with NaN)
    df_test['Survived'] = np.nan 
    
    # 2. Combine securely (ignore_index=True fixes the index 0..1309)
    full_data = pd.concat([df_train, df_test], ignore_index=True)
    
    # 2. Initialize the feature with a default value of 0.5
    # 0.5 means "We have no clue" (unknown family or solo traveler)
    full_data['Family_Survival'] = 0.5

    # 2. But if you are SOLO, set to -1 (Not Applicable)
    # This forces the model to ignore this feature for solo travelers
    full_data['FamilySize'] = full_data['SibSp'] + full_data['Parch'] + 1
    full_data.loc[full_data['FamilySize'] == 1, 'Family_Survival'] = -1
    
    # 3. Extract Surname to help identify families
    full_data['Surname'] = full_data['Name'].apply(lambda x: x.split(',')[0])
    
    # 4. Grouping Logic
    # We will group by Surname + Fare (to distinguish between different "Smiths")
    # And then by Ticket (which is even more reliable)
    
    for grp, grp_df in full_data.groupby(['Surname', 'Fare']):
        if (len(grp_df) > 1):
            # A family was found!
            # Check if anyone in this group has a KNOWN survival status (from Train set)
            if (grp_df['Survived'].notnull().any()):
                # If anyone survived, we assume the family had a way out
                if (grp_df['Survived'].max() == 1): 
                    full_data.loc[grp_df.index, 'Family_Survival'] = 1
                # If everyone known died, we assume the family was trapped
                else:
                    full_data.loc[grp_df.index, 'Family_Survival'] = 0
                    
    # 5. Overwrite with Ticket Grouping (Higher Priority)
    # Tickets are stronger evidence than Surnames (covers friends/nannies too)
    for _, grp_df in full_data.groupby('Ticket'):
        if (len(grp_df) > 1):
            if (grp_df['Survived'].notnull().any()):
                if (grp_df['Survived'].max() == 1):
                    full_data.loc[grp_df.index, 'Family_Survival'] = 1
                else:
                    full_data.loc[grp_df.index, 'Family_Survival'] = 0
                    
    # 6. Clean up
    # Reset the "Survived" column in Test set to NaN (just in case)
    # Then split back into Train and Test
    train_with_feature = full_data[full_data['dataset'] == 'train'].copy()
    test_with_feature = full_data[full_data['dataset'] == 'test'].copy()
    
    return train_with_feature, test_with_feature
