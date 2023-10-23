import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import chi2_contingency
#Read data as df
species = pd.read_csv("species_info.csv")
print(species.head())

#Basic info
species.info()
#Basic statistics
species.describe()

observations = pd.read_csv("observations.csv")
print(observations.head())

#Basic info
observations.info()
#Basic statistics
observations.describe()

#Number of unique species in national park:
print(f"number of unique species:{species.scientific_name.nunique()}")
#Next is to find the number of category that are represented in the data
print(f"number of categories:{species.category.nunique()}")
print(f"categories:{species.category.unique()}")

species.groupby("category").size()

print(f"number of conservation_statuses:{species.conservation_status.nunique()}")
print(f"unique conservation_statuses:{species.conservation_status.unique()}")

#breakdown of conservation status, the nulls contain species which are not of concern
species.groupby("conservation_status").size()
species.conservation_status.isnull().sum()

#Observation data breakdown:
#Park details:
print(f"number of parks:{observations.park_name.nunique()}")
print(f"unique parks:{observations.park_name.unique()}")
#Observation numbers:
print(f"number of observations:{observations.observations.sum()}")

#Fill in blank values with "no intervention"

species.fillna('No intervention', inplace = True)
species.groupby("conservation_status").size()

#Remove no intervention and grouby the following to obtain the species by category and conservation status rearrange the dataframe

conservationCategory = species[species.conservation_status != "No intervention"]\
    .groupby(["conservation_status", "category"])['scientific_name']\
    .count()\
    .unstack()

print(conservationCategory)

ax = conservationCategory.plot(kind = 'bar', figsize = (8,6), stacked = True)
ax.set_xlabel("Conservation status")
ax.set_ylabel("number of species")

species["is protected"] = species.conservation_status != "No intervention"

category_counts = species.groupby(["category", "is protected"])\
    .scientific_name.nunique()\
        .reset_index()\
            .pivot(columns = "is protected"
                   ,index = "category"
                   ,values = "scientific_name")\
                .reset_index()
category_counts.columns = ["category", "not_protected", "protected"]

category_counts["percent protected"] = (category_counts.protected/(category_counts.not_protected+category_counts.protected))*100
print(category_counts)

#Use chi squared test for statistical evaluation relationship between mammal and bird
contingency_1 = [[30,146], [75,413]]  
     
print(contingency_1)
print(chi2_contingency(contingency_1))

#The results from the chi-squared test returns many values, the second value which is 0.69 is the p-value. 
#The standard p-value to test statistical significance is 0.05. 
#For the value retrieved from this test, the value of 0.69 is much larger than 0.05. In the case of mammals and birds there doesn't seem to be any significant relationship between them i.e. the variables independent.

# relationship between reptile and mammal
contingency_2 = [[30,146], [5,73]]
print(chi2_contingency(contingency_2))

# p value is less than 0.05 showing mammals have a higher rate of needed protection compared to reptiles


#The first step is to look at the the common names from species to get an idea of the most prevalent animals in the dataset. The data will be need to be split up into individual names.
from itertools import chain
import string

def remove_punctuations(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text

common_names = species[species.category == 'Mammal'].common_names.apply(remove_punctuations).str.split().tolist()
print(common_names[:6])

# cleanrows = []
# for item in common_names:
#     item = list(dict.fromkeys(item))
#     cleanrows.append(item)

# cleanrows[:6]

cleanrows = [list(set(row)) for row in common_names]
#print(cleanrows[:6])
# Collapse the list of lists into a single list

collapsed_list = [item for sublist in cleanrows for item in sublist]

#print(collapsed_list)

from collections import Counter

name_counts = Counter(collapsed_list)
# Convert the result to a DataFrame
name_counts_df = pd.DataFrame.from_dict(name_counts, orient='index', columns=['Count']).reset_index()

# Rename columns
name_counts_df.columns = ['Name', 'Count']

print(name_counts_df[name_counts_df.Name == 'Bat'])

# def animal_func():
    
#     animal_name = input("Enter animal name: ")
#     # CREATE THIS AS FUNCTION so we can check any animal per week by user input
#     species_mask = species.common_names.str.contains(rf"\b{animal_name}\b", regex = True, case = False)
#     #print(species[species["is bat"]])
    
#     bat_observations = observations.merge(species[species_mask])
#     print(bat_observations)
    
#     print(bat_observations.groupby('park_name').observations.sum().reset_index())
#     obs_by_park = bat_observations.groupby(['park_name', 'is protected']).observations.sum().reset_index()
#     print(obs_by_park)
    
#     plt.figure(figsize = (16,4))
#     sns.barplot(x = obs_by_park.park_name, y = obs_by_park.observations, hue = obs_by_park["is protected"])
#     plt.xlabel("National parks")
#     plt.ylabel("Number of Observations")
#     plt.title("Observations of"+ animal_name +" per Week")
#     plt.show()
    
def animal_func():
    while True:
            
        # Prompt the user for an animal name
        animal_name = input("Enter the name of an animal: ")
        
        try:
            # Create a boolean mask for the specified animal_name
            mask = species['common_names'].str.contains(rf"\b{animal_name}\b", regex=True, case=False)
            
            # Filter the 'species' DataFrame
            animal_species = species[mask]
    
            # Check if the filtered DataFrame is empty
            if animal_species.empty:
                raise ValueError(f"No species found for {animal_name}.")
                
            # Perform the merge operation
            animal_observations = observations.merge(animal_species)
    
            # Check if the merged DataFrame is empty
            if animal_observations.empty:
                raise ValueError(f"No observations found for {animal_name}.")
    
      
            print(animal_observations.groupby('park_name').observations.sum().reset_index())
            obs_by_park = animal_observations.groupby(['park_name', 'is protected']).observations.sum().reset_index()
            print(obs_by_park)
    
            plt.figure(figsize=(16, 4))
            sns.barplot(x=obs_by_park.park_name, y=obs_by_park.observations, hue=obs_by_park["is protected"])
            plt.xlabel("National parks")
            plt.ylabel("Number of Observations")
            plt.title(f"Observations of {animal_name} per Week")
            plt.show()
    # Exit the loop if everything is successful
            break
        
        except Exception as e:
            print(f"An error occurred: {str(e)}")

# Call the function to start the analysis
animal_func()
