import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import statsmodels.api as sm
#Read data as df
df = pd.read_csv("all_data.csv")
#Output first 5 rows of data
print(df.head())

#Basic info
df.info()
#Basic statistics
df.describe()

# time series line chart for multiple catagorical variables over time against quantative variable (comparison chart)

def line_chart(data, time, observation, categories):
        graph = sns.FacetGrid(df, col=categories, col_wrap=3,
                          hue = categories, sharey = False)
    
        graph = (graph.map(sns.lineplot,time,observation)
             .add_legend()
             .set_axis_labels(time,observation)
             .set_titles(col_template="{col_name}"))
        for ax in graph.axes.flat:
            country = ax.get_title().split('=')[-1].strip()
            title = "{} for {}".format(observation, country)
            ax.set_title(title)

    
        

#Has life expectancy increased over time in the six nations? (answer is yes)
line_chart(df,"Year","Life expectancy at birth (years)", "Country")

### graph checks
print(df.Year.max(), df.Year.min())
print(df["Life expectancy at birth (years)"].max(), df["Life expectancy at birth (years)"].min())
print(df.Country.unique())

#Has GDP increased over time in the six nations? (answer is yes)
line_chart(df,"Year","GDP", "Country")



#Scatter chart showing correlation between a quantative and categorical variable (relationship chart) for multiple charts
def scatter_chart(data,x,y,categories):
    graph = sns.FacetGrid(df, col = categories, col_wrap = 3, hue = categories, sharey = False, sharex= False)
    graph = (graph.map(sns.scatterplot,x,y)
             .add_legend()
             .set_axis_labels(x,y)
             .set_titles(col_template="{col_name}"))
    for ax in graph.axes.flat:
        country = ax.get_title().split('=')[-1].strip()
        title = "{} for {}".format(x, country)
        ax.set_title(title)



scatter_chart(df,"Life expectancy at birth (years)",'GDP','Country')

avg_life_expectancy = df.groupby ('Country')["Life expectancy at birth (years)"].mean().reset_index()
avg_life_expectancy.rename(columns={"Life expectancy at birth (years)": 'AverageLifeExpectancy'}, inplace=True)
df = df.merge(avg_life_expectancy, on='Country')
avg_gdp = df.groupby ('Country')['GDP'].mean().reset_index()
avg_gdp.rename(columns={"GDP": 'AverageGDP'}, inplace=True)
df = df.merge(avg_gdp, on='Country')

#Plots 1 categorical variable on y axis against two different x axis as a bar chart
def bar_chart(data,x,y,x2):
    
    fig, axes = plt.subplots(1,2, sharey= True, figsize = (12,6))
    # Plot the first subplot (Left)
    sns.barplot(ax=axes[0], y=y, x=x, data=data)
    axes[0].set_title("Average Life Expectancy by Country")
    axes[0].set_xlabel("Average Life Expectancy")
    
    # Plot the second subplot (Right)
    sns.barplot(ax=axes[1], y=y, x=x2, data=data)
    axes[1].set_title("Average GDP by Country")
    axes[1].set_xlabel("Average GDP")
    
    # Adjust layout and display the plots
    plt.tight_layout()
    plt.show()
    plt.clf()

bar_chart(df, "AverageLifeExpectancy", "Country", "AverageGDP")
#Used to plot 1 categorical variable on y axis against two different x axis
def distribution_plots(data,x,y,x2):
    
    fig, axes = plt.subplots(1,2, sharey= True, figsize = (12,6))
    # Plot the first subplot (Left)
    sns.violinplot(ax=axes[0], y=y, x=x, data=data, inner = None)
    sns.swarmplot(ax=axes[0], y=y, x=x, data=df, color='k', alpha=0.7)
    axes[0].set_title("Distribution of Life Expectancy by Country")
    axes[0].set_xlabel("Life Expectancy")
    
    # Plot the second subplot (Right)
    sns.violinplot(ax=axes[1], y=y, x=x2, data=df, inner = None)
    sns.swarmplot(ax=axes[1], y=y, x=x2, data=df, color='k', alpha=0.7)
    axes[1].set_title("Distribuition of GDP by Country")
    axes[1].set_xlabel("GDP")
    
    # Adjust layout and display the plots
    plt.tight_layout()
    plt.show()
    plt.clf()

distribution_plots(df, "Life expectancy at birth (years)", "Country", "GDP")
