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
#change time to x and observation to y in function
def line_chart(data,time,observation,categories):
    #convert time to datetime64 if data is string
    # data[time] = data[time].apply(pd.to_datetime)
    # data.set_index(time, inplace = True)
    
    #create line plot of data
    plt.figure(figsize=(10, 6))
    sns.lineplot(data, x = time, y=observation, hue=categories, marker= 'o')
    
    # Perform linear regression and plot the line of best fit for each category
    category = data[categories].unique()
    for x in category:
        subset = data[data[categories] == x]
        X = sm.add_constant(subset[time].index)  # Using index as predictor
        model = sm.OLS(subset[observation], X).fit()
        plt.plot(subset[time], model.predict(X), color = 'black', linestyle = '--')
        
    
    plt.xlabel(time)
    plt.xticks(data[time], rotation=45)
    plt.ylabel(observation)
    plt.legend(title='Country')
    plt.title('Scatter Plot of {} vs. {}'.format(time, observation))

    
    plt.plot()
    plt.tight_layout()
    plt.show()
    plt.clf()
    

#Has life expectancy increased over time in the six nations? (answer is yes)
line_chart(df,"Year","Life expectancy at birth (years)", "Country")

### graph checks
print(df.Year.max(), df.Year.min())
print(df["Life expectancy at birth (years)"].max(), df["Life expectancy at birth (years)"].min())
print(df.Country.unique())

#Has GDP increased over time in the six nations? (answer is for mexico,chile and zimbabwe no.. for the other three yes)
line_chart(df,"Year","GDP", "Country")


#Scatter chart showing correlation between a quantative and categorical variable (relationship chart)
def scatter_chart(data,x,y,category):
    
    log_y = data[y][data[y]>0]
    log_y = np.log(log_y)
    
    # scatter plot with a visual cue
    sns.scatterplot(x=x, y=log_y, hue = category, data=data)
    plt.title('Scatter Plot of GDP vs. Life Expectancy')
    plt.xlabel('GDP')
    plt.ylabel('Life Expectancy')

    plt.legend(title='Country')
    plt.tight_layout()
    plt.show()
    plt.clf()


scatter_chart(df,'GDP',"Life expectancy at birth (years)",'Country')    

