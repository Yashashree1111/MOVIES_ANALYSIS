
# importing all the required libraries


import numpy as np
import pandas as pd
import pandas_profiling
import matplotlib.pyplot as plt         # Plotting library for Python programming language and it's numerical mathematics extension NumPy
import seaborn as sns                   # Provides a high level interface for drawing attractive and informative statistical graphics



#Importing Dataset:

md = pd.read_csv("https://raw.githubusercontent.com/insaid2018/Term-1/master/Data/Projects/1000%20movies%20data.csv")


# first five rows
md.head(15)

# describes the Shape of the Dataset
md.shape


# - Next, I will do descriptive statistics for numerical variables.
# - This will help in finding distribution, Standard Deviations and min-max of numerical columns

md.describe()

md.info()

# we will replace missing Values with mean


md['Revenue (Millions)'].mean()         # Mean of the column Revenue_(Millions).

md['Revenue (Millions)']= md['Revenue (Millions)'].replace(np.NaN , md['Revenue (Millions)'].mean())

md['Metascore']= md['Metascore'].replace(np.NaN , md['Metascore'].mean())


from pandas import DataFrame                       #Splitting the column 'Genre' into 3 columns
df=pd.DataFrame(md)
GenSplt=md['Genre'].str.split(",", n=-1, expand = True)
GenSplt.columns = ['G1','G2','G3']

GenSplt.columns = ['G1','G2','G3']
GenOne= GenSplt['G1'].append(GenSplt['G2'])        #Appending all the 3 columns to single column
GenOne= GenOne.append(GenSplt['G3'])
GenOne= GenOne.dropna(axis = 0, how ='any')        #Dropping NaN Values
df2=pd.DataFrame(GenOne)                           #Making the splitted&appended column to DataFrame
df2.columns=['G']
#df2
md.head(10)


# Splitting Actors column for more insights:

from pandas import DataFrame                       #Splitting the column 'Actors' into 4 columns
df=pd.DataFrame(md)
ActSplt=md['Actors'].str.split(",", n=-1, expand = True)
ActSplt.columns = ['A1','A2','A3','A4']                 #Giving names to the columns
#GenSplt



ActOne= ActSplt['A1'].append(ActSplt['A2'])        #Appending all the 4 columns to single column
ActOne= ActOne.append(ActSplt['A3'])
ActOne= ActOne.append(ActSplt['A3'])
ActOne= ActOne.dropna(axis = 0, how ='any')        #Dropping NaN Values
df10=pd.DataFrame(ActOne)                           #Making the splitted&appended column to DataFrame
df10.columns=['A']

#Dropping the column unsoughtful for data insights.

md.drop(['Description'], axis=1)   # Dropping the Description column for no insights to analyze data.

import pandas_profiling                          # Get a quick overview for all the variables using pandas_profiling.
profile = pandas_profiling.ProfileReport(md)
profile.to_file("1000movies_after_preprocessing.html")     # HTML file will be downloaded to local workspace.


# In 1000movies_after_preprocessing.html report, observations:
# - In the Dataset info, Total __Missing(%)__ = __0.0%__
# - Number of __variables__ = __12__
# - Observe the newly created variable __G__.

#4.1 How many movies are released per year over the period: 2006-2016?


plt.figure(figsize=(20,20))
MovTrnd = sns.factorplot("Year", data=md, aspect=2, kind="count", color='Skyblue')
 # What are the top 10 Movies with Highest Revenue?


plt.figure(figsize=(10,5))
df4=md.sort_values("Revenue (Millions)", ascending=False).head(10)
df_4 = sns.barplot(x="Revenue (Millions)", y="Title",hue='Year', linewidth=0, data=df4,ci= None)
#plt.title('Top 10 Movies with Highest Revenue',fontsize=18,fontweight="bold")
df_4.set_xlim(990,1000)
plt.legend(bbox_to_anchor=(1.05, 0.9), loc=2, borderaxespad=0.)  #Moving the hue to right for graph visibility.
plt.show()

# What is the Revenue trend of the Movies over the Years?

x =[]
y =[]

for i in md['Year'].unique():
    y.append(md[md['Year']==i]['Revenue (Millions)'].sum())
    x.append(i)


z = pd.DataFrame(list(zip(x, y)), columns=['Year','Revenue_Sum'])
z


plt.figure(figsize=(10,7))
TrenMov = sns.lineplot(x="Year", y="Revenue_Sum",data=z,ci= None)
#Displays cummulative revenue generated on the movies per year.


# - The Linegraph shows the sum of Revenues per Year.
# - Total Revenue was gradually increasing with Years.
# - Key point is that, There is sudden increase in the Revenue in 2013.
# - And then, stable or slightly decreased in 2014.
# - The year 2016 generated highest total revenue.



plt.figure(figsize=(20,20))
sns.jointplot("Rating","Revenue (Millions)", data=md, kind='hex', color='DarkBlue')
#



## Analysing Genres of all the movies individually.

import matplotlib.pyplot as plt                     #Finding individual uniques out of profiled genre column
df2.G.unique()
df2.pivot_table(index=['G'], aggfunc='size')

Genre_names=['Action', 'Adventure', 'Horror', 'Animation', 'Comedy',
       'Biography', 'Drama', 'Crime', 'Romance', 'Mystery', 'Thriller',
       'Sci-Fi', 'Fantasy', 'Family', 'History', 'Music', 'Western',
       'War', 'Musical', 'Sport']
Genre_Size=[303,259,49,81,279,150,513,51,101,29,119,16,5,106,141,120,18,195,13,7]

plt.figure(figsize=(15,15))
fig, gen = plt.subplots()
gen.axis('equal')
Genpie, _ = gen.pie(Genre_Size, radius=2.5, labels=Genre_names, colors = ['skyblue', 'gold','red','orange','blue','pink','violet','grey','gold','yellowgreen','brown'])
plt.setp(Genpie, width=1, edgecolor='white')
plt.margins(0,0)



# Who are the Highest and Lowest Revenue(cummulative) generated Directors??

x =[]
y =[]

for i in md['Director'].unique():
    y.append(md[md['Director']==i]['Revenue (Millions)'].sum())
    x.append(i)

df6 = (pd.DataFrame(list(zip(x, y)), columns=['Director','Revenue_Sum'])).sort_values('Revenue_Sum', ascending=False)


df7 = df6.tail(5)  #Lowest Revenue generated directors
df6 = df6.head(5)  #Highest Revenue generated directors



sns.barplot(x="Revenue_Sum", y="Director", data=df6, ci= None)
plt.show()

md.Director.value_counts().head(15).plot.bar()
#sns.barplot(x="Revenue_Sum", y="Director", data=df7, ci= None)


md[md['Director']=='Alexandre Aja']



# Do all the parameters(Votes, Metascore, Rating) speak the same about a movie?

#plt.figure(figsize=(8,8))
a=sns.scatterplot(x="Rating", y="Votes",data=md, ci= None)



# ### Which Actors had appeared in highest number of Movies?

j = df10.A.unique()
k = df10.pivot_table(index=['A'], aggfunc='size')
ab = ((pd.DataFrame(zip(j,k),columns=['Actors','MovCount'])).sort_values('MovCount',ascending = False)).head(50)


plt.figure(figsize=(10,10))
sns.barplot('MovCount','Actors', data = ab)


# - 'Malin Akarmen' is the only one who acted in 14 Movies.
# - And the next 5 actors: Mia, Susan, Michael, Laura, Roman and Joan are part of 12 movies each.


# ### What are the movies with highest Rating, Metascore and Votes?

# Combinations
"""    [['Votes','Metascore','Rating'],['Votes','Rating','Metascore'],
        ['Rating','Metascore','Votes'],['Rating','Votes','Metascore'],
        ['Metascore','Rating','Votes'],['Metascore','Votes','Rating']]
"""

tmd = pd.DataFrame()
tmd = tmd.append((md.sort_values(by=['Votes','Metascore','Rating'], ascending = False)).head(10))
tmd = tmd.append((md.sort_values(by=['Votes','Rating','Metascore'], ascending = False)).head(10))
tmd = tmd.append((md.sort_values(by=['Rating','Metascore','Votes'], ascending = False)).head(10))
tmd = tmd.append((md.sort_values(by=['Rating','Votes','Metascore'], ascending = False)).head(10))
tmd = tmd.append((md.sort_values(by=['Metascore','Rating','Votes'], ascending = False)).head(10))
tmd = tmd.append((md.sort_values(by=['Metascore','Votes','Rating'], ascending = False)).head(10))


tmd.Rank.nunique()

tmd = tmd.drop_duplicates(subset='Rank', keep='first', inplace=False)

tmd.shape

plt.figure(figsize=(7,7))
sns.barplot(y="Title", x="Revenue (Millions)", data=tmd, ci= None)
plt.show()

#  Which are the highest Revenue generated Genres ?

# In[42]:


md['Genre'].value_counts().head(15)   #To view the most frequent genres and their counts in descending order


# In[43]:


# Created list to use for loop
lst = ['Action,Adventure,Sci-Fi',
        'Drama',
        'Comedy,Drama,Romance',
        'Comedy',
        'Drama,Romance',
        'Animation,Adventure,Comedy',
        'Comedy,Drama',
        'Action,Adventure,Fantasy',
        'Comedy,Romance',
        'Crime,Drama,Thriller',
        'Crime,Drama,Mystery',
        'Action,Adventure,Drama',
        'Action,Crime,Drama',
        'Horror,Thriller',
        'Drama,Thriller']


# Summation of revenue for each genre
TopGnr = []
for i in lst:
    x = int(md[md['Genre']==i]['Revenue (Millions)'].sum())
    TopGnr.append(x)

# TopGen DataFrame for the values.
TopGen= pd.DataFrame(TopGnr)
TopGen.columns =['Revenue_Sum(Millions)']
TopGen['Genres'] = lst

#TopGn.head()

TopGen=TopGen.sort_values('Revenue_Sum(Millions)', ascending=False) #Sorting Values in descending order for graph

plt.figure(figsize=(8,8))
sns.barplot(x="Revenue_Sum(Millions)", y="Genres",data=TopGen,linewidth=1,ci= None)



# ### What are the top 10 Genres (Combination) ?


plt.figure(figsize=(7,7))
md['Genre'].value_counts().head(10).plot.bar() #What are most interested Genres ?



# ### Is there a change in movie run time(avg) over the decade?

g = []
h = []
for i in range(2006,2017):
    den = md[md['Year']==i]['Runtime (Minutes)'].count()
    g.append(int((md[md['Year']==i]['Runtime (Minutes)'].sum())/den))
    h.append(i)

df9 = (pd.DataFrame(list(zip(g,h)), columns=['AvgRuntime','Year'])) #.sort_values('Revenue_Sum', ascending=False)
df9

plt.figure(figsize=(10,7))
sns.barplot(x="Year", y="AvgRuntime",data=df9,ci= None)


# ### Insights or Takeaways:

# - Drama is the most interested genre by majority of audience. And it is followed by Comedy and Romance. So, Producers can be rest assured while investing in these genres in future.
# - Fantasy, Sports genres are less explored by the existing movie makers. So this is a good space for the new directors, at the same time they would help avoid competition.
# - Most of the existing directors are concentrating on the genres: Action, Adventure and Sci-Fi even though they are not doing well commercially. Hence, Directors should start combinations of those with well performed genres such as, Drama, Comedy and Romance.
# - There are movies with higher metrics of rating, metascore and votes but, they have not done well in terms of revenue. So, These movies can be pushed to OTT platforms for further revenue generation.
# - Distributors can safely opt for the movies involving the Actors: Malin Akarmen, Mia Goth, Susan Loughnane, Michael Varten, Laura Dern, Roman Kolinka and Joan Allen, And the Directors: Alexandre Aja, Paul Anderson, Ridley Scott and Woody Allen.
