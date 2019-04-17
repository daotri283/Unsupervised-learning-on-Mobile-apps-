# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 18:19:09 2019

@author: daotr
"""



# Importing new libraries
from sklearn.preprocessing import StandardScaler # standard scaler
from sklearn.decomposition import PCA # principal component analysis


# Importing known libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Setting pandas print options
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)



# Importing dataset
customers_df = pd.read_excel('finalExam_Mobile_App_Survey_Data.xlsx')


###############################################################################
# PCA One Last Time!!!
###############################################################################

########################
# Step 1: Remove demographic information
########################
list=['caseID','q1','q48','q49','q50r1','q50r2',	'q50r3',	'q50r4',	'q50r5','q54','q55',
      'q56','q57']
demo_var=customers_df[list]

customer_features_reduced = customers_df.drop(list,axis=1)

customer_features_reduced.columns=['Iphone',
                                   'Ipod Touch',
                                   'Android',
                                   'BlackBerry',
                                   'Nokia',
                                   'Window Phone',
                                   'HP',
                                   'Tablet',
                                   'Other Smart Phone',
                                   'No Smartphone',
                                   'Music apps',
                                   'TV Check-in apps',
                                   'Entertainment apps',
                                   'TV Show apps',
                                   'Gaming apps',
                                   'Social Network apps',
                                   'General News apps',
                                   'Shopping apps',
                                   'Specific News apps',
                                   'Other apps',
                                   'No apps',
                                   'Number of apps',
                                   'Percent of free apps',
                                   'Facebook',
                                   'Twitter',
                                   'Myspace',
                                   'Pandora radio',
                                   'Vevo',
                                   'YouTube',
                                   'AOL Radio',
                                   'Last.fm',
                                   'Yahoo',
                                   'IMDB',
                                   'LinkedIn',
                                   'Netflix',
                                   'q24r1',
                                   'q24r2',
                                   'q24r3',
                                   'q24r4',
                                   'q24r5',
                                   'q24r6',
                                   'q24r7',
                                   'q24r8',
                                   'q24r9',
                                   'q24r10',
                                   'q24r11',
                                   'q24r12',
                                   'q25r1',
                                   'q25r2',
                                   'q25r3',
                                   'q25r4',
                                   'q25r5',
                                   'q25r6',
                                   'q25r7',
                                   'q25r8',
                                   'q25r9',
                                   'q25r10',
                                   'q25r11',
                                   'q25r12',
                                   'q26r18',
                                   'q26r3',
                                   'q26r4',
                                   'q26r5',
                                   'q26r6',
                                   'q26r7',
                                   'q26r8',
                                   'q26r9',
                                   'q26r10',
                                   'q26r11',
                                   'q26r12',
                                   'q26r13',
                                   'q26r14',
                                   'q26r15',
                                   'q26r16',
                                   'q26r17']


########################
# Step 2: Scale to get equal variance
########################

scaler = StandardScaler()


scaler.fit(customer_features_reduced)


X_scaled_reduced = scaler.transform(customer_features_reduced)


########################
# Step 3: Run PCA without limiting the number of components
########################

customer_pca_reduced = PCA(n_components = None,
                           random_state = 508)


customer_pca_reduced.fit(X_scaled_reduced)


X_pca_reduced = customer_pca_reduced.transform(X_scaled_reduced)



########################
# Step 4: Analyze the scree plot to determine how many components to retain
########################

fig, ax = plt.subplots(figsize=(10, 8))

features = range(customer_pca_reduced.n_components_)


plt.plot(features,
         customer_pca_reduced.explained_variance_ratio_,
         linewidth = 2,
         marker = 'o',
         markersize = 10,
         markeredgecolor = 'black',
         markerfacecolor = 'grey')


plt.title('Reduced Wholesale Customer Scree Plot')
plt.xlabel('PCA feature')
plt.ylabel('Explained Variance')
plt.xticks(features)
plt.show()



########################
# Step 5: Run PCA again based on the desired number of components
########################

print(f"""
Right now we have the following:
    1 Principal Component : {customer_pca_reduced.explained_variance_ratio_[0].round(2)}
    5 Principal Components: {
    (customer_pca_reduced.explained_variance_ratio_[0] + 
    customer_pca_reduced.explained_variance_ratio_[1] +
    customer_pca_reduced.explained_variance_ratio_[2] + 
    customer_pca_reduced.explained_variance_ratio_[3]+
    customer_pca_reduced.explained_variance_ratio_[4]+
    customer_pca_reduced.explained_variance_ratio_[5]).round(2)
    }
""")

customer_pca_reduced = PCA(n_components = 5,
                           random_state = 508)


customer_pca_reduced.fit(X_scaled_reduced)



########################
# Step 6: Analyze factor loadings to understand principal components
########################

factor_loadings_df = pd.DataFrame(pd.np.transpose(customer_pca_reduced.components_))


factor_loadings_df = factor_loadings_df.set_index(customer_features_reduced.columns)


print(factor_loadings_df)

factor_loadings_df.to_excel('Final_factor_loadings.xlsx')


###############################################################################
# Combining PCA and Clustering!!!
###############################################################################
from sklearn.cluster import KMeans # k-means clustering

########################
# Step 1: Take your transformed dataframe
########################

clustering_pca_data = customer_pca_reduced.transform(X_scaled_reduced)

clustering_pca_dataframe = pd.DataFrame(clustering_pca_data)


########################
# Step 2: Scale to get equal variance
########################

scaler = StandardScaler()


scaler.fit(clustering_pca_dataframe)


X_pca_clust = scaler.transform(clustering_pca_dataframe)


X_pca_clust_df = pd.DataFrame(X_pca_clust)


print(pd.np.var(X_pca_clust_df))





########################
# Step 3: Experiment with different numbers of clusters
########################

customers_k_pca = KMeans(n_clusters = 5,
                         random_state = 508)


customers_k_pca.fit(X_pca_clust_df)


customers_kmeans_pca = pd.DataFrame({'cluster': customers_k_pca.labels_})


print(customers_kmeans_pca.iloc[: , 0].value_counts())


########################
# Step 4: Analyze cluster centers
########################

centroids_pca = customers_k_pca.cluster_centers_


centroids_pca_df = pd.DataFrame(centroids_pca)


# Rename your principal components
X_pca_clust_df.columns = ['Gamers', 'Jolly', 'Movie Lovers', 'Iphone Users', 'Non Iphone Users']

centroids_pca_df.columns = X_pca_clust_df.columns

print(centroids_pca_df)

#centroids_pca_df= centroids_pca_df.T
# Sending data to Excel

centroids_pca_df.to_excel('Final_clustering_PLUS_pca_centriods.xlsx')



###############################################################################
# BOX PLOTS
###############################################################################

final_pca_df = pd.concat([customer_features_reduced, customers_df[list]], axis = 1)


# Renaming age
age = {1 :'Under 18',
       2 :'18-24',
       3 :'25-29',
       4 :'30-34',
       5 :'35-39',
       6 :'40-44',
       7 :'45-49',
       8 :'50-54',
       9 :'55-59',
       10:'60-64',
       11:'65 and over'
       }
final_pca_df['q1'].replace(age, inplace = True)

#renaming education level 
education_level= {1:'Some high school',
                  2:'High School grad',
                  3:'Some College',
                  4:'College grad',
                  5:'Some post grad',
                  6:'Post grad degree'
                  }
final_pca_df['q48'].replace(education_level, inplace = True)


#renaming marital status 
marital_status= {1:'Married',
                 2:'Single',
                 3:'Single with a partner',
                 4:'Divorced'
                 }

final_pca_df['q49'].replace(marital_status, inplace = True)

#renaming race
race={1:'White',
      2:'Black',
      3:'Asian',
      4:'Native Hawaiian',
      5:'American Indian',
      6:'Other race'
      }
final_pca_df['q54'].replace(race, inplace = True)

#renaming Hispanic or latino ethnicity 
latino={1:'Latino',
        2:'Not Latino'
        }
final_pca_df['q55'].replace(latino, inplace = True)

#renaming income group
income_group={1:'Under 10K',
              2:'10K-15K',
              3:'15K-20K',
              4:'20K-30K',
              5:'30K-40K',
              6:'40K-50K',
              7:'50K-60K',
              8:'60K-70K',
              9:'70K-80K',
              10:'80K-90K',
              11:'90K-100K',
              12:'100K-125K',
              13:'125K-150K',
              14:'Above 150K',
                }
final_pca_df['q56'].replace(income_group, inplace = True)

#renamne gender
gender={1:'Male',
        2:'Female'}
final_pca_df['q57'].replace(gender, inplace = True)

#Analyzing by age 
#age and number of apps 
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q1',
            y =  'Number of apps',
            data = final_pca_df)

plt.ylim(0, 8)
plt.tight_layout()
plt.show()

#age and percentage of free apps 
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q1',
            y =  'Percent of free apps',
            data = final_pca_df)

plt.ylim(0, 8)
plt.tight_layout()
plt.show()

#Analyzing by educantion level
# education level and number of apps 
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q48',
            y =  'Number of apps',
            data = final_pca_df)

plt.ylim(0, 8)
plt.tight_layout()
plt.show()

#educantion level and percentage of free apps 
fig, ax = plt.subplots(figsize = (12, 8))
sns.boxplot(x = 'q48',
            y =  'Percent of free apps',
            data = final_pca_df)

plt.ylim(0, 8)
plt.tight_layout()
plt.show()

#Analyzing Maital Status 
# Maital Status  and number of apps 
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q49',
            y =  'Number of apps',
            data = final_pca_df)

plt.ylim(0, 8)
plt.tight_layout()
plt.show()

# Maital Status and percentage of free apps 
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q49',
            y =  'Percent of free apps',
            data = final_pca_df)

plt.ylim(0, 8)
plt.tight_layout()
plt.show()

#Analyzing Race
# Race  and number of apps 
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q54',
            y =  'Number of apps',
            data = final_pca_df)

plt.ylim(0, 8)
plt.tight_layout()
plt.show()

#Race and percentage of free apps 
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q54',
            y =  'Percent of free apps',
            data = final_pca_df)

plt.ylim(0, 8)
plt.tight_layout()
plt.show()

#Analyzing Latino ethnicity 
# Latino ethnicity   and number of apps 
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q55',
            y =  'Number of apps',
            data = final_pca_df)

plt.ylim(0, 8)
plt.tight_layout()
plt.show()

#Latino ethnicity  and percentage of free apps 
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q55',
            y =  'Percent of free apps',
            data = final_pca_df)

plt.ylim(0, 8)
plt.tight_layout()
plt.show()

#Analyzing income group 
# income group and number of apps 
fig, ax = plt.subplots(figsize = (12, 8))
sns.boxplot(x = 'q56',
            y =  'Number of apps',
            data = final_pca_df)

plt.ylim(0, 8)
plt.tight_layout()
plt.show()

#income group and percentage of free apps 
fig, ax = plt.subplots(figsize = (12, 8))
sns.boxplot(x = 'q56',
            y =  'Percent of free apps',
            data = final_pca_df)

plt.ylim(0, 8)
plt.tight_layout()
plt.show()

#Analyzing gender
# gender   and number of apps 
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q57',
            y =  'Number of apps',
            data = final_pca_df)

plt.ylim(0, 8)
plt.tight_layout()
plt.show()

#gender and percentage of free apps 
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'q57',
            y =  'Percent of free apps',
            data = final_pca_df)

plt.ylim(0, 8)
plt.tight_layout()
plt.show()


""" 
Some mobile apps stats 
Gaming apps 
Of all the mobile apps in the Apple App Store, 
the most popular category for developers is gaming apps.
Nearly 25 percent of all mobile apps fall into the games category
That’s 2.5 times more apps than the second-highest category (business) and
4 times more than entertainment, the fifth highest




The younger demographics are spending more hours using mobile apps than the older demos.
The average 18- to 24-year-old spends more than three hours a day using the mobile apps 
on their phone while people 55 and older spend less than two hours.




Millennials are more willing to pay for mobile apps,
with 1/3 purchasing more than one per month

In 2017, only 36 percent of millennials made zero app purchases the year before, 
while 80 percent of smartphone users age 55 and older did the same. 
During the same period, 36 percent of millennials purchased five or more apps,
 while just 3 percent of age 55+ users did the same.
 
 
 
In 2017, 46 percent of millennials made 5+ in-app purchases & 
70 percent made more than 1
 
In 2017, 21 percent of millennials said they deleted a mobile app 
because they didn’t like the app logo
 
 
90% of smartphone owners use their phones 
while making purchasing decisions in a brick and mortar store. 

Gradually, year-over-year in-app purchases began to take over other monetization models 
and by 2017 way over 50% a mobile app revenue is generated via in-app purchases.
 
In a pursue to increase an app revenue, app developers turn to placing ads inside their apps. 
There are four major types of mobile ads – Display, Search, Social and Video. 
"""
