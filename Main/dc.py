import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import PowerTransformer

def outliers(col_name : str):
    q1 = df[col_name].quantile(0.25)
    q3 = df[col_name].quantile(0.75)
    iqr = q3 - q1
    up = q3+1.5*iqr
    lo = q1-1.5*iqr

    outliers = df[(df[col_name]>up) | (df[col_name]<lo)]
    return len(outliers)

#LOADING
df = pd.read_csv('movie_metadata.csv')

# print("raw shape:", df.shape)
# (5043,28)
# since there are no null  entries, now we drop link col as it is of no use
df = df.drop(columns=['movie_imdb_link'])

# dropping duplicates
df.drop_duplicates(subset=["movie_title", "title_year"], inplace=True)
# print(df.shape)
# (4919, 27)

# og dataset has 4919 rows

#CATEGORICAL COLUMNS

cat_cols = ['color', 'director_name', 'actor_2_name', 'genres', 'actor_1_name',
            'movie_title', 'actor_3_name', 'plot_keywords', 'language', 'country',
            'content_rating']

# dealing with cat cols first 

# print(df[cat_cols].isnull().sum())
# cols with null vals 

#color              19
#director_name     102
#actor_2_name       13
#genres              0
#actor_1_name        7
#movie_title         0
#actor_3_name       23
#plot_keywords     152
#language           14
#country             5
#content_rating    301
 
#drop colour column-logic:unoloured movies are extremely sparse and old-year gives that information.
df = df.drop(columns=["color"])

# imputing name cols with 'Unknown'
for c in ["director_name", "actor_1_name", "actor_2_name", "actor_3_name", "plot_keywords"]:
    df[c] = df[c].fillna("Unknown")

# 11/14 null val of lang from usa so imputing eng dropping rest 3

for x in df[df['language'].isna()]:
    if x in df[(df['language'].isna()) & (df['country']=='USA')]:
        df['language'] = df['language'].fillna('English')
    else:
        df.dropna(subset=['language'],inplace=True)
# (4919,26)

cat_cols = df.select_dtypes(exclude='number').columns
for col in cat_cols:
    print(col,': ',df[col].unique())

# old and tv labels folded into the modern ones, 
# missing rating as Not Rated
df["content_rating"] = df["content_rating"].replace({
    "Approved": "PG", "Passed": "PG", "GP": "PG", "M": "PG",
    "X": "NC-17", "TV-MA": "R", "TV-14": "PG-13", "TV-PG": "PG",
    "TV-G": "G", "TV-Y": "G", "TV-Y7": "G"})

df['content_rating'] = df['content_rating'].fillna('Not Rated') 

# plot_keywords need std rest r fine 
df['plot_keywords'] = df['plot_keywords'].str.strip().str.title()

# Std done 

# dealt with cats now will go for nums
# no rows dropped upto here 
# only 124 DUPLICATED ROWS DROPPED 
# (4919,26)




# 3. NUMERICAL COLUMNS

num_cols = df.select_dtypes(include='number').columns

# num cols = ['num_critic_for_reviews', 'duration', 'director_facebook_likes',
#        'actor_3_facebook_likes', 'actor_1_facebook_likes', 'gross',
#        'num_voted_users', 'cast_total_facebook_likes', 'facenumber_in_poster',
#        'num_user_for_reviews', 'budget', 'title_year',
#        'actor_2_facebook_likes', 'imdb_score', 'aspect_ratio',
#        'movie_facebook_likes'],

# dropping gross no data leakage this time 
df.dropna(subset=['gross'],inplace=True)
# (4056,26) #863 more rows dropped 

# budget in local currency (yen, won, rupee..) while gross is in usd
LOCAL = ["India", "Thailand", "Hungary", "Czech Republic", "Russia", "Turkey",
         "Indonesia", "Iran", "Pakistan", "Philippines"]
bad = ((df["budget"] > 4e8)
       | ((df["budget"] > 3e8) & (df["country"] != "USA"))
       | ((df["budget"] > 5e7) & df["country"].isin(LOCAL)))
# print("\nbudgets in local currency, set to missing:", bad.sum())
# budgets in local currency, set to missing: 18
df.loc[bad, "budget"] = np.nan

# print(len(df[df['budget'].isna()]))
# 284 rows with null budget

#apply KN inputation on it with GROSS not being one of the neigbours to avoid any kind of data leakage 

knn_cols = ["budget", "num_voted_users", "num_critic_for_reviews",
            "num_user_for_reviews", "movie_facebook_likes", "cast_total_facebook_likes",
            "imdb_score", "duration", "title_year"]

pt = PowerTransformer(method="yeo-johnson")  # handles zeros, standardises as it goes
Z = pt.fit_transform(df[knn_cols])
# comparing skewness of raw vs yj 
print("\nskew raw vs yeo-johnson:")
print(pd.DataFrame({"raw": df[knn_cols].skew(),
                    "yj": pd.DataFrame(Z, columns=knn_cols).skew()}).round(2))

Z = KNNImputer(n_neighbors=7, weights="uniform").fit_transform(Z)
imp = pd.DataFrame(pt.inverse_transform(Z), columns=knn_cols, index=df.index)
# clipping to avoid extreme outliers
df["budget"] = df["budget"].fillna(imp["budget"].clip(lower=df["budget"].min())).round()  # observed stay as is

# no person listed so no likes, so 0 rather than a median
for name, likes in [("director_name", "director_facebook_likes"),
                    ("actor_1_name", "actor_1_facebook_likes"),
                    ("actor_2_name", "actor_2_facebook_likes"),
                    ("actor_3_name", "actor_3_facebook_likes")]:
    df.loc[df[name].eq("Unknown") & df[likes].isna(), likes] = 0

# now imputing median safest opn + no big nan vals
# rounding upto 0 because followings cant be in fracn 

for c in ["director_facebook_likes", "actor_1_facebook_likes",
          "actor_2_facebook_likes", "actor_3_facebook_likes",
          'num_critic_for_reviews','num_user_for_reviews',
          'facenumber_in_poster']:
    df[c] = df[c].fillna(df[c].median()).round(0)

df['duration'] = df['duration'].fillna(df['duration'].median())

# now dealing with face one
# poster face counts go up to 43,Lets Winsorize at 10
df["facenumber_in_poster"] = df["facenumber_in_poster"].clip(upper=10)

# aspect ratio
# just 60 null vals of aspect raio r left to deal with-deaing with them imputing mode 

df["aspect_ratio"] = df["aspect_ratio"].replace(16.0, 1.78)  # 16 means 16:9
df["aspect_ratio"] = df["aspect_ratio"].fillna(df["aspect_ratio"].mode()[0])

# OUTLIERS now 
for col in num_cols:
    print('outliers in ',col,': ',outliers(col))
'''
outliers in  num_critic_for_reviews :  160
outliers in  duration :  142
outliers in  director_facebook_likes :  464
outliers in  actor_3_facebook_likes :  158
outliers in  actor_1_facebook_likes :  141
outliers in  gross :  292
outliers in  num_voted_users :  377
outliers in  cast_total_facebook_likes :  254
outliers in  facenumber_in_poster :  171
outliers in  num_user_for_reviews :  310
outliers in  budget :  277
outliers in  title_year :  143
outliers in  actor_2_facebook_likes :  588
outliers in  imdb_score :  95
outliers in  aspect_ratio :  0
outliers in  movie_facebook_likes :  428
'''

# 5. EXTRA FEATURES

# extra features
INFLATION = 0.018 # # same flat rate the tmdb notebook used--this is to standardize
df["decade"] = df["title_year"] // 10 * 10
df["budget_adj"] = (df["budget"] * (1 + INFLATION) ** (2016 - df["title_year"])).round()
df["budget_to_year_mean"] = df["budget"] / df.groupby("title_year")["budget"].transform("mean")
df["duration_to_year_mean"] = df["duration"] / df.groupby("title_year")["duration"].transform("mean")

df["title_len"] = df["movie_title"].str.len()
df["title_words"] = df["movie_title"].str.split().str.len()
df["num_keywords"] = (df["plot_keywords"].where(df["plot_keywords"] != "Unknown")
                      .str.count(r"\|").add(1).fillna(0).astype(int))

# 2 rows still null in title year removing them 
df.dropna(subset=['title_year'],inplace=True)

df.to_csv("Clean.csv", index=False)

# done with data cleaning

