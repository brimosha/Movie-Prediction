import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import PowerTransformer

def imputing_med(col_name : str):
    df[col_name] = df[col_name].fillna(df[col_name].median())
    return df[col_name]

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

print("raw shape:", df.shape)
#since there are no null  entries, now we drop
df = df.drop(columns=['movie_imdb_link'])

df.drop_duplicates(subset=["movie_title", "title_year"], inplace=True)
print(df.shape)


#CATEGORICAL COLUMNS


# cat cols = ['color', 'director_name', 'actor_2_name', 'genres', 'actor_1_name',
#        'movie_title', 'actor_3_name', 'plot_keywords', 'language', 'country',
#        'content_rating']

# dealing with cat cols first cuz i love cats---(me too ifykyk)

#df[cat_cols].isnull().sum()
# cols with null vals 
      
#df['color'] = df['color'].fillna(df['color'].mode()[0]) 
#drop colour column-logic:unoloured movies are extremely sparse and old-year gives that information.

df = df.drop(columns=["color"])


#df.dropna(subset=['director_name', 
#                  'actor_2_name', 
#                  'actor_1_name',
#                  'actor_3_name', 
#                  'plot_keywords'],
#                  inplace=True)----DO NOT DROP WILL ENCODE THIS LATER
#Instead impute them for now(ik sounds weird but lets try)
for c in ["director_name", "actor_1_name", "actor_2_name", "actor_3_name", "plot_keywords"]:
    df[c] = df[c].fillna("Unknown")


# 11/14 null val of lang from usa so imputing eng dropping rest 3

for x in df[df['language'].isna()]:
    if x in df[(df['language'].isna()) & (df['country']=='USA')]:
        df['language'] = df['language'].fillna('English')
    else:
        df.dropna(subset=['language'],inplace=True)

# country baad mein 

# old and tv labels folded into the modern ones, missing rating as unrated(instead of Not Rated-no issue with that just one word instead of two)
df["content_rating"] = df["content_rating"].replace({
    "Not Rated": "Unrated", "Approved": "PG", "Passed": "PG", "GP": "PG", "M": "PG",
    "X": "NC-17", "TV-MA": "R", "TV-14": "PG-13", "TV-PG": "PG",
    "TV-G": "G", "TV-Y": "G", "TV-Y7": "G"})

df['content_rating'] = df['content_rating'].fillna('Unrated') 

cat_cols = df.select_dtypes(exclude='number').columns
for col in cat_cols:
    print(col,': ',df[col].unique())

# plot_keywords need std rest r fine 
df['plot_keywords'] = df['plot_keywords'].str.strip().str.title()

# Std done 

#dealt with cats now will go for nums
#305 rows dropped upto here

# ==============================================================================
# 3. NUMERICAL COLUMNS
# ==============================================================================

num_cols = df.select_dtypes(include='number').columns

# num cols = ['num_critic_for_reviews', 'duration', 'director_facebook_likes',
#        'actor_3_facebook_likes', 'actor_1_facebook_likes', 'gross',
#        'num_voted_users', 'cast_total_facebook_likes', 'facenumber_in_poster',
#        'num_user_for_reviews', 'budget', 'title_year',
#        'actor_2_facebook_likes', 'imdb_score', 'aspect_ratio',
#        'movie_facebook_likes'],

# dropping null gross and budget no impurities this time
#don't drop null budget thats feature impurity not target retain budget 
#apply KN inputation on it with GROSS not being one of the neigbours
#BUT BEFORE THAT LETS FIX THE SKEW

# --- 3a. target (gross) + budget (local currency fix, KNN imputation)
df.dropna(subset=['gross'],inplace=True)

# budget in local currency (yen, won, rupee..) while gross is in usd
LOCAL = ["India", "Thailand", "Hungary", "Czech Republic", "Russia", "Turkey",
         "Indonesia", "Iran", "Pakistan", "Philippines"]
bad = ((df["budget"] > 4e8)
       | ((df["budget"] > 3e8) & (df["country"] != "USA"))
       | ((df["budget"] > 5e7) & df["country"].isin(LOCAL)))
print("\nbudgets in local currency, set to missing:", bad.sum())
print(df.loc[bad, ["movie_title", "country", "budget"]].head(8).to_string(index=False))
df.loc[bad, "budget"] = np.nan

knn_cols = ["budget", "num_voted_users", "num_critic_for_reviews",
            "num_user_for_reviews", "movie_facebook_likes", "cast_total_facebook_likes",
            "imdb_score", "duration", "title_year"]

pt = PowerTransformer(method="yeo-johnson")  # handles zeros, standardises as it goes
Z = pt.fit_transform(df[knn_cols])
print("\nskew raw vs yeo-johnson:")
print(pd.DataFrame({"raw": df[knn_cols].skew(),
                    "yj": pd.DataFrame(Z, columns=knn_cols).skew()}).round(2))

Z = KNNImputer(n_neighbors=7, weights="uniform").fit_transform(Z)
imp = pd.DataFrame(pt.inverse_transform(Z), columns=knn_cols, index=df.index)
df["budget"] = df["budget"].fillna(imp["budget"].clip(lower=df["budget"].min())).round()  # observed stay as is



# now imputing median safest opn + no big nan vals
df['num_critic_for_reviews'] = round(imputing_med('num_critic_for_reviews'),0)
df['duration'] = imputing_med('duration')
df['num_user_for_reviews'] = round(imputing_med('num_user_for_reviews'),2)

# no person listed so no likes, so 0 rather than a median
for name, likes in [("director_name", "director_facebook_likes"),
                    ("actor_1_name", "actor_1_facebook_likes"),
                    ("actor_2_name", "actor_2_facebook_likes"),
                    ("actor_3_name", "actor_3_facebook_likes")]:
    df.loc[df[name].eq("Unknown") & df[likes].isna(), likes] = 0


for c in ["facenumber_in_poster",
          "director_facebook_likes", "actor_1_facebook_likes",
          "actor_2_facebook_likes", "actor_3_facebook_likes"]:
    df[c] = df[c].fillna(df[c].median())


# now dealing with face one, cant fill with med, imputing mode just 11 null vals

df['facenumber_in_poster'] = df['facenumber_in_poster'].fillna(df['facenumber_in_poster'].mode()[0])

# poster face counts go up to 43,Lets Winsorize at 10
df["facenumber_in_poster"] = df["facenumber_in_poster"].clip(upper=10)

# aspect ratio
# dk how to deal with aspect ratio btw country got treated automatically 
# just 60 null vals of aspect raio r left to deal with-deaing with them

df["aspect_ratio"] = df["aspect_ratio"].replace(16.0, 1.78)  # 16 means 16:9
df["aspect_ratio"] = df["aspect_ratio"].fillna(df["aspect_ratio"].mode()[0])

# net rows removed 1147/4998 cuz 4998 og rows hence 23% data removed n 1 col removed of link, was unnec 

# DUP + NULL VALS HANDLED




# OUTLIERS now 

bud_out = outliers('budget') #276 out in bud
for col in num_cols:
    print('outliers in ',col,': ',outliers(col))


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


df.to_csv("Clean.csv", index=False)

''' 
outliers in  num_critic_for_reviews :  149
outliers in  duration :  128
outliers in  director_facebook_likes :  407
outliers in  actor_3_facebook_likes :  167
outliers in  actor_1_facebook_likes :  143
outliers in  gross :  281
outliers in  num_voted_users :  321
outliers in  cast_total_facebook_likes :  230
outliers in  facenumber_in_poster :  157
outliers in  num_user_for_reviews :  293
outliers in  budget :  276
outliers in  title_year :  141
outliers in  actor_2_facebook_likes :  600
outliers in  imdb_score :  90
outliers in  aspect_ratio :  1
outliers in  movie_facebook_likes :  402
'''
# done with data cleaning