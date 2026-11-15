from collections import Counter
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import PowerTransformer

MIN_FILMS = 5 #an actor / director needs this many films in the training data to count as a "star"
KW_MIN = 25  #a plot keyword needs this many films to get its own column
FR_MIN = 3  # a franchise needs this many films to get its own column
SEED = 42


def frequent_labels(rows, min_films):
    counts = Counter(label for row in rows for label in set(row))
    return sorted(label for label, count in counts.items() if count >= min_films)


def multi_hot_frame(rows, labels, columns, index):
    return pd.DataFrame(
        {column: [int(label in row) for row in rows]
         for label, column in zip(labels, columns)},
        index=index,
    )


df = pd.read_csv("Clean.csv")

# genres: multi-hot, rare ones dropped (still counted in num_genres)
g = df["genres"].str.get_dummies()
g.columns = "genre_" + g.columns.str.replace("-", "_")
rare = g.columns[g.sum() < 30]
print("dropped rare genres:", list(rare))

df["num_genres"] = df["genres"].str.count(r"\|") + 1
df = pd.concat([df.drop(columns="genres"), g.drop(columns=rare)], axis=1)


df["is_english"] = (df["language"] == "English").astype(int)
df["is_usa"] = (df["country"] == "USA").astype(int)

# 0 likes on a film page mostly means no page existed, not that nobody liked it
df["has_movie_fb"] = (df["movie_facebook_likes"] > 0).astype(int)

# rare countries pooled
df["country_group"] = df["country"].where(df["country"].map(df["country"].value_counts()) >= 50, "Other")


# Age rating: ordinal "how adult" scale + pooled one-hot
rating_scale = {"G": 0, "TV-G": 0, "TV-Y": 0, "TV-Y7": 0,
                "Approved": 1, "Passed": 1, "GP": 1, "PG": 1, "TV-PG": 1,
                "PG-13": 2, "TV-14": 2,
                "R": 3, "TV-MA": 3, "M": 3,
                "X": 4, "NC-17": 4}
df["rating_level"] = df["content_rating"].map(rating_scale).fillna(-1).astype(int)   # -1 = unrated / unknown
df["has_rating"] = (df["rating_level"] >= 0).astype(int)
df["is_family"] = (df["rating_level"].between(0, 1)).astype(int)
df["is_restricted"] = (df["rating_level"] >= 3).astype(int)
df["content_rating"] = df["content_rating"].where(
    df["content_rating"].map(df["content_rating"].value_counts()) >= 30, "Other")

# Plot keywords: one column per keyword used in enough films
def clean_kw(s):
    return re.sub(r"[^0-9a-z]+", "_", s.strip().lower()).strip("_")

kw_lists = (df["plot_keywords"].fillna("").astype(str)
            .map(lambda s: [clean_kw(k) for k in s.split("|") if k.strip() and k != "Unknown"]))
df["num_keywords"] = kw_lists.map(len)

kw_classes = frequent_labels(kw_lists, KW_MIN)
kw = multi_hot_frame(kw_lists, kw_classes, ["kw_" + label for label in kw_classes], df.index)
print("keywords kept: {} (>= {} films)".format(kw.shape[1], KW_MIN))

# broader themes caught by substring, so rare keywords still contribute
kw_text = df["plot_keywords"].fillna("").str.lower()
themes = {
    "theme_superhero": r"superhero|comic book|marvel|dc comics",
    "theme_sequel_kw": r"sequel|part \d|prequel|reboot|remake",
    "theme_based_on_book": r"based on (novel|book|comic|true|play|young adult)",
    "theme_true_story": r"based on (a )?true|true story|biography",
    "theme_animation": r"animat|cartoon|pixar|talking animal",
    "theme_sci_fi": r"alien|space|future|robot|time travel|dystopia|spaceship",
    "theme_war": r"\bwar\b|soldier|world war|military|battle",
    "theme_crime": r"murder|serial killer|detective|heist|police|gangster|drug",
    "theme_horror": r"ghost|zombie|vampire|demon|slasher|haunted|monster",
    "theme_romance": r"love|wedding|romance|marriage|girlfriend|boyfriend",
    "theme_magic_fantasy": r"magic|wizard|witch|dragon|fantasy|fairy",
    "theme_kids_family": r"\bkid|child|family|school|teen|friendship",
    "theme_sports": r"football|basketball|baseball|boxing|racing|sport",
}
theme_df = pd.DataFrame({k: kw_text.str.contains(v).astype(int) for k, v in themes.items()},
                        index=df.index)

# Franchise detection from the title
title = df["movie_title"].fillna("").astype(str).str.strip().str.lower()

def base_title(t):
    t = re.split(r"\s*[:\-–]\s+", t)[0]                           # "x-men: apocalypse" -> "x-men"
    t = re.sub(r"\b(part|chapter|volume|vol)\.?\s*\w+$", "", t)    # "... part 2"
    t = re.sub(r"\s+(\d+|ii|iii|iv|v|vi|vii|viii|ix|x)$", "", t)   # "... 2", "... iii"
    return t.strip()

base = title.map(base_title)
base_counts = base.value_counts()
auto_franchise = base.where((base_counts.reindex(base).values >= 2) & (base.str.len() >= 3), "")

# curated list for franchises whose titles don't share a base name
known = {
    "star_wars": r"star wars", "star_trek": r"star trek", "harry_potter": r"harry potter",
    "tolkien": r"lord of the rings|the hobbit", "x_men": r"x-men|wolverine", "spider_man": r"spider-man",
    "batman": r"batman|dark knight", "superman": r"superman|man of steel",
    "marvel": r"avengers|iron man|thor\b|captain america|hulk|guardians of the galaxy|ant-man",
    "transformers": r"transformers", "fast_furious": r"fast (and|&) furious|furious 7|fast five|2 fast",
    "pirates": r"pirates of the caribbean", "shrek": r"shrek", "toy_story": r"toy story",
    "ice_age": r"ice age", "madagascar": r"madagascar", "despicable_me": r"despicable me|minions",
    "hunger_games": r"hunger games", "twilight": r"twilight", "matrix": r"the matrix",
    "mission_impossible": r"mission: impossible", "die_hard": r"die hard", "rocky": r"rocky|creed\b",
    "terminator": r"terminator", "alien_predator": r"\baliens?\b|predators?\b|prometheus",
    "jurassic": r"jurassic|lost world", "godzilla": r"godzilla", "apes": r"planet of the apes",
    "scary_movie": r"scary movie", "saw": r"^saw\b", "final_destination": r"final destination",
    "resident_evil": r"resident evil", "rush_hour": r"rush hour", "oceans": r"ocean's",
    "bourne": r"bourne", "indiana_jones": r"indiana jones", "back_to_future": r"back to the future",
    "lethal_weapon": r"lethal weapon", "american_pie": r"american pie|american reunion",
    "hangover": r"the hangover", "night_museum": r"night at the museum",
    "bond": r"\b007\b|casino royale|skyfall|quantum of solace|goldfinger|die another day|spectre",
    "kung_fu_panda": r"kung fu panda", "how_to_train": r"how to train your dragon",
    "cars_planes": r"^cars\b|^planes\b", "men_in_black": r"men in black",
    "mad_max": r"mad max", "the_conjuring": r"conjuring|annabelle|insidious",
    "paranormal": r"paranormal activity", "step_up": r"step up", "narnia": r"narnia",
    "pitch_perfect": r"pitch perfect", "meet_parents": r"meet the (parents|fockers)|little fockers",
}
fr_lists = []
for t, af in zip(title, auto_franchise):
    labels = [k for k, p in known.items() if re.search(p, t)]
    if af:
        labels.append("auto_" + re.sub(r"[^0-9a-z]+", "_", af).strip("_"))
    fr_lists.append(labels)

fr_classes = frequent_labels(fr_lists, FR_MIN)
fr = multi_hot_frame(fr_lists, fr_classes, ["fr_" + label for label in fr_classes], df.index)
print("franchise columns kept: {} (>= {} films)".format(fr.shape[1], FR_MIN))

df["franchise_key"] = [l[0] if l else "" for l in fr_lists]
df["is_franchise"] = (df["franchise_key"] != "").astype(int)
df["franchise_size"] = df["franchise_key"].map(df["franchise_key"].value_counts()).where(df["is_franchise"] == 1, 0).astype(int)
df["is_sequel"] = (title.str.contains(r"(\s\d+$|\s(ii|iii|iv|vi|vii|viii|ix)$|\bpart\b|\bchapter\b|returns|reloaded|revolutions|rises|begins)")
                   | (theme_df["theme_sequel_kw"] == 1)).astype(int)

# Title text features
df["title_len"] = title.str.len()
df["title_words"] = title.str.split().str.len()
df["title_has_colon"] = title.str.contains(":").astype(int)
df["title_has_number"] = title.str.contains(r"\d").astype(int)
df["title_starts_the"] = title.str.startswith("the ").astype(int)

# Era / age of the film
if "title_year" in df.columns:
    yr = df["title_year"].fillna(df["title_year"].median())
    df["decade"] = (yr // 10 * 10).astype(int)
    df["film_age"] = (yr.max() - yr).astype(int)
    df["is_post2000"] = (yr >= 2000).astype(int)
    df["is_post2010"] = (yr >= 2010).astype(int)

# Simple binaries from other columns, only if present
if "color" in df.columns:
    df["is_color"] = (df["color"].astype(str).str.strip().str.lower() == "color").astype(int)
if "budget" in df.columns:
    df["has_budget"] = (df["budget"] > 0).astype(int)
if "duration" in df.columns:
    df["is_long"] = (df["duration"] >= 150).astype(int)
    df["is_short"] = (df["duration"] <= 90).astype(int)
if "aspect_ratio" in df.columns:
    df["is_widescreen"] = (df["aspect_ratio"] >= 2.0).astype(int)
df["is_non_english_usa"] = ((df["is_english"] == 0) & (df["is_usa"] == 1)).astype(int)
df["is_english_foreign"] = ((df["is_english"] == 1) & (df["is_usa"] == 0)).astype(int)

# Skewed count / money columns: add log1p versions
skewed = ["budget", "duration", "movie_facebook_likes", "director_facebook_likes",
          "actor_1_facebook_likes", "actor_2_facebook_likes", "actor_3_facebook_likes",
          "cast_total_facebook_likes", "num_voted_users", "num_critic_for_reviews",
          "num_user_for_reviews", "facenumber_in_poster"]
for c in skewed:
    if c in df.columns:
        df["log_" + c] = np.log1p(df[c].clip(lower=0))

df = pd.concat([df, kw, fr, theme_df], axis=1)

# target rows only, imputed gross is not trusted
df = df[df["gross"].notna()].reset_index(drop=True)
print("rows with real gross:", len(df))


# split first, everything that looks at gross is fitted on train only
train, test = train_test_split(df, test_size=0.2, random_state=SEED)
train, test = train.reset_index(drop=True), test.reset_index(drop=True)

pt = PowerTransformer()
train["gross_yj"] = pt.fit_transform(train[["gross"]])[:, 0]
test["gross_yj"] = pt.transform(test[["gross"]])[:, 0]

actor_cols = ["actor_1_name", "actor_2_name", "actor_3_name"]


def fit_stats(frame, cols, target="gross_yj"):
    """mean target per person over every slot, only people with >= MIN_FILMS films."""
    long = pd.concat([frame[[c, target]].rename(columns={c: "name"}) for c in cols])
    long = long[long["name"] != "Unknown"]
    g = long.groupby("name")[target].agg(["mean", "size"])
    g = g[g["size"] >= MIN_FILMS]
    return g["mean"].to_dict(), frame[target].mean()


def person_features(frame, stats, prior, cols, prefix):
    out = pd.DataFrame(index=frame.index)
    for i, c in enumerate(cols, 1):
        out["{}{}_score".format(prefix, i)] = frame[c].map(stats).fillna(prior)   # non-stars get the global mean
        out["{}{}_is_star".format(prefix, i)] = frame[c].isin(stats).astype(int)
    if len(cols) > 1:
        score_cols = ["{}{}_score".format(prefix, i) for i in range(1, len(cols) + 1)]
        star_cols = ["{}{}_is_star".format(prefix, i) for i in range(1, len(cols) + 1)]
        out["cast_star_score"] = out[score_cols].mean(axis=1)
        out["cast_star_max"] = out[score_cols].max(axis=1)
        out["n_star_actors"] = out[star_cols].sum(axis=1)
    return out


def encode(train, test, cols, prefix):
    # train rows are scored out-of-fold so a film never sees its own gross
    oof = []
    for tr_idx, va_idx in KFold(shuffle=True, random_state=SEED).split(train):
        stats, prior = fit_stats(train.iloc[tr_idx], cols)
        oof.append(person_features(train.iloc[va_idx], stats, prior, cols, prefix))
    train_f = pd.concat(oof).sort_index()
    stats, prior = fit_stats(train, cols)
    test_f = person_features(test, stats, prior, cols, prefix)
    print("{}: {} stars in train (>= {} films)".format(prefix, len(stats), MIN_FILMS))
    return train_f, test_f


a_tr, a_te = encode(train, test, actor_cols, "actor")
d_tr, d_te = encode(train, test, ["director_name"], "director")

# how many films a name has in train (no gross involved)
freq_cols = actor_cols + ["director_name"]
all_names = pd.concat([train[c] for c in freq_cols])
for frame, a, d in ((train, a_tr, d_tr), (test, a_te, d_te)):
    for c, s in zip(["actor_1_films", "actor_2_films", "actor_3_films", "director_films"], freq_cols):
        frame[c] = frame[s].map(all_names.value_counts()).where(frame[s] != "Unknown", 0).fillna(0).astype(int)

train = pd.concat([train, a_tr, d_tr], axis=1)
test = pd.concat([test, a_te, d_te], axis=1)

# Actors / directors: multi-hot (names pooled over all 3 actor slots)
ACTOR_MIN = 8   # an actor needs this many films in train to get their own column
DIR_MIN = 6

def cast_lists(frame):
    return frame[actor_cols].apply(
        lambda r: sorted({n for n in r if pd.notna(n) and n != "Unknown"}), axis=1)

def director_lists(frame):
    return frame["director_name"].map(lambda n: [] if pd.isna(n) or n == "Unknown" else [n])

def multi_hot(train, test, lists_fn, prefix, min_films):
    tr_l, te_l = lists_fn(train), lists_fn(test)
    classes = frequent_labels(tr_l, min_films)
    cols = [prefix + re.sub(r"[^0-9a-z]+", "_", name.lower()).strip("_") for name in classes]
    tr_m = multi_hot_frame(tr_l, classes, cols, train.index)
    te_m = multi_hot_frame(te_l, classes, cols, test.index)
    print("{}: {} columns (>= {} films in train)".format(prefix, len(cols), min_films))
    return tr_m, te_m

act_tr, act_te = multi_hot(train, test, cast_lists, "act_", ACTOR_MIN)
dir_tr, dir_te = multi_hot(train, test, director_lists, "dir_", DIR_MIN)
train = pd.concat([train, act_tr, dir_tr], axis=1)
test = pd.concat([test, act_te, dir_te], axis=1)

# content rating one-hot, the rest of the text columns go
cat = ["content_rating", "country_group"]
train = pd.get_dummies(train, columns=cat, prefix=["rating", "country"], dtype=int)
test = pd.get_dummies(test, columns=cat, prefix=["rating", "country"], dtype=int)
test = test.reindex(columns=train.columns, fill_value=0)

drop = ["movie_title", "director_name", "actor_1_name", "actor_2_name", "actor_3_name",
        "plot_keywords", "gross_missing", "country", "language", "franchise_key"]

train = train.drop(columns=drop, errors="ignore")
test = test.drop(columns=drop, errors="ignore")

# any text column I didn't know about (color, imdb link, ...) can't go to a model, list and drop it
leftover = train.select_dtypes(exclude="number").columns.tolist()
if leftover:
    print("dropping leftover text columns:", leftover)
    train, test = train.drop(columns=leftover), test.drop(columns=leftover)

print("train:", train.shape, "| test:", test.shape)
train.to_csv("train_encoded.csv", index=False)
test.to_csv("test_encoded.csv", index=False)
