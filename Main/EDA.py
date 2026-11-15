from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

import funcnsvis as vis

df = pd.read_csv("Clean.csv")
out = Path(__file__).with_name("eda_plots")

genres = df["genres"].fillna("").str.get_dummies(sep="|")
genre_n = genres.sum().sort_values(ascending=False)

keywords = Counter(w.strip().lower() for s in df["plot_keywords"].fillna("") for w in s.split("|") if w.strip() and w != "Unknown")

country_n = df["country"].value_counts()

rating_n = df["content_rating"].value_counts()
actor_n = df[["actor_1_name", "actor_2_name", "actor_3_name"]].stack().value_counts().drop(index="Unknown", errors="ignore")
director_n = df.loc[df["director_name"] != "Unknown", "director_name"].value_counts()

save = vis.save_g

save(
    vis.hist(
        np.log10(df["gross"]), 40, "Gross distribution (log10)",
        "log10 gross ($)", "Films", "#4f81bd", "white", show=False,
    ),
    out / "gross_log_histogram.png",
)
save(
    vis.hist(
        np.log10(df["budget"]), 40, "Budget distribution (log10)",
        "log10 budget ($)", "Films", "#9bbb59", "white", show=False,
    ),
    out / "budget_log_histogram.png",
)
save(
    vis.ecdf_g(
        df["gross"] / 1e6, "Cumulative film share by gross",
        "Gross ($M, log scale)", x_scale="log", color="#4f81bd", show=False,
    ),
    out / "gross_ecdf.png",
)
save(
    vis.scatter_g(
        df["budget"], df["gross"], "Budget vs gross",
        "Budget ($)", "Gross ($)", "#4f81bd", 12,
        x_scale="log", y_scale="log", alpha=.3, show=False,
    ),
    out / "budget_vs_gross.png",
)
save(
    vis.hist(
        df["duration"].dropna(), 40, "Movie duration",
        "Minutes", "Films", "#8064a2", "white", show=False,
    ),
    out / "duration_histogram.png",
)
save(
    vis.violin_g(
        df, "content_rating", "imdb_score", "IMDb score by content rating",
        order=rating_n.head(6).index, inner="quart", show=False,
    ),
    out / "score_by_rating.png",
)
save(
    vis.barh_g(
        rating_n.head(12).sort_values().values,
        rating_n.head(12).sort_values().index,
        "Films by content rating", "Films", "Rating", "#c0504d", show=False,
    ),
    out / "rating_counts.png",
)
save(
    vis.barh_g(
        country_n.head(15).sort_values().values,
        country_n.head(15).sort_values().index,
        "Films by country (top 15)", "Films", "Country", "#4f81bd",
        reference=50, reference_label="50-film cutoff", show=False,
    ),
    out / "country_counts.png",
)
save(
    vis.barh_g(
        genre_n.head(15).sort_values().values,
        genre_n.head(15).sort_values().index,
        "Films by genre (top 15)", "Films", "Genre", "#9bbb59", show=False,
    ),
    out / "genre_counts.png",
)
save(
    vis.wordcloud_g(
        keywords, "Plot keywords (larger = more films)", "viridis", show=False,
    ),
    out / "keyword_wordcloud.png",
)
keyword_counts = pd.Series(keywords)
retained = int((keyword_counts >= 25).sum())
near_cutoff = keyword_counts[keyword_counts.between(10, 40)]
at_cutoff = int((keyword_counts == 25).sum())
save(
    vis.hist(
        near_cutoff, np.arange(9.5, 40.5, 1),
        f"Keyword counts near cutoff ({retained} pass; {at_cutoff} at 25)",
        "Films containing keyword (10-40)", "Keywords",
        "#8064a2", "white", reference=24.5,
        reference_label="25-film cutoff", show=False,
    ),
    out / "keyword_cutoff.png",
)
top = genre_n.head(12).index
save(
    vis.heatmap_g(
        genres[top].T.dot(genres[top]), "Common genre co-occurrence",
        mask=np.eye(len(top), dtype=bool), cmap="YlGnBu", show=False,
    ),
    out / "genre_cooccurrence.png",
)
save(
    vis.hist_group(
        [np.log10(actor_n.values), np.log10(director_n.values)],
        ["Actors", "Directors"], "Person appearances", "log10 films per person",
        colors=["#4f81bd", "#c0504d"], show=False,
    ),
    out / "person_frequency.png",
)
save(
    vis.bar_g(
        df["decade"].value_counts().sort_index().index,
        df["decade"].value_counts().sort_index().values,
        "Films per decade", "Decade", "Films", "#8064a2", show=False,
    ),
    out / "films_per_decade.png",
)
era = df.groupby("decade").agg(gross=("gross", "median"), budget=("budget", "median"))
save(
    vis.dumbbell_g(
        era.index.astype(str), era["gross"] / 1e6, era["budget"] / 1e6,
        "Gross", "Budget", "Median gross and budget by decade",
        "$M (log scale)", x_scale="log", show=False,
    ),
    out / "gross_budget_by_decade.png",
)
genre_rating = df.assign(genre=df["genres"].str.split("|")).explode("genre").groupby(["genre", "content_rating"]).size().unstack(fill_value=0)
genre_rating = genre_rating.loc[genre_n.head(8).index].reindex(columns=["G", "PG", "PG-13", "R", "NC-17", "Unrated"], fill_value=0)
rating_colors = ["#59A14F", "#76B7B2", "#4E79A7", "#F28E2B", "#B07AA1", "#9D9D9D"]
save(
    vis.sunburst_g(
        genre_rating, "Genre and content-rating mix (top 8 genres)",
        rating_colors, show=False,
    ),
    out / "genre_rating_sunburst.png",
)
numeric = df[["gross", "budget", "num_voted_users", "num_critic_for_reviews", "num_user_for_reviews", "movie_facebook_likes", "duration", "imdb_score"]]
corr = numeric.corr(method="spearman").fillna(0)
save(
    vis.dendrogram_g(
        corr, "Clusters of raw numeric features (Spearman)", show=False,
    ),
    out / "feature_dendrogram.png",
)
