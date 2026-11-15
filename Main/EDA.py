import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PowerTransformer
from sklearn.feature_selection import mutual_info_regression
from wordcloud import WordCloud

df = pd.read_csv("Clean.csv")
os.makedirs("eda_plots", exist_ok=True)
sns.set_theme(style="whitegrid")
plt.rcParams["axes.titlesize"] = 10
issues, n_fig = [], 0


# helpers: every panel below is one call
def save(name):
    global n_fig
    n_fig += 1
    plt.tight_layout()
    plt.savefig(f"eda_plots/{n_fig:02d}_{name}.png", dpi=130)
    plt.show()


def grid(r, c, w=6.3, h=5.5):                      # r x c figure, axes as one flat array
    return np.atleast_1d(plt.subplots(r, c, figsize=(c * w, r * h))[1]).ravel()


def hide(a, n):
    for ax in a[n:]:
        ax.axis("off")


def hist(ax, v, t, bins=40, **kw):
    sns.histplot(v, bins=bins, ax=ax, **kw)
    ax.set_title(t)


def box(ax, x, y, t, data=None, log=True, **kw):
    sns.boxplot(data=df if data is None else data, x=x, y=y, ax=ax, **kw)
    if log:
        ax.set_yscale("log")
    ax.set_title(t)


def bar(ax, s, t, h=False, **kw):
    if len(s):
        (s.plot.barh if h else s.plot.bar)(ax=ax, **kw)
    ax.set_title(t)


def dec(ax, c, y, t, xlog=False):                  # median of y inside each decile of c
    b = pd.qcut(df[c].rank(method="first"), 10, labels=False)
    ax.plot(df.groupby(b)[c].median(), df.groupby(b)[y].median(), "o-")
    ax.set(yscale="log", xscale="log" if xlog else "linear", xlabel=c, ylabel=f"median {y}", title=t)


def by_origin(ax, x, y):
    for u, col in [("USA", "#4f81bd"), ("non-USA", "#c0504d")]:
        m = df["origin"] == u
        ax.scatter(df.loc[m, x], df.loc[m, y], s=7, alpha=.3, c=col, label=u)
    ax.legend(fontsize=8)


def cloud(freq, title, cmap, name, **kw):
    wc = WordCloud(width=1200, height=700, background_color="white", colormap=cmap, prefer_horizontal=.9,
                   random_state=1, **kw).generate_from_frequencies(dict(freq))
    plt.figure(figsize=(12, 7))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(title)
    save(name)


def spear(a, b):
    return df[[a, b]].corr(method="spearman").iloc[0, 1]


# numeric columns of the file as it comes, derived stuff is added after
num = [c for c in df.select_dtypes("number").columns if c != "decade"]
feat = [c for c in num if c != "gross"]
likes = ["director_facebook_likes", "actor_1_facebook_likes", "actor_2_facebook_likes",
         "actor_3_facebook_likes", "cast_total_facebook_likes", "movie_facebook_likes"]
post = ["num_voted_users", "num_critic_for_reviews", "num_user_for_reviews", "imdb_score", "movie_facebook_likes"]
order = ["G", "PG", "PG-13", "R", "NC-17", "Unrated"]

df["movie_title"] = df["movie_title"].str.replace("\xa0", " ").str.strip()
df["gM"], df["bM"] = df["gross"] / 1e6, df["budget"] / 1e6
df["f_factor"] = df["gross"] / df["budget"]
df["num_genres"] = df["genres"].str.count(r"\|") + 1
G = df["genres"].str.get_dummies("|")
df["is_english"] = (df["language"] == "English").astype(int)
df["is_usa"] = (df["country"] == "USA").astype(int)
df["origin"] = np.where(df["is_usa"] == 1, "USA", "non-USA")
df["era"] = pd.cut(df["title_year"], [0, 1979, 1989, 1999, 2009, 2100], labels=["<1980", "1980s", "1990s", "2000s", "2010+"])
df["country_group"] = df["country"].where(df["country"].map(df["country"].value_counts()) >= 40, "Other")
df["is_sequel"] = df["movie_title"].str.contains(r"\b[2-9]\b|\bII+\b|\bIV\b|:|\bPart\b")
yj_gross = pd.Series(PowerTransformer().fit_transform(df[["gross"]])[:, 0], index=df.index)
yr = df.dropna(subset=["title_year"])
by_era = df.groupby("era", observed=True)


# 1. overview
print(df.shape, "\n\ndtypes:\n", df.dtypes.value_counts().to_string())
print("\nnulls:\n", df.isnull().sum()[lambda s: s > 0].to_string())
print("\ncolumn summary:\n", df[num].describe().T.round(2).to_string())
a = grid(1, 3, 6.5, 8)
bar(a[0], df.drop(columns=["f_factor", "gM", "bM"]).nunique().sort_values(), "unique values per column (log)", h=True, logx=True)
z = (df[num] == 0).mean().sort_values()
bar(a[1], z[z > 0], "share of exact zeros", h=True, color="#c0504d")
nul = df.isnull().sum()
bar(a[2], nul[nul > 0], "null counts", h=True, color="#8064a2")
save("overview")

odd = df["budget"] % 1000 != 0
unk = {c: int((df[c] == "Unknown").sum()) for c in ["director_name", "actor_1_name", "actor_2_name", "actor_3_name", "plot_keywords"]}
issues += [f"{df.duplicated(['movie_title', 'title_year', 'gross']).sum()} films appear twice (same title/year/gross): drop before splitting",
           f"{df['title_year'].isnull().sum()} rows with no title_year: {df.loc[df['title_year'].isnull(), 'movie_title'].unique().tolist()}",
           f"{odd.sum()} budgets ({odd.mean():.1%}) are not round numbers: probably imputed, no flag column",
           f"'Unknown' placeholders {unk}: would become one fake person / keyword",
           "post-release columns, only usable if you predict after release: " + ", ".join(post)]
for c in ["language", "country", "content_rating", "aspect_ratio", "num_keywords", "facenumber_in_poster"]:
    top = df[c].value_counts(normalize=True)
    if top.iloc[0] > .9:
        issues.append(f"{c}: '{top.index[0]}' is {top.iloc[0]:.1%} of rows (near constant)")


# 2. target: gross
g = df["gross"]
print(f"\ngross: mean ${g.mean() / 1e6:.1f}M, median ${g.median() / 1e6:.1f}M, max ${g.max() / 1e6:.0f}M, skew {g.skew():.2f}, kurtosis {g.kurt():.1f}")
print((g.quantile([.01, .05, .25, .5, .75, .95, .99]) / 1e6).round(2).to_string())
print(f"top 10% of films earn {g.nlargest(len(g) // 10).sum() / g.sum():.0%} of all gross, top 1% earn {g.nlargest(len(g) // 100).sum() / g.sum():.0%}")
a = grid(2, 3, 5.7, 4.5)
hist(a[0], df["gM"], f"gross ($M), skew {g.skew():.1f}", bins=60, color="#c0504d")
hist(a[1], np.log10(g), f"log10 gross, skew {np.log10(g).skew():.2f}", bins=60, color="#4f81bd")
hist(a[2], yj_gross, f"yeo-johnson gross, skew {yj_gross.skew():.2f}", bins=60, color="#9bbb59")
sns.boxplot(x=np.log10(g), ax=a[3])
a[3].set_title("log10 gross boxplot")
xs = np.sort(g.values)
p = np.arange(1, len(xs) + 1) / len(xs)
a[4].plot(xs / 1e6, p)
a[4].set(xscale="log", title="gross ECDF ($M)")
a[5].plot(p, np.cumsum(xs) / xs.sum())
a[5].plot([0, 1], [0, 1], "r--")
a[5].set_title("share of total gross held by the lowest x% of films")
save("gross_distribution")

a = grid(1, 2, 8, 7)
t = df.nlargest(15, "gross")
a[0].barh(t["movie_title"][::-1], t["gM"][::-1], color="#4f81bd")
a[0].set_title("top 15 gross ($M)")
t = df.nsmallest(15, "gross")
a[1].barh(t["movie_title"][::-1], t["gross"][::-1], color="#c0504d")
a[1].set(xscale="log", title="15 lowest gross ($)")
save("gross_top_bottom")
issues.append(f"{(g < 1e4).sum()} films have gross under $10k (mostly foreign titles, looks like US-only box office)")


# 3. every numeric column
skew = df[num].skew()
a = grid(5, 5, 3.8, 2.8)
for ax, c in zip(a, num):
    hist(ax, df[c].dropna(), f"{c} (skew {skew[c]:.1f})")
hide(a, len(num))
save("numeric_histograms")

sk = skew[skew.abs() > 1].index
a = grid(4, 5, 3.8, 2.8)
for ax, c in zip(a, sk):
    hist(ax, np.log1p(df[c].clip(lower=0).dropna()), f"log1p {c}", color="#9bbb59")
hide(a, len(sk))
save("numeric_histograms_log")

a = grid(5, 5, 3.8, 2.4)
for ax, c in zip(a, num):
    sns.boxplot(x=df[c].dropna(), ax=ax, color="#8064a2", fliersize=2)
    ax.set_title(c)
hide(a, len(num))
save("numeric_boxplots")

after = pd.Series({c: pd.Series(PowerTransformer().fit_transform(df[[c]].dropna())[:, 0]).skew() for c in num})
q1, q3 = df[num].quantile(.25), df[num].quantile(.75)
out = ((df[num] < q1 - 1.5 * (q3 - q1)) | (df[num] > q3 + 1.5 * (q3 - q1))).mean() * 100
a = grid(1, 2, 8, 8)
bar(a[0], pd.DataFrame({"raw": skew, "yeo-johnson": after}).sort_values("raw"), "skew before / after yeo-johnson", h=True)
bar(a[1], out.sort_values(), "% of rows outside 1.5 x IQR", h=True, color="#c0504d")
save("skew_and_outliers")
issues.append("skew above 2: " + ", ".join(skew[skew.abs() > 2].index))


# 4. budget and F-factor (gross / budget)
r = df["f_factor"]
print(f"\nF-factor: median {r.median():.2f}, share >= 1 is {(r >= 1).mean():.1%}, 5% {r.quantile(.05):.2f}, 95% {r.quantile(.95):.1f}")
print("highest:", df.nlargest(5, "f_factor")[["movie_title", "f_factor"]].round(1).values.tolist())
print("lowest :", df.nsmallest(5, "f_factor")[["movie_title", "f_factor"]].round(4).values.tolist())
a = grid(1, 3, 6.3, 6)
hist(a[0], np.log10(df["budget"]), "log10 budget", bins=50)
by_origin(a[1], "budget", "gross")
xx = np.array([df["budget"].min(), df["budget"].max()])
for k, ls in [(1, "-"), (3, "--"), (10, ":")]:
    a[1].plot(xx, xx * k, "k", ls=ls, lw=1, label=f"F-factor {k}x")
a[1].set(xscale="log", yscale="log", xlabel="budget", ylabel="gross", title="budget vs gross")
a[1].legend(fontsize=8)
hist(a[2], np.log10(r), "log10 F-factor (0 = break even)", bins=60, color="#9bbb59")
a[2].axvline(0, color="red", ls="--")
save("budget_f_factor_overview")

a = grid(1, 3, 6.3, 6)
a[0].scatter(df["budget"][~odd], df["gross"][~odd], s=8, alpha=.3, label="round budget")
a[0].scatter(df["budget"][odd], df["gross"][odd], s=10, alpha=.6, c="red", label="odd budget (imputed?)")
a[0].set(xscale="log", yscale="log", title="where the odd budgets sit")
a[0].legend()
a[1].scatter(df["budget"], df["budget_adj"], s=6, alpha=.4)
a[1].set(xscale="log", yscale="log", xlabel="budget", ylabel="budget_adj", title="budget vs inflation adjusted")
box(a[2], "era", "bM", "budget ($M) by era")
save("budget_details")
m = df.groupby("origin")["f_factor"].median()
issues += [f"budget_adj / budget_to_year_mean are budget rescaled by year (Spearman with budget {spear('budget', 'budget_adj'):.2f})",
           f"{(df['budget'] < 1e4).sum()} budgets under $10k: {df.loc[df['budget'] < 1e4, 'movie_title'].head(5).tolist()}",
           f"F-factor extremes: {(r < .01).sum()} films under 0.01x and {(r > 50).sum()} above 50x",
           f"median F-factor USA {m['USA']:.2f}x vs non-USA {m['non-USA']:.2f}x: gross is domestic, budget is worldwide"]

a = grid(2, 3)
bar(a[0], by_era["f_factor"].median(), "median F-factor by era")
bar(a[1], by_era["f_factor"].apply(lambda s: (s >= 1).mean()) * 100, "% of films with F-factor >= 1 by era", color="#9bbb59")
box(a[2], "content_rating", "f_factor", "F-factor by content rating", order=order)
box(a[3], "origin", "f_factor", "F-factor USA vs non-USA")
dec(a[4], "budget", "f_factor", "median F-factor by budget decile", xlog=True)
dec(a[5], "imdb_score", "f_factor", "median F-factor by imdb_score decile")
save("f_factor_breakdowns")


# 5. time
print("\nby era:\n", by_era.agg(films=("gross", "size"), gross=("gM", "median"), budget=("bM", "median"), duration=("duration", "median"),
                               score=("imdb_score", "mean"), usa=("is_usa", "mean")).round(2).to_string())
a = grid(2, 2, 8.5, 5)
n_year = yr.groupby("title_year").size()
a[0].bar(n_year.index, n_year.values, color="#8064a2")
a[0].set_title("films per year")
bar(a[1], yr.groupby("decade").size(), "films per decade")
yr.groupby("title_year")[["gM", "bM"]].median().loc[1960:].plot(ax=a[2], logy=True, title="median gross and budget per year ($M, log)")
yr.groupby("title_year")[["duration", "imdb_score"]].median().loc[1960:].plot(ax=a[3], secondary_y="imdb_score", title="median duration and imdb_score per year")
save("time_counts_medians")
issues.append(f"{(df['title_year'] < 1980).mean():.1%} of films are before 1980 ({(df['title_year'] < 1980).sum()} rows): era imbalance")

a = grid(2, 3)
sns.violinplot(data=yr, x="era", y="gross", cut=0, ax=a[0])
a[0].set(yscale="log", title="gross by era")
pd.crosstab(yr["era"], yr["content_rating"], normalize="index").plot.bar(stacked=True, ax=a[1], title="content rating mix by era")
for ax, (c, t, col) in zip(a[2:], [("is_usa", "share of USA films", "#4f81bd"), ("num_genres", "mean number of genres", "#9bbb59"),
                                   ("facenumber_in_poster", "mean faces in poster", "#f79646"), ("is_english", "share of English films", "#c0504d")]):
    bar(ax, yr.groupby("era", observed=True)[c].mean(), t, color=col)
save("era_breakdowns")

a = grid(1, 3)
for ax, c in zip(a, ["movie_facebook_likes", "director_facebook_likes", "actor_1_facebook_likes"]):
    bar(ax, (yr[c] == 0).groupby(yr["era"], observed=True).mean(), f"share of zeros in {c}", color="#c0504d")
save("zeros_by_era")
issues.append(f"movie_facebook_likes is 0 for {(df['movie_facebook_likes'] == 0).mean():.0%} of films (page did not exist before ~2010): zero means missing, not unpopular")


# 6. duration
d = df["duration"]
print(f"\nduration: median {d.median():.0f} min, 5-95% {d.quantile(.05):.0f}-{d.quantile(.95):.0f}, under 60: {(d < 60).sum()}, over 240: {(d > 240).sum()}")
a = grid(2, 3)
hist(a[0], d, "duration (min)", bins=50, kde=True)
box(a[1], "era", "duration", "duration by era", log=False)
a[2].hexbin(d, np.log10(df["gross"]), gridsize=35, cmap="viridis", mincnt=1)
a[2].set(xlabel="duration", ylabel="log10 gross", title="duration vs gross")
dec(a[3], "duration", "gM", "median gross ($M) by duration decile")
a[4].hexbin(d, df["imdb_score"], gridsize=35, cmap="magma", mincnt=1)
a[4].set(xlabel="duration", ylabel="imdb_score", title="duration vs score")
box(a[5], "content_rating", "duration", "duration by rating", order=order, log=False)
save("duration")
ex = df[(d < 60) | (d > 240)].sort_values("duration")
issues.append(f"{len(ex)} films under 60 or over 240 min: " + ", ".join(f"{t} ({int(x)}m)" for t, x in ex[["movie_title", "duration"]].head(4).values))


# 7. imdb score
s = df["imdb_score"]
print(f"\nimdb_score: mean {s.mean():.2f}, median {s.median():.2f}, sd {s.std():.2f}, skew {s.skew():.2f}")
print("best :", df.nlargest(5, "imdb_score")[["movie_title", "imdb_score"]].values.tolist())
print("worst:", df.nsmallest(5, "imdb_score")[["movie_title", "imdb_score"]].values.tolist())
a = grid(2, 3)
hist(a[0], s, "imdb_score", kde=True)
sns.violinplot(data=df, x="era", y="imdb_score", cut=0, ax=a[1])
a[1].set_title("score by era")
sc = a[2].scatter(np.log10(df["num_voted_users"]), s, c=np.log10(df["gross"]), s=8, cmap="viridis")
plt.colorbar(sc, ax=a[2], label="log10 gross")
a[2].set(xlabel="log10 votes", ylabel="score", title="score vs votes (colour = gross)")
dec(a[3], "imdb_score", "gM", "median gross ($M) by score decile")
box(a[4], "content_rating", "imdb_score", "score by rating", order=order, log=False)
box(a[5], "num_genres", "imdb_score", "score by number of genres", data=df[df["num_genres"] <= 7], log=False)
save("imdb_score")
issues.append(f"{(df['num_voted_users'] < 100).sum()} films have under 100 votes: their imdb_score is unreliable")


# 8. votes and reviews
a = grid(2, 3)
for ax, c in zip(a, ["num_voted_users", "num_critic_for_reviews", "num_user_for_reviews"]):
    hist(ax, np.log10(df[c]), f"log10 {c}  (Spearman with gross {spear(c, 'gross'):.2f})")
a[3].scatter(df["num_voted_users"], df["gross"], s=6, alpha=.3)
a[3].set(xscale="log", yscale="log", xlabel="votes", ylabel="gross", title="votes vs gross")
a[4].scatter(df["num_critic_for_reviews"], df["num_user_for_reviews"], s=6, alpha=.3)
a[4].set(xscale="log", yscale="log", xlabel="critic reviews", ylabel="user reviews", title="critic vs user reviews")
hist(a[5], np.log10(df["num_user_for_reviews"] / df["num_voted_users"]), "log10 user reviews per vote")
save("votes_reviews")


# 9. facebook likes
a = grid(2, 3, 6.3, 5)
for ax, c in zip(a, likes):
    hist(ax, np.log1p(df[c]), f"log1p {c}  ({(df[c] == 0).mean():.0%} zeros)", color="#4f81bd")
save("likes_distributions")

a = grid(2, 3, 6.3, 5)
for ax, c in zip(a, likes):
    ax.scatter(np.log1p(df[c]), np.log10(df["gross"]), s=5, alpha=.3)
    ax.set(xlabel="log1p likes", ylabel="log10 gross", title=f"{c}  rho={spear(c, 'gross'):.2f}")
save("likes_vs_gross")

top3 = df[["actor_1_facebook_likes", "actor_2_facebook_likes", "actor_3_facebook_likes"]].sum(axis=1)
bad = (df["cast_total_facebook_likes"] < top3).sum()
a = grid(1, 3, 6.3, 6)
a[0].scatter(top3 + 1, df["cast_total_facebook_likes"] + 1, s=6, alpha=.3)
a[0].plot([1, 1e6], [1, 1e6], "r--")
a[0].set(xscale="log", yscale="log", xlabel="sum of top 3 actors", ylabel="cast_total", title=f"cast total vs top 3 ({bad} inconsistent rows)")
hist(a[1], df["actor_1_facebook_likes"] / df["cast_total_facebook_likes"].replace(0, np.nan), "actor 1 share of cast total likes")
sns.heatmap(df[likes].corr(method="spearman"), annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=a[2])
a[2].set_title("likes correlation (Spearman)")
save("likes_consistency")
issues.append(f"director_facebook_likes is 0 for {(df['director_facebook_likes'] == 0).mean():.0%} of films, cast_total vs top-3 inconsistent rows: {bad}")


# 10. poster faces and aspect ratio
top_ar = df["aspect_ratio"].value_counts()
ar = df[df["aspect_ratio"].isin(top_ar.index[:5])]
a = grid(2, 3)
bar(a[0], df["facenumber_in_poster"].value_counts().sort_index(), "faces in poster")
box(a[1], "facenumber_in_poster", "gross", "gross by faces")
box(a[2], "facenumber_in_poster", "imdb_score", "score by faces", log=False)
bar(a[3], top_ar.head(10), "aspect ratio counts")
box(a[4], "aspect_ratio", "gross", "gross by aspect ratio (top 5)", data=ar)
pd.crosstab(ar["era"], ar["aspect_ratio"], normalize="index").plot.bar(stacked=True, ax=a[5], title="aspect ratio mix by era")
save("poster_aspect")
issues.append(f"aspect_ratio: {top_ar.index[0]} and {top_ar.index[1]} cover {top_ar.iloc[:2].sum() / len(df):.0%} of rows: treat as categorical")


# 11. content rating
a = grid(2, 3)
bar(a[0], df["content_rating"].value_counts().reindex(order), "films per rating")
for ax, c in zip(a[1:4], ["gross", "budget", "imdb_score"]):
    box(ax, "content_rating", c, f"{c} by rating", order=order, log=c != "imdb_score")
bar(a[4], df.groupby("content_rating")["f_factor"].apply(lambda s: (s >= 1).mean()).reindex(order) * 100, "% with F-factor >= 1", color="#9bbb59")
sns.heatmap(pd.crosstab(df["content_rating"], df["country_group"], normalize="index").reindex(order), annot=True, fmt=".2f", cmap="Blues", cbar=False, ax=a[5])
a[5].set_title("country mix inside each rating")
save("content_rating")
cr = df["content_rating"].value_counts()
issues.append(f"rare rating levels {cr[cr < 40].to_dict()}; 'Unrated' mixes real Not Rated with missing")


# 12. country and language
cc = df["country"].value_counts()
cm = df.groupby("country").agg(n=("gross", "size"), g=("gM", "median"), b=("bM", "median"), s=("imdb_score", "mean"))
cm = cm[cm["n"] >= 15].sort_values("g")
a = grid(2, 3)
bar(a[0], cc.head(15)[::-1], "films per country (top 15)", h=True)
bar(a[1], cm[["g", "b"]], "median gross and budget ($M), countries with 15+ films", h=True)
bar(a[2], cm["s"], "mean imdb_score", h=True, color="#9bbb59")
for ax, c in zip(a[3:], ["gross", "budget", "imdb_score"]):
    for u in ["USA", "non-USA"]:
        v = df.loc[df["origin"] == u, c]
        sns.kdeplot(v if c == "imdb_score" else np.log10(v), ax=ax, label=u, fill=True, alpha=.3)
    ax.set_title(f"{c} USA vs non-USA" + ("" if c == "imdb_score" else " (log10)"))
    ax.legend()
save("country")

a = grid(1, 3, 6.3, 6)
bar(a[0], df["language"].value_counts().head(12)[::-1], "films per language (log)", h=True, logx=True)
box(a[1], "is_english", "gross", "gross, English (1) vs other (0)")
sns.heatmap(pd.crosstab(df["country_group"], df["era"]), annot=True, fmt="d", cmap="Blues", cbar=False, ax=a[2])
a[2].set_title("country x era counts")
save("language")
issues.append(f"{df['is_usa'].mean():.0%} USA and {df['is_english'].mean():.1%} English: both are mostly one flag; {(cc < 10).sum()} countries have under 10 films")


# 13. genres
gc = G.sum().sort_values()
print("\ngenre counts:", gc[::-1].to_dict())
a = grid(1, 3, 6.5, 7)
bar(a[0], gc, "films per genre", h=True)
bar(a[1], df["num_genres"].value_counts().sort_index(), "genres per film")
bar(a[2], df["genres"].value_counts().head(15)[::-1], "15 most common genre combinations", h=True)
save("genre_counts")
issues.append(f"rare genres (under 30 films): {gc[gc < 30].to_dict()}")

big = G.sum().nlargest(20).index
co = G[big].T.dot(G[big])
n_g = G[big].sum().values
jac = co / (n_g[:, None] + n_g[None, :] - co)
a = grid(1, 2, 10.5, 9)
sns.heatmap(co, annot=True, fmt="d", cmap="YlGnBu", cbar=False, annot_kws={"size": 6}, ax=a[0])
a[0].set_title("genre co-occurrence counts")
sns.heatmap(jac.mask(np.eye(len(jac), dtype=bool)), annot=True, fmt=".2f", cmap="YlOrRd", cbar=False, annot_kws={"size": 6}, ax=a[1])
a[1].set_title("Jaccard overlap")
save("genre_cooccurrence")
print("most frequent genre pairs:\n", co.where(np.triu(np.ones(co.shape, bool), 1)).stack().nlargest(10).to_string())


def prof(m):
    d = df[m]
    return {"n": m.sum(), "gross": d["gM"].median(), "budget": d["bM"].median(), "f_factor": d["f_factor"].median(),
            "score": d["imdb_score"].mean(), "duration": d["duration"].median(), "profit%": 100 * (d["f_factor"] >= 1).mean()}


gt = pd.DataFrame({n: prof(G[n] == 1) for n in G.columns}).T
gt30 = gt[gt["n"] >= 30]
print(gt30.sort_values("gross", ascending=False).round(2).to_string())
a = grid(2, 3, 6.7, 6)
for ax, c, t in zip(a, ["gross", "budget", "f_factor", "score", "duration", "profit%"],
                    ["median gross ($M)", "median budget ($M)", "median F-factor", "mean imdb_score", "median duration", "% with F-factor >= 1"]):
    bar(ax, gt30[c].sort_values(), t, h=True)
save("genre_profiles")

a = grid(1, 3, 7, 7)
box(a[0], "num_genres", "gross", "gross by number of genres", data=df[df["num_genres"] <= 7])
sns.heatmap(G[big].corr().mask(np.eye(len(big), dtype=bool)), cmap="coolwarm", center=0, ax=a[1])
a[1].set_title("genre correlation (phi)")
sns.heatmap(G.join(df["era"]).groupby("era", observed=True).mean().T.loc[G.sum().nlargest(18).index], annot=True, fmt=".2f", cmap="YlOrRd", ax=a[2])
a[2].set_title("share of films with the genre, by era")
save("genre_extras")
cloud(G.sum(), "genres, bigger = more films", "viridis", "wordcloud_genres")
cloud(gt30["gross"], "genres, bigger = higher median gross", "magma", "wordcloud_genres_gross")


# 14. plot keywords
kw = df.loc[df["plot_keywords"] != "Unknown", ["plot_keywords", "gM", "imdb_score", "f_factor"]].copy()
kw["kw"] = kw["plot_keywords"].str.lower().str.split("|")   # file is Title Case, lowercase makes matching easy
kwx = kw.explode("kw")
kc = kwx["kw"].value_counts()
print(f"\n{len(kc)} distinct keywords, {(kc == 1).mean():.0%} appear once, {(kc >= 20).sum()} appear 20+ times")
print("top 15:", kc.head(15).to_dict())
cloud(kc, "plot keywords, bigger = more films", "plasma", "wordcloud_keywords", max_words=170)
kmed = kwx.groupby("kw")["gM"].agg(["size", "median"])
kmed = kmed[kmed["size"] >= 12]
cloud(kmed["median"], "keywords (12+ films), bigger = higher median gross", "cividis", "wordcloud_keywords_gross", max_words=150)

a = grid(1, 3, 6.7, 9)
bar(a[0], kc.head(30)[::-1], "30 most common keywords", h=True)
a[1].loglog(np.arange(1, len(kc) + 1), kc.values)
a[1].set_title("keyword rank vs frequency (log-log)")
bar(a[2], df["num_keywords"].value_counts().sort_index(), "keywords per film")
save("keywords_counts")

low = kmed.sort_values("median")
sel = pd.concat([low.head(12), low.tail(12)])
plt.figure(figsize=(9, 9))
plt.barh(sel.index, sel["median"], color=["#c0504d"] * 12 + ["#4f81bd"] * 12)
plt.title("keywords (12+ films) with lowest / highest median gross ($M)")
save("keywords_gross_extremes")

top25 = kc.head(25).index
K = pd.DataFrame({t: kw["kw"].apply(lambda L, t=t: t in L) for t in top25}).astype(int)
plt.figure(figsize=(12, 10))
sns.heatmap(K.T.dot(K).mask(np.eye(25, dtype=bool)), annot=True, fmt=".0f", cmap="YlGnBu", cbar=False, annot_kws={"size": 6})
plt.title("how often the top 25 keywords appear together")
save("keyword_cooccurrence")

terms = [t for t in ["sequel", "based on novel", "murder", "love", "friend", "revenge", "new york city", "female protagonist",
                     "violence", "blood", "death", "police", "teenager", "alien", "serial killer", "box office flop",
                     "box office hit", "cult film", "title directed by female"] if kc.get(t, 0) >= 20]
rows = []
for t in terms:
    has = kw["kw"].apply(lambda L: t in L)
    rows.append({"keyword": t, "films": has.sum(), "median gross with": kw.loc[has, "gM"].median(),
                 "median gross without": kw.loc[~has, "gM"].median(), "mean score with": kw.loc[has, "imdb_score"].mean()})
kt = pd.DataFrame(rows).set_index("keyword")
print(kt.round(2).to_string())
a = grid(1, 2, 8.5, 6)
bar(a[0], kt[["median gross with", "median gross without"]], "median gross ($M) with / without the keyword")
bar(a[1], kt["mean score with"], "mean imdb_score of films with the keyword (red = overall)", color="#9bbb59")
a[1].axhline(df["imdb_score"].mean(), color="red", ls="--")
save("keyword_effects")

leak = [(k, int(kc[k])) for k in kc.index if k.startswith(("box office", "critically")) or k == "cult film"]
meta = [(k, int(kc[k])) for k in kc.index if " title" in k or k.startswith("title")][:5]
issues += [f"keywords that describe the outcome (target leak): {leak}",
           f"keywords about the title itself: {meta}",
           f"{(df['plot_keywords'] == 'Unknown').sum()} films without keywords, {(df['num_keywords'] == 5).mean():.1%} have exactly 5",
           f"{(kc == 1).mean():.0%} of keywords appear in a single film, only {(kc >= 20).sum()} appear 20+ times: needs a min count"]


# 15. titles
stop = ["the", "of", "and", "in", "to", "for", "on", "at", "is", "it", "my", "your", "her", "his", "with", "from", "by", "you", "are"]
tw = df["movie_title"].str.lower().str.replace(r"[^a-z ]", "", regex=True).str.split().explode()
tw = tw[~tw.isin(stop) & (tw.str.len() > 2)]
a = grid(2, 3)
hist(a[0], df["title_len"], "title length (characters)")
bar(a[1], df["title_words"].value_counts().sort_index(), "words in title")
dec(a[2], "title_len", "gM", "median gross ($M) by title length decile")
box(a[3], "is_sequel", "gross", "gross: title looks like a sequel / has a colon")
box(a[4], "is_sequel", "imdb_score", "score: sequel-like title", log=False)
box(a[5], "title_words", "f_factor", "F-factor by words in title", data=df[df["title_words"] <= 8])
save("titles")
cloud(tw.value_counts(), "words in titles", "tab10", "wordcloud_titles", max_words=120)
dt = df[df.duplicated("movie_title", keep=False)].groupby("movie_title")["title_year"].nunique()
print(f"\n{df['is_sequel'].mean():.1%} of titles look like sequels, {(dt > 1).sum()} titles are different films with the same name (remakes)")


# 16. directors and actors
dc = df.loc[df["director_name"] != "Unknown", "director_name"].value_counts()
act = pd.concat([df[[c, "gM", "imdb_score"]].rename(columns={c: "actor"}) for c in ["actor_1_name", "actor_2_name", "actor_3_name"]])
act = act[act["actor"] != "Unknown"]
ac = act["actor"].value_counts()
d5 = df[df["director_name"].isin(dc[dc >= 5].index)]
print(f"\ndirectors {len(dc)} ({(dc == 1).mean():.0%} one film), actors {len(ac)} ({(ac == 1).mean():.0%} one film)")
print("most frequent director + lead pairs:\n", df[df["director_name"] != "Unknown"].groupby(["director_name", "actor_1_name"]).size().nlargest(6).to_string())
a = grid(2, 3, 6.7, 6)
bar(a[0], dc.head(20)[::-1], "directors with most films", h=True)
bar(a[1], ac.head(20)[::-1], "actors with most appearances (any slot)", h=True)
bar(a[2], d5.groupby("director_name")["gM"].mean().nlargest(15)[::-1], "directors (5+ films) by mean gross ($M)", h=True, color="#4f81bd")
bar(a[3], act[act["actor"].isin(ac[ac >= 8].index)].groupby("actor")["gM"].mean().nlargest(15)[::-1], "actors (8+ films) by mean gross ($M)", h=True, color="#4f81bd")
bar(a[4], d5.groupby("director_name")["imdb_score"].mean().nlargest(15)[::-1], "directors (5+ films) by mean score", h=True, color="#9bbb59")
cum = np.cumsum(np.sort(ac.values)[::-1]) / ac.sum()
a[5].plot(np.arange(1, len(cum) + 1) / len(cum), cum)
a[5].set(xlabel="share of actors (most active first)", ylabel="share of all appearances", title="how concentrated are actors")
save("people_top")

a = grid(1, 3)
vd, va = dc.value_counts().sort_index(), ac.value_counts().sort_index()
a[0].loglog(vd.index, vd.values, "o", label="directors")
a[0].loglog(va.index, va.values, "s", label="actors")
a[0].set(xlabel="films per person", ylabel="people", title="films per person (log-log)")
a[0].legend()
for ax, tab, col, who in [(a[1], df.groupby("director_name")["gM"].agg(["size", "mean"]), "#4f81bd", "director"),
                          (a[2], act.groupby("actor")["gM"].agg(["size", "mean"]), "#c0504d", "actor")]:
    ax.scatter(tab["size"], tab["mean"], s=8, alpha=.4, c=col)
    ax.set(yscale="log", xlabel="films", ylabel="mean gross $M", title=f"{who} mean gross vs films (noisy at 1)")
save("people_noise")
issues.append(f"{(dc == 1).mean():.0%} of directors and {(ac == 1).mean():.0%} of actors appear once, only {(dc >= 5).sum()} directors and {(ac >= 5).sum()} actors have 5+ films: needs a min count rule")


# 17. correlations and redundancy
a = grid(1, 2, 13, 11)
for ax, how in zip(a, ["pearson", "spearman"]):
    cmat = df[num].corr(method=how)
    sns.heatmap(cmat, mask=np.triu(np.ones_like(cmat, bool)), cmap="coolwarm", vmin=-1, vmax=1, annot=True, fmt=".2f",
                annot_kws={"size": 6}, cbar=False, ax=ax)
    ax.set_title(how)
save("correlation_heatmaps")
sns.clustermap(df[num].corr(method="spearman"), cmap="coolwarm", vmin=-1, vmax=1, figsize=(13, 13))
save("correlation_clustermap")

sp = df[feat].corrwith(df["gross"], method="spearman").sort_values()
a = grid(1, 2, 8.5, 8)
bar(a[0], sp, "Spearman with gross", h=True)
yj_r = {c: np.corrcoef(PowerTransformer().fit_transform(df[[c]].fillna(df[c].median()))[:, 0], yj_gross)[0, 1] for c in feat}
bar(a[1], pd.Series(yj_r).reindex(sp.index), "Pearson after yeo-johnson on both sides", h=True, color="#9bbb59")
save("correlation_with_gross")

c = df[feat].corr(method="spearman").abs()
pairs = c.where(np.triu(np.ones(c.shape, bool), 1)).stack()
pairs = pairs[pairs > .85].sort_values(ascending=False)
print("\nfeature pairs with |Spearman| above 0.85:\n", pairs.round(2).to_string())
issues.append("highly correlated pairs: " + ", ".join(f"{x}/{y} ({v:.2f})" for (x, y), v in pairs.items()))

Xm = df[feat].fillna(df[feat].median()).copy()
Xm["rating_code"] = df["content_rating"].astype("category").cat.codes
Xm["country_code"] = df["country_group"].astype("category").cat.codes
Xm["is_usa"] = df["is_usa"]
Xm = pd.concat([Xm, G.add_prefix("g_")], axis=1)
mi = pd.Series(mutual_info_regression(Xm, yj_gross, random_state=0), index=Xm.columns).sort_values().tail(25)
a = grid(1, 1, 8, 9)
bar(a[0], mi, "mutual information with gross (yeo-johnson)", h=True)
save("mutual_information")

sub = df.sample(1200, random_state=1)
sns.pairplot(pd.DataFrame({"log gross": np.log10(sub["gross"]), "log budget": np.log10(sub["budget"]), "log votes": np.log10(sub["num_voted_users"]),
                           "score": sub["imdb_score"], "duration": sub["duration"], "year": sub["title_year"]}),
             corner=True, plot_kws={"s": 8, "alpha": .4}, diag_kind="kde")
save("pairplot")


# 18. what moves gross
key = ["budget", "num_voted_users", "num_critic_for_reviews", "num_user_for_reviews", "imdb_score", "duration",
       "title_year", "actor_1_facebook_likes", "cast_total_facebook_likes", "director_facebook_likes"]
a = grid(2, 5, 5.2, 5)
for ax, c in zip(a, key):
    b = pd.qcut(df[c].rank(method="first"), 10, labels=False)
    t = df.groupby(b).agg(x=(c, "median"), m=("gM", "median"), lo=("gM", lambda s: s.quantile(.25)), hi=("gM", lambda s: s.quantile(.75)))
    ax.plot(t["x"], t["m"], "o-")
    ax.fill_between(t["x"], t["lo"], t["hi"], alpha=.2)
    ax.set(yscale="log", title=c)
plt.suptitle("median gross ($M) with IQR band across deciles of each feature", y=1.001)
save("gross_vs_features")

a = grid(1, 3, 6.7, 6)
for ax, c in zip(a, ["budget", "num_voted_users", "imdb_score"]):
    by_origin(ax, c, "gross")
    ax.set(yscale="log", xscale="linear" if c == "imdb_score" else "log", title=f"{c} vs gross by origin")
save("gross_by_origin")


# 19. odd rows
Z = pd.DataFrame(PowerTransformer().fit_transform(df[num].fillna(df[num].median())), columns=num)
print("\nrows furthest from the rest (largest |z| after yeo-johnson):")
print(df.assign(max_abs_z=Z.abs().max(axis=1), driver=Z.abs().idxmax(axis=1)).nlargest(15, "max_abs_z")
      [["movie_title", "title_year", "max_abs_z", "driver"]].round(2).to_string(index=False))
a = grid(1, 1, 8, 8)
bar(a[0], (Z.abs() > 4).sum().sort_values(), "rows with |z| > 4 after yeo-johnson", h=True)
save("outliers")


# issues
print("\n" + "=" * 90 + "\nISSUES FOUND\n" + "=" * 90)
for i, msg in enumerate(issues, 1):
    print(f"{i:2d}. {msg}")