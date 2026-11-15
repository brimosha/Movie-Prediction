from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Wedge
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
import seaborn as sm
from wordcloud import WordCloud


def _axis(ax, figsize):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        return fig, ax, True
    return ax.figure, ax, False


def _show(fig, ax, show, created):
    if show is None:
        show = created
    if show:
        plt.show()
    return fig, ax


def save_g(plot, path):
    fig, _ = plot
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# Histograms
def hist(x: list, bins, title: str, x_l: str, y_l: str,
         color: str, edgecolor: str, ax=None, show=None,
         reference=None, reference_label=None, **options):
    fig, ax, created = _axis(ax, (8, 6))
    sm.histplot(x, bins=bins, color=color, edgecolor=edgecolor, ax=ax, **options)
    if reference is not None:
        ax.axvline(reference, color="#c0504d", linestyle="--", label=reference_label)
        ax.legend()
    ax.set(xlabel=x_l, ylabel=y_l, title=title)
    return _show(fig, ax, show, created)


def hist_group(x_list: list, labels: list, title: str, x_l: str, y_l: str = "Density",
               colors: list = None, vlines=None, ax=None, show=None, **options):
    fig, ax, created = _axis(ax, (8, 6))
    colors = colors or [None] * len(x_list)
    for values, label, color in zip(x_list, labels, colors):
        sm.kdeplot(x=values, ax=ax, label=label, color=color, fill=True, **options)
    for value, label in vlines or []:
        ax.axvline(value, linestyle="--", linewidth=1, label=label)
    ax.set(xlabel=x_l, ylabel=y_l, title=title)
    ax.legend()
    return _show(fig, ax, show, created)


# Vertical bar charts
def bar_g(x: list, y: list, title: str, x_l: str, y_l: str, color: str,
          ax=None, show=None, **options):
    fig, ax, created = _axis(ax, (8, 6))
    ax.bar(x, y, color=color, **options)
    ax.set(xlabel=x_l, ylabel=y_l, title=title)
    return _show(fig, ax, show, created)


def bar_multi_g(df: pd.DataFrame, kind: str, stacked: bool, title: str,
                x_l: str, y_l: str, colors: list, loc: str = "best",
                ax=None, show=None, **options):
    fig, ax, created = _axis(ax, (8, 6))
    df.plot(kind=kind, stacked=stacked, color=colors, ax=ax, **options)
    ax.set(xlabel=x_l, ylabel=y_l, title=title)
    ax.legend(loc=loc)
    return _show(fig, ax, show, created)


# Horizontal bar charts
def barh_g(x: list, y: list, title: str, x_l: str, y_l: str, color: str,
           reference=None, reference_label=None, ax=None, show=None, **options):
    fig, ax, created = _axis(ax, (8, 6))
    ax.barh(y, x, color=color, **options)
    if reference is not None:
        ax.axvline(reference, color="#c0504d", linestyle="--", label=reference_label)
        ax.legend()
    ax.set(xlabel=x_l, ylabel=y_l, title=title)
    return _show(fig, ax, show, created)


# Pie charts
def pie_g(x: list, labels: list, title: str, colors: list,
          ax=None, show=None, **options):
    fig, ax, created = _axis(ax, (7, 7))
    ax.pie(x, labels=labels, colors=colors, **options)
    ax.set_title(title)
    return _show(fig, ax, show, created)


# Box plots
def boxplot_g(x: list, title: str, x_l: str, ax=None, show=None, **options):
    fig, ax, created = _axis(ax, (8, 4))
    ax.boxplot(x, patch_artist=True, **options)
    ax.set(title=title, xlabel=x_l)
    return _show(fig, ax, show, created)


def boxplot_group_g(data, x_col: str, y_col: str, title: str, order=None,
                    ax=None, show=None, **options):
    fig, ax, created = _axis(ax, (8, 5))
    sm.boxplot(data=data, x=x_col, y=y_col, order=order, ax=ax, **options)
    ax.set_title(title)
    return _show(fig, ax, show, created)


def violin_g(data, x_col: str, y_col: str, title: str, order=None,
             ax=None, show=None, **options):
    fig, ax, created = _axis(ax, (8, 5))
    sm.violinplot(data=data, x=x_col, y=y_col, order=order, cut=0, ax=ax, **options)
    ax.set_title(title)
    return _show(fig, ax, show, created)


# Scatter plots
def scatter_g(x: list, y: list, title: str, x_l: str, y_l: str, color: str,
              size: int, ax=None, show=None, **options):
    fig, ax, created = _axis(ax, (8, 6))
    x_scale = options.pop("x_scale", "linear")
    y_scale = options.pop("y_scale", "linear")
    ax.scatter(x, y, color=color, s=size, **options)
    ax.set(xscale=x_scale, yscale=y_scale, xlabel=x_l, ylabel=y_l, title=title)
    return _show(fig, ax, show, created)


def scatter_group_g(df: pd.DataFrame, x_col: str, y_col: str, group_col: str,
                    title: str, x_l: str, y_l: str, colors: list, size: int,
                    ax=None, show=None, **options):
    fig, ax, created = _axis(ax, (8, 6))
    for group, color in zip(df[group_col].unique(), colors):
        part = df[df[group_col] == group]
        ax.scatter(part[x_col], part[y_col], color=color, s=size, label=str(group), **options)
    ax.set(xlabel=x_l, ylabel=y_l, title=title)
    ax.legend()
    return _show(fig, ax, show, created)


def wordcloud_g(frequencies, title: str, colormap: str, ax=None, show=None, **options):
    fig, ax, created = _axis(ax, (10, 6))
    cloud = WordCloud(
        width=1200, height=700, background_color="white",
        colormap=colormap, random_state=1, **options
    ).generate_from_frequencies(dict(frequencies))
    ax.imshow(cloud, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title)
    return _show(fig, ax, show, created)


def ecdf_g(x, title: str, x_l: str, y_l: str = "Cumulative share",
           x_scale: str = "linear", ax=None, show=None, **options):
    values = np.sort(pd.Series(x).dropna().to_numpy())
    fig, ax, created = _axis(ax, (8, 5))
    ax.step(values, np.arange(1, len(values) + 1) / len(values), where="post", **options)
    ax.set(xscale=x_scale, xlabel=x_l, ylabel=y_l, title=title)
    return _show(fig, ax, show, created)

#Coeff heatmap
def heatmap_g(data, title: str, ax=None, show=None, **options):
    fig, ax, created = _axis(ax, (9, 7))
    sm.heatmap(data, ax=ax, **options)
    ax.set_title(title)
    return _show(fig, ax, show, created)

#Dendogram for Spearman Coeff
def dendrogram_g(correlation: pd.DataFrame, title: str, ax=None, show=None):
    distance = 1 - np.abs(correlation.fillna(0).to_numpy())
    np.fill_diagonal(distance, 0)
    tree = linkage(squareform(distance, checks=False), method="average")
    fig, ax, created = _axis(ax, (10, 6))
    dendrogram(tree, labels=correlation.columns, leaf_rotation=35,
               leaf_font_size=8, color_threshold=.45, ax=ax)
    ax.set(xlabel="", ylabel="1 - |Spearman rho|", title=title)
    return _show(fig, ax, show, created)


#Sunburst
def sunburst_g(data: pd.DataFrame, title: str, colors: list = None, ax=None, show=None):
    counts = data.fillna(0)
    parents = counts.index
    children = counts.columns
    parent_colors = sm.color_palette("pastel", n_colors=len(parents))
    child_colors = colors or sm.color_palette("colorblind", n_colors=len(children))
    fig, ax, created = _axis(ax, (10, 7))
    total = counts.to_numpy().sum()
    start = 90
    for parent, color in zip(parents, parent_colors):
        row = counts.loc[parent]
        parent_span = row.sum() / total * 360
        ax.add_patch(Wedge((0, 0), .66, start, start + parent_span, width=.34,
                           facecolor=color, edgecolor="white", linewidth=1))
        if parent_span > 12:
            angle = np.deg2rad(start + parent_span / 2)
            ax.text(.49 * np.cos(angle), .49 * np.sin(angle), str(parent),
                    ha="center", va="center", fontsize=8)
        child_start = start
        for child, child_color in zip(children, child_colors):
            child_span = row[child] / total * 360
            if child_span:
                ax.add_patch(Wedge((0, 0), 1, child_start, child_start + child_span,
                                   width=.34, facecolor=child_color,
                                   edgecolor="white", linewidth=.7))
                child_start += child_span
        start += parent_span
    ax.text(0, 0, "Films", ha="center", va="center", fontsize=12, weight="bold")
    ax.set(aspect="equal", xlim=(-1.05, 2), ylim=(-1.1, 1.1), title=title)
    ax.axis("off")
    handles = [Patch(color=color, label=str(child))
               for child, color in zip(children, child_colors)]
    ax.legend(handles=handles, title="Content rating", loc="center left",
              bbox_to_anchor=(.72, .5), frameon=False)
    return _show(fig, ax, show, created)


#Lolipop
def lollipop_g(labels: list, values: list, title: str, x_l: str,
               color: str = "#4f81bd", reference=None, reference_label=None,
               ax=None, show=None, **options):
    order = np.argsort(values)
    labels = np.asarray(labels)[order]
    values = np.asarray(values)[order]
    fig, ax, created = _axis(ax, (8, 5))
    ax.hlines(labels, 0, values, color=color, alpha=.65)
    ax.scatter(values, labels, color=color, zorder=3, **options)
    if reference is not None:
        ax.axvline(reference, color="#c0504d", linestyle="--", label=reference_label)
        ax.legend()
    ax.set(xlabel=x_l, title=title)
    return _show(fig, ax, show, created)

#Dumbbell
def dumbbell_g(labels: list, first: list, second: list, first_name: str,
               second_name: str, title: str, x_l: str, x_scale: str = "linear",
               ax=None, show=None, **options):
    y = np.arange(len(labels))
    fig, ax, created = _axis(ax, (8, 5))
    ax.hlines(y, first, second, color="#b7b7b7")
    ax.scatter(first, y, label=first_name, color="#4f81bd", **options)
    ax.scatter(second, y, label=second_name, color="#c0504d", **options)
    ax.set(yticks=y, yticklabels=labels, xscale=x_scale, xlabel=x_l, title=title)
    ax.legend()
    return _show(fig, ax, show, created)


