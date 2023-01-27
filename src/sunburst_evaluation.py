import os
import numpy as np
import pandas as pd
import plotly
import plotly.express as px
from sklearn.metrics import roc_auc_score

def add_cats_to_dataframe(df, categories):
    simple_df = df[['post1_category', "is_duplicate", "prediction"]]
    simple_df['Slug'] = simple_df['post1_category']
    simple_df = simple_df.join(categories.set_index("Slug"), "Slug")
    simple_df['Cat2'] = simple_df['Cat2'].fillna("NONE_" + simple_df['Cat1'])
    simple_df['Cat3'] = simple_df['Cat3'].fillna("NONE_" + simple_df['Cat2'])
    simple_df['null'] = ""
    simple_df['Queue'] = os.environ["QUEUE_NAME"]
    return simple_df


def calculate_metrics(df):
    try:
        auc_score = roc_auc_score(y_true=df['is_duplicate'], y_score=df['prediction'])
    except:
        auc_score = 0
    df_count = len(df)
    duplicate_percentage = df['is_duplicate'].mean() 
    
    return (auc_score, df_count, duplicate_percentage)


def get_per_cat_metrics(df):
    parent_relations = [["null", "Queue"], ["Queue", "Cat1"], ["Cat1", "Cat2"], ["Cat2", "Cat3"]]
    per_cat_metrics = []
    for relation in parent_relations:
        metrics = df.groupby(relation).apply(calculate_metrics)
        per_cat_metrics.append(metrics)

    return per_cat_metrics


def get_metrics_for_plot(per_cat_metrics, df):
    data = dict(
        node=[],
        parent=[],
        duplicate_ratio=[],
        partition_count=[],
        auc_score=[],
    )

    for metrics in per_cat_metrics:
        for a in metrics.items():
            data["node"].append(a[0][1])
            data["parent"].append(a[0][0])
            
            data["duplicate_ratio"].append(a[1][2])
            data["partition_count"].append(a[1][1])
            data["auc_score"].append(a[1][0])
    metrics = pd.DataFrame(data = data)   
    metrics['submit_percentage'] = (metrics['partition_count'] / len(df)) * 100.0
     
    return metrics

def check_row_should_exist(parent, parents_with_alone_childs):
    return (parent not in parents_with_alone_childs) or (parent == "") or (parent == os.environ["QUEUE_NAME"])

def remove_alone_childs(metrics):
    parents_children_count = metrics.groupby("parent").apply(lambda x: len(x))
    parents_with_alone_childs = list(parents_children_count[parents_children_count == 1].index)
    cleaned_metrics = metrics[metrics['parent'].apply(lambda parent: check_row_should_exist(parent, parents_with_alone_childs))]
    return cleaned_metrics


def calculate_cleaned_metrics(df):
    categories = pd.read_csv("../../storage/data/prepared/categories.csv")
    simple_df = add_cats_to_dataframe(df, categories)
    per_cat_metrics = get_per_cat_metrics(simple_df)
    metrics = get_metrics_for_plot(per_cat_metrics, df)
    cleaned_metrics = remove_alone_childs(metrics)
    return cleaned_metrics

    
def write_plot(cleaned_metrics, plot_name):
    fig = px.sunburst(
        cleaned_metrics,
        names='node',
        parents='parent',
        hover_data=['duplicate_ratio', "submit_percentage"],
        values='partition_count',
        color='auc_score',
        branchvalues="total",
        color_continuous_scale='RdBu',
    
    )
    fig.write_html(f"../../logs/{plot_name}.html")

def create_and_save_sunburst_plot(df, plot_name):    
    cleaned_metrics = calculate_cleaned_metrics(df)
    write_plot(cleaned_metrics, plot_name)
    