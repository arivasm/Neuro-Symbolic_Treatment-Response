from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline
from pykeen.models import predict
import pykeen.nn
from typing import List
import pandas as pd
import numpy as np
import statistics
from scipy.ndimage import gaussian_filter1d
import math
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.colors as mcolors
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from sklearn.decomposition import PCA
from rdflib import Graph
from rdflib.plugins.sparql.processor import SPARQLResult


def load_dataset(path, name):
    triple_data = open(path + name).read().strip()
    data = np.array([triple.split('\t') for triple in triple_data.split('\n')])
    tf_data = TriplesFactory.from_labeled_triples(triples=data)
    return tf_data, triple_data


def filter_prediction(predicted_heads_df, constraint):
    predicted_heads_df = predicted_heads_df[predicted_heads_df.head_label.str.contains(constraint)]
    predicted_heads_df = reset_index(predicted_heads_df)
    return predicted_heads_df


def filter_by_type(predicted_heads, triple_data, entity_type):
    list_entity = predicted_heads.head_label
    entity = []
    for s in list_entity:
        for triple in triple_data:
            b = [s, '<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>', entity_type] == triple
            if np.all(b):
                entity.append(s)
                break
    predicted_heads = predicted_heads.loc[predicted_heads.head_label.isin(entity)]
    predicted_heads = reset_index(predicted_heads)
    return predicted_heads, entity


def get_threshold(predicted_heads, percentile):
    score_values = predicted_heads.score.values
    threshold = np.percentile(score_values, percentile)
    threshold_index = predicted_heads.loc[predicted_heads.score > threshold].shape[0]
    print(threshold, threshold_index)
    return threshold, threshold_index


def get_inflection_point(score_values):
    # standard deviation
    stdev = statistics.stdev(score_values)
    # smooth
    smooth = gaussian_filter1d(score_values, stdev)
    # compute second derivative
    smooth_d2 = np.gradient(np.gradient(smooth))
    # find switching points
    infls = np.where(np.diff(np.sign(smooth_d2)))[0]
    if len(infls) == 1:
        return infls[0]
    if len(infls) == 0:
        return len(score_values)
    # middle inflection point
    m_infls = infls[math.ceil(len(infls) / 2)]
    return m_infls


def get_precision(predicted_heads, inflection_index):
    tp_fp = predicted_heads.iloc[0:inflection_index + 1]
    tp = tp_fp.loc[tp_fp.in_training == True].shape[0]
    prec = tp / tp_fp.shape[0]
    return prec, tp


def get_recall(predicted_heads, tp):
    tp_fn = predicted_heads.loc[predicted_heads.in_training == True].shape[0]
    rec = tp / tp_fn
    return rec


def get_f_measure(precision, recall):
    f_measure = 2 * (precision * recall) / (precision + recall)
    return f_measure


def reset_index(predicted_heads):
    predicted_heads.reset_index(inplace=True)
    predicted_heads.drop(columns=['index'], inplace=True)
    return predicted_heads


def compute_metrics(predicted_heads, cut_index, model):
    precision, tp = get_precision(predicted_heads, cut_index)
    recall = get_recall(predicted_heads, tp)
    f_measure = get_f_measure(precision, recall)
    return pd.DataFrame(columns=['precision', 'recall', 'f_measure'],
                        data=[[precision, recall, f_measure]], index=[model])


def plot_score_value(score_values, title):
    plt.plot(score_values)
    plt.xlabel("Entities")
    plt.ylabel("Score")
    plt.title(title)
    plt.show()
    plt.close()


def get_learned_embeddings(model):
    entity_representation_modules: List['pykeen.nn.RepresentationModule'] = model.entity_representations
    relation_representation_modules: List['pykeen.nn.RepresentationModule'] = model.relation_representations

    entity_embeddings: pykeen.nn.Embedding = entity_representation_modules[0]
    relation_embeddings: pykeen.nn.Embedding = relation_representation_modules[0]

    entity_embedding_tensor: torch.FloatTensor = entity_embeddings()
    relation_embedding_tensor: torch.FloatTensor = relation_embeddings()
    return entity_embedding_tensor, relation_embedding_tensor


def create_dataframe_predicted_entities(entity_embedding_tensor, entity, training):
    df = pd.DataFrame(entity_embedding_tensor.cpu().detach().numpy())
    df['target'] = list(training.entity_to_id)
    new_df = df.loc[df.target.isin(list(entity))]
    return new_df.iloc[:, :-1], new_df, df


def elbow_KMeans(matrix, k_min, k_max, n):
    model = KMeans()
    visualizer = KElbowVisualizer(model, k=(k_min, k_max))
    visualizer.fit(matrix)
    num_cls = visualizer.elbow_value_
    visualizer.show(outpath='Plots/elbow_KG_' + str(n) + ".pdf", bbox_inches='tight')
    return num_cls


def plot_cluster(num_cls, new_df, n):
    X = new_df.copy()
    kmeans = KMeans(n_clusters=num_cls, random_state=0)
    new_df['cluster'] = kmeans.fit_predict(new_df)
    # define and map colors
    col = list(colors.cnames.values())
    col = col[:num_cls]
    index = list(range(num_cls))
    color_dictionary = dict(zip(index, col))
    new_df['c'] = new_df.cluster.map(color_dictionary)
    #####PLOT#####
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(1, figsize=(8, 8))
    # plot data
    pca = PCA(n_components=2).fit(X)
    pca_c = pca.transform(X)
    plt.scatter(pca_c[:, 0], pca_c[:, 1], c=new_df.c, alpha=0.6, s=50)

    # create a list of legend elemntes
    ## markers / records
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='Cluster {}'.format(i + 1),
                              markerfacecolor=mcolor, markersize=10) for i, mcolor in enumerate(col)]
    # plot legend
    plt.legend(handles=legend_elements, loc='upper right', fontsize=16)
    # title and labels
    plt.title('Clusters of Entities predicted', loc='left', fontsize=22)
    plt.savefig(fname='Plots/KMeans_KG_' + str(n) + ".pdf", format='pdf', bbox_inches='tight')
    plt.show()


def plot_treatment(new_df, predicted_heads_tox, predicted_heads_eff, n):
    toxicity = list(predicted_heads_tox.loc[predicted_heads_tox.in_training == True].head_label)
    effect = list(predicted_heads_eff.loc[predicted_heads_eff.in_training == True].head_label)
    # new_df['cls'] = 'safe'
    new_df.loc[new_df.target.isin(toxicity), 'cls'] = 'effective'
    new_df.loc[new_df.target.isin(effect), 'cls'] = 'low-effect'
    X = new_df.iloc[:, :-2].copy()

    # define and map colors
    col = list(colors.cnames.values())
    # col = [col[9], col[3]]
    col = [mcolors.CSS4_COLORS['brown'], mcolors.CSS4_COLORS['lightcoral']]
    index = ['effective', 'low-effect']
    color_dictionary = dict(zip(index, col))
    new_df['c'] = new_df.cls.map(color_dictionary)
    #####PLOT#####
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(1, figsize=(8, 8))
    # plot data
    pca = PCA(n_components=2).fit(X)
    pca_c = pca.transform(X)
    plt.scatter(pca_c[:, 0], pca_c[:, 1], c=new_df.c, s=50)  # alpha=0.6,

    # create a list of legend elemntes
    ## markers / records
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=key,
                              markerfacecolor=mcolor, markersize=10) for key, mcolor in color_dictionary.items()]
    # plot legend
    plt.legend(handles=legend_elements, loc='upper right', fontsize=16)
    # title and labels
    if n == 1:
        plt.title('Treatments in ' + '${\cal{T\_KG}}_{basic}$', loc='left', fontsize=22)
    elif n == 2:
        plt.title('Treatments in ' + '$\cal{T\_KG}$', loc='left', fontsize=22)
    else:
        plt.title('Treatments in ' + '${\cal{T\_KG}}_{random}$', loc='left', fontsize=22)
    #plt.savefig(fname='Plots/PCA_KG_' + str(n) + ".png", format='png', bbox_inches='tight', dpi=300, transparent=True)
    #plt.savefig(fname='Plots/PCA_KG_' + str(n) + ".pdf", format='pdf', bbox_inches='tight')
    plt.show()


def load_graph(file_name):
    g1 = Graph()
    g1.parse(file_name, format="ttl")
    return g1


def sparql_results_to_df(results: SPARQLResult) -> pd.DataFrame:
    """
    Export results from an rdflib SPARQL query into a `pandas.DataFrame`,
    using Python types. See https://github.com/RDFLib/rdflib/issues/1179.
    """
    return pd.DataFrame(
        data=([None if x is None else x.toPython() for x in row] for row in results),
        columns=[str(x) for x in results.vars],
    )


def get_triple(graph, predicted_heads, entity_type):
    list_entity = list(predicted_heads.head_label)
    list_entity = ', '.join(list_entity)
    query = """    
    select distinct ?s
    where {
        ?s <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> """ + entity_type + """
        FILTER (?s in (""" + str(list_entity) + """))
        }
        """
    results = graph.query(query)
    df_cls = sparql_results_to_df(results)
    df_cls['s'] = '<' + df_cls['s'].astype(str) + '>'
    entity = list(df_cls.s)

    predicted_heads = predicted_heads.loc[predicted_heads.head_label.isin(entity)]
    predicted_heads = reset_index(predicted_heads)
    return predicted_heads, entity
