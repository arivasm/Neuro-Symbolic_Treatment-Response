import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
from pyDatalog import pyDatalog
from pyDatalog.pyDatalog import assert_fact, load, ask

from SPARQLWrapper import SPARQLWrapper, JSON
from rdflib import Graph
import json

import networkx as nx

from scipy.stats import wilcoxon, kruskal
import random


def store_pharmacokinetic_ddi(effect):
    if effect in ['Excretion_rate', 'Excretory_function', 'excretion_rate']:
        effect = 'excretion'
    elif effect in ['Process_of_absorption']:
        effect = 'absorption'
    elif effect in ['Serum_concentration', 'Serum_concentration_of', 'Serum_level', 'serum_concentration']:
        effect = 'serum_concentration'
    elif effect in ['Metabolism', 'metabolism']:
        effect = 'metabolism'
    else:
        return 'pharmacodynamic'
    return effect


def rename_impact(impact):
    if impact in ['Increase', 'Higher', 'Worsening', 'higher', 'increase', 'worsening']:
        return 'increase'
    return 'decrease'


def combine_col(corpus, cols):
    # corpus = corpus.apply(lambda x: x.astype(str).str.lower())
    name = '_'.join(cols)
    corpus[name] = corpus[cols].apply(lambda x: '_'.join(x.values.astype(str)), axis=1)
    return corpus


def get_drug_label_by_category(drugs_cui, set_DDIs):
    d_label = set(set_DDIs.loc[set_DDIs.precipitantDrug.isin(drugs_cui)].EffectorDrugLabel.unique())
    d_label.update(set_DDIs.loc[set_DDIs.objectDrug.isin(drugs_cui)].AffectedDrugLabel.unique())
    return d_label


def load_ddi(asymmetric_ddi, input_data):
    set_asymmetric_ddi = asymmetric_ddi.loc[asymmetric_ddi.precipitantDrug.isin(input_data)]
    set_asymmetric_ddi = set_asymmetric_ddi.loc[set_asymmetric_ddi.objectDrug.isin(input_data)]
    set_DDIs = combine_col(set_asymmetric_ddi, ['Effect', 'Impact'])
    set_asymmetric_ddi.drop_duplicates(keep='first', inplace=True, ignore_index=True)
    comorbidity_drug = get_drug_label_by_category(input_data, set_asymmetric_ddi)
    set_DDIs = set_DDIs[['EffectorDrugLabel', 'AffectedDrugLabel', 'Effect_Impact']]
    return set_asymmetric_ddi, comorbidity_drug, set_DDIs


pyDatalog.create_terms('rdf_star_triple, inferred_rdf_star_triple, wedge, A, B, C, T, T2')


def build_datalog_model(union):
    pyDatalog.clear()
    for d in union.values:
        # Extensional Database
        assert_fact('rdf_star_triple', d[0], d[1], d[2])
    # Intentional Database
    inferred_rdf_star_triple(A, B, T) <= rdf_star_triple(A, B, T) & (T._in(ddiTypeToxicity))
    inferred_rdf_star_triple(A, C, T2) <= inferred_rdf_star_triple(A, B, T) & rdf_star_triple(B, C, T2) & (
        T._in(ddiTypeToxicity)) & (T2._in(ddiTypeToxicity)) & (A != C)
    wedge(A, B, C, T, T2) <= inferred_rdf_star_triple(A, B, T) & inferred_rdf_star_triple(B, C, T2) & (
        T._in(ddiTypeToxicity)) & (T2._in(ddiTypeToxicity)) & (A != C)

    inferred_rdf_star_triple(A, B, T) <= rdf_star_triple(A, B, T) & (T._in(ddiTypeEffectiveness))
    inferred_rdf_star_triple(A, C, T2) <= inferred_rdf_star_triple(A, B, T) & rdf_star_triple(B, C, T2) & (
        T._in(ddiTypeEffectiveness)) & (T2._in(ddiTypeEffectiveness)) & (A != C)
    wedge(A, B, C, T, T2) <= inferred_rdf_star_triple(A, B, T) & inferred_rdf_star_triple(B, C, T2) & (
        T._in(ddiTypeEffectiveness)) & (T2._in(ddiTypeEffectiveness)) & (A != C)


def compute_wedge_datalog(union):
    pyDatalog.clear()
    for d in union.values:
        # Extensional Database
        assert_fact('rdf_star_triple', d[0], d[1], d[2])
    # Intentional Database
    wedge(A, B, C, T, T2) <= rdf_star_triple(A, B, T) & rdf_star_triple(B, C, T2) & (T2._in(ddiTypeToxicity)) & (
        T._in(ddiTypeToxicity)) & (A != C)
    wedge(A, B, C, T, T2) <= rdf_star_triple(A, B, T) & rdf_star_triple(B, C, T2) & (T2._in(ddiTypeEffectiveness)) & (
        T._in(ddiTypeEffectiveness)) & (A != C)


def get_indirect_ddi(list_deduced_ddi, dsd, write=False):
    derived_ddi = []
    deduced_ddi = inferred_rdf_star_triple(C, dsd, T)
    for i in range(len(deduced_ddi)):
        x = {'EffectorDrugLabel': [deduced_ddi[i][0]], 'AffectedDrugLabel': dsd,
             'Effect_Impact': deduced_ddi[i][1]}  # + '_derived'
        list_deduced_ddi = pd.concat([list_deduced_ddi, pd.DataFrame(data=x)])
        if write:
            impact = deduced_ddi[i][1].split('_')[-1]
            effect = deduced_ddi[i][1].split('_')[:-1]
            effect = '_'.join([l for l in effect])
            # print(effect, impact)
            derived_ddi.append(deduced_ddi[i][0] + ' ' + impact + ' ' + effect + ' of ' + dsd + ' (derived)')

    return list_deduced_ddi, derived_ddi


def get_indirect_ddi_treatment(set_dsd_label, write):
    list_deduced_ddi = pd.DataFrame(columns=['EffectorDrugLabel', 'AffectedDrugLabel', 'Effect_Impact'])
    text_derived_ddi = []
    for dsd in set_dsd_label:
        list_deduced_ddi, derived_ddi = get_indirect_ddi(list_deduced_ddi, dsd, write)
        text_derived_ddi = text_derived_ddi + derived_ddi
    return list_deduced_ddi, text_derived_ddi


def create_json_to_cytoscape(union, k):
    graph_json = dict()
    graph_json['nodes'] = []
    graph_json['edges'] = []
    drug_id = dict()
    id_x = 0
    for i in range(union.shape[0]):
        precipitant = union.iloc[i]['EffectorDrugLabel']
        object_d = union.iloc[i]['AffectedDrugLabel']
        ddi = union.iloc[i]['Effect_Impact']
        edge = dict()
        edge['data'] = dict()

        if precipitant in drug_id.keys():
            edge['data']['id'] = id_x
            edge['data']['source'] = drug_id[precipitant]
            edge['data']['Effect_Impact'] = ddi
            id_x += 1
        else:
            node = dict()
            node['data'] = dict()
            drug_id[precipitant] = id_x
            node['data']['id'] = id_x
            node['data']['name'] = precipitant
            edge['data']['id'] = id_x + 1
            edge['data']['source'] = id_x
            edge['data']['Effect_Impact'] = ddi
            graph_json['nodes'].append(node)
            id_x += 2
        if object_d in drug_id.keys():
            edge['data']['target'] = drug_id[object_d]
        else:
            node = dict()
            node['data'] = dict()
            drug_id[object_d] = id_x
            node['data']['id'] = id_x
            node['data']['name'] = object_d
            edge['data']['target'] = id_x
            graph_json['nodes'].append(node)
            id_x += 1
            if object_d == k:
                node['classes'] = 'red'  # Single class

        graph_json['edges'].append(edge)

    return graph_json


# # Whole Graph enriched
def get_graph_enriched(plot_ddi, comorbidity_drug, set_DDIs):
    build_datalog_model(plot_ddi)
    indirect_ddi, text_derived_ddi = get_indirect_ddi_treatment(comorbidity_drug, write=True)
    # mechanism = ddiTypeEffectiveness + ddiTypeToxicity
    # g_i, increased_ddi = visualise_treatment(plot_ddi, list(set_dsd_label), 'Graph_initial', (7, 5), mechanism,
    #                                         adverse_event,
    #                                         plot_treatment=True, graph_enriched=False)
    graph_ddi = pd.concat([plot_ddi, indirect_ddi])
    graph_ddi.drop_duplicates(keep='first', inplace=True)

    # if plot_ddi.shape[0] == 0:
    #     return graph_ddi, plot_ddi
    # remove_ddi = pd.merge(indirect_ddi, plot_ddi, how='inner')

    # return ((graph_ddi.shape[0] - plot_ddi.shape[0]) / graph_ddi.shape[0]) * 100, graph_ddi, text_derived_ddi
    return graph_ddi, text_derived_ddi


def computing_wedge(set_drug_label):
    dict_wedge = dict()
    dict_frequency = dict()
    for d in set_drug_label:
        w = wedge(A, d, C, T, T2)
        indirect_ddi = pd.DataFrame(columns=['EffectorDrugLabel', 'AffectedDrugLabel', 'Effect_Impact'])
        for i in range(len(w)):
            x = {'EffectorDrugLabel': [w[i][0], d], 'AffectedDrugLabel': [d, w[i][1]],
                 'Effect_Impact': [w[i][2], w[i][3]]}
            indirect_ddi = pd.concat([indirect_ddi, pd.DataFrame(data=x)])

        indirect_ddi.drop_duplicates(keep='first', inplace=True)
        dict_wedge[d] = indirect_ddi
        dict_frequency[d] = len(w)
    return dict_wedge, dict_frequency


ddiTypeToxicity = ["serum_concentration_increase", "metabolism_decrease", "absorption_increase", "excretion_decrease"]
ddiTypeEffectiveness = ["serum_concentration_decrease", "metabolism_increase", "absorption_decrease",
                        "excretion_increase"]


# adverse_event, union, set_dsd_label, comorbidity_drug, set_DDIs = load_data(file)


def capture_knowledge(union, comorbidity_drug, set_DDIs):
    plot_graph = union[['EffectorDrugLabel', 'AffectedDrugLabel', 'Effect_Impact']]
    # plot_ddi.drop_duplicates(keep='first', inplace=True)
    graph_ddi, text_derived_ddi = get_graph_enriched(plot_graph, comorbidity_drug, set_DDIs)  # plot_ddi

    union['Class'] = ''
    union.loc[union.Effect_Impact.isin(ddiTypeToxicity), "Class"] = 'HigherToxicity'
    union.loc[union.Effect_Impact.isin(ddiTypeEffectiveness), "Class"] = 'LowerEffectiveness'
    g1 = union[['objectDrug', 'AffectedDrugLabel', 'Class']]
    g1.drop_duplicates(keep='first', inplace=True)
    graph_ddi['Class'] = ''
    graph_ddi.loc[graph_ddi.Effect_Impact.isin(ddiTypeToxicity), "Class"] = 'HigherToxicity'
    graph_ddi.loc[graph_ddi.Effect_Impact.isin(ddiTypeEffectiveness), "Class"] = 'LowerEffectiveness'

    graph_ddi['precipitantDrug'] = ''
    graph_ddi['objectDrug'] = ''
    precipitant = dict(zip(union.precipitantDrug, union.EffectorDrugLabel))
    object_d = dict(zip(union.objectDrug, union.AffectedDrugLabel))
    for key, value in precipitant.items():
        graph_ddi.loc[graph_ddi.EffectorDrugLabel == value, 'precipitantDrug'] = key
    for key, value in object_d.items():
        graph_ddi.loc[graph_ddi.AffectedDrugLabel == value, 'objectDrug'] = key

    g2 = graph_ddi[['AffectedDrugLabel', 'Class']]
    g2.drop_duplicates(keep='first', inplace=True)
    return g1, g2, union, graph_ddi


def discovering_knowledge(adverse_event, union, set_dsd_label, comorbidity_drug):
    plot_ddi = union[['EffectorDrugLabel', 'AffectedDrugLabel', 'Effect_Impact']]
    plot_ddi.drop_duplicates(keep='first', inplace=True)
    build_datalog_model(plot_ddi)
    dict_wedge, dict_frequency = computing_wedge(comorbidity_drug.union(set_dsd_label))
    dict_frequency = dict(sorted(dict_frequency.items(), key=lambda item: item[1], reverse=True))

    dict_graph_json = dict()
    for k, v in dict_frequency.items():
        if v > 0:
            # visualise_treatment(dict_wedge[k], {k}, 'middle_vertex' + k, (7, 5), ddiTypeToxicity, adverse_event,
            #                    plot_treatment=True, graph_enriched=False)
            graph_json = create_json_to_cytoscape(dict_wedge[k], k)
            dict_graph_json[k] = graph_json
    return dict_graph_json, dict_frequency


def evaluation_without_deduction(union, set_dsd_label, comorbidity_drug):
    plot_ddi = union[['EffectorDrugLabel', 'AffectedDrugLabel', 'Effect_Impact']]
    plot_ddi.drop_duplicates(keep='first', inplace=True)
    compute_wedge_datalog(plot_ddi)
    dict_wedge, dict_frequency = computing_wedge(comorbidity_drug.union(set_dsd_label))
    dict_frequency = dict(sorted(dict_frequency.items(), key=lambda item: item[1], reverse=True))
    return dict_frequency


def load_dataset_ddi(path):
    asymmetric_ddi = pd.read_csv(path, delimiter=",")
    asymmetric_ddi.rename(columns={'EffectorDrugID': 'precipitantDrug', 'AffectedDrugID': 'objectDrug'},
                          inplace=True)
    asymmetric_ddi.loc[asymmetric_ddi['Adverse events'].isin(
        ['Excretion_rate', 'Excretory_function', 'excretion_rate']), 'Effect'] = 'excretion'
    asymmetric_ddi.loc[asymmetric_ddi['Adverse events'].isin(['Process_of_absorption', 'Absorption']),
                       'Effect'] = 'absorption'
    asymmetric_ddi.loc[asymmetric_ddi['Adverse events'].isin(
        ['Serum_concentration', 'Serum_concentration_of', 'Serum_level',
         'serum_concentration']), 'Effect'] = 'serum_concentration'
    asymmetric_ddi.loc[asymmetric_ddi['Adverse events'].isin(['Metabolism', 'metabolism']), 'Effect'] = 'metabolism'

    asymmetric_ddi = asymmetric_ddi.loc[
        asymmetric_ddi.Effect.isin(['excretion', 'absorption', 'serum_concentration', 'metabolism'])]

    asymmetric_ddi = asymmetric_ddi[
        ['EffectorDrugLabel', 'AffectedDrugLabel', 'Effect', 'Impact', 'precipitantDrug', 'objectDrug']]

    asymmetric_ddi.loc[asymmetric_ddi.Impact.isin(
        ['Increase', 'Higher', 'Worsening', 'higher', 'increase', 'worsening']), 'Impact'] = 'increase'
    asymmetric_ddi.loc[asymmetric_ddi.Impact.isin(['Decrease', 'Lower', 'Reduced', 'Reduction']),
                       'Impact'] = 'decrease'

    asymmetric_ddi = asymmetric_ddi.dropna()
    asymmetric_ddi.drop_duplicates(keep='first', inplace=True, ignore_index=True)
    return asymmetric_ddi
