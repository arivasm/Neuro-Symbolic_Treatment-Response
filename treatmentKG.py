import itertools
import sys

import deductive_system
import random
from collections import OrderedDict
import pandas as pd
import numpy as np
import time


class Treatment:

    def set_decrease_effectiveness(self):
        self.decrease_effectiveness = True

    def set_effective(self):
        self.effective = True

    def set_decrease_effectiveness_2(self):
        self.decrease_effectiveness_2 = True

    def set_effective_2(self):
        self.effective_2 = True

    def set_name(self, name):
        self.name = name

    def set_value_treatment(self, treatment, classified_drug, graph_ddi, g1_classified, g2_classified):
        self.treatment = treatment
        self.classified_drug = classified_drug
        self.graph_ddi = graph_ddi
        self.g1_classified = g1_classified
        self.g2_classified = g2_classified

    def __init__(self):
        self.name = ''
        self.effective = False
        self.decrease_effectiveness = False
        self.effective_2 = False
        self.decrease_effectiveness_2 = False
        self.treatment = []
        self.num_ddi_deduced = ''
        self.classified_drug = ''
        self.graph_ddi = ''
        self.g1_classified = ''
        self.g2_classified = ''


def preprocessing_oncological_drugs(treatment):
    replacement_mapping_dict = {
        'Bevacizumab vs Placebo': 'Bevacizumab',
        'Capecitabina': 'Capecitabine',
        'Carboplatino': 'Carboplatin',
        'Ciclofosfamida': 'Cyclophosphamide',
        'Cisplatino': 'Cisplatin',
        'Custirsen(OGX-011)': 'Custirsen',
        'Demcizumab vs Placebo': 'Demcizumab',
        'Doxorrubicina': 'Doxorubicin',
        'Durvalumab (MEDI4736)': 'Durvalumab',
        'Durvalumab(MEDI4736) vs Placebo': 'Durvalumab',
        'Emibetuzumab (LY2875358)': 'Emibetuzumab',
        'Etopósido VP16': 'Etoposide',
        'Gemcitabina': 'Gemcitabine',
        'Lurbinectedina (PM1183)': 'Lurbinectedin',
        'Onartuzumab (MetMAb) vs Placebo': 'Onartuzumab',
        'PM060184': 'Pm-060184',
        'Pembrolizumab vs placebo': 'Pembrolizumab',
        'Prexasertib(LY2606368)': 'Prexasertib',
        'Temozolomida': 'Temozolomide',
        'Trabectidina': 'Trabectedin',
        'Veliparib(ABT-888)': 'Veliparib',
        'Veliparib(ABT-888) vs Placebo': 'Veliparib',
        'Vincristina': 'Vincristine',
        'Vinorelbina': 'Vinorelbine',

        # ===== from positive treatment =====
        'Vadimezan (ASA404)': 'Vadimezan',

        # ===== unknow_drug =====
        'Irvalec': np.nan,
        'Otro': np.nan,
    }

    treatment.rename(columns={'medicalrecord_CT': 'patient_id'}, inplace=True)
    treatment['ct_drug'].replace(replacement_mapping_dict, inplace=True)
    treatment['ct_drug.1'].replace(replacement_mapping_dict, inplace=True)
    treatment['ct_drug.2'].replace(replacement_mapping_dict, inplace=True)
    return treatment


def preprocessing_nonOncological_drugs(drug_no_onco):
    replacement_mapping_dict = {
        'Enalaprile': 'Enalapril',
        'Atenolole': 'Atenolol',
        'Azitromicina': 'Azithromycin',
        'Immunoglobulin_g': 'Human_immunoglobulin_g',
    }
    drug_no_onco['drug_name'].replace(replacement_mapping_dict, inplace=True)
    drug_no_onco = drug_no_onco.loc[
        ~drug_no_onco.drug_name.isin(['Antibiotic', 'Analgesic', 'Antibiotics', 'Insulin', 'Corticoids'])]
    return drug_no_onco


def create_treatment(drug_no_onco, treatment):
    # == combine oncological drugs by patient and date
    treatment['drug_name'] = treatment[['ct_drug', 'ct_drug.1', 'ct_drug.2']].applymap(
        lambda x: [] if x is np.nan else [x]).apply(lambda x: sum(x, []), axis=1)
    treatment = treatment[['patient_id', 'fecinitto', 'drug_name']]
    treatment = treatment.loc[treatment.astype(str).drop_duplicates().index].reset_index()
    treatment = treatment.drop(columns=['index'])

    # == concat non-oncological drug with patients taking oncological drugs ==
    cancer_treatment = pd.merge(treatment[['patient_id', 'fecinitto']], drug_no_onco, how='inner', on=['patient_id'])
    cancer_treatment.drop_duplicates(keep='first', inplace=True)
    cancer_treatment = cancer_treatment.groupby(by=['patient_id', 'fecinitto']).agg(lambda x: x.tolist()).reset_index()
    treatment = pd.concat([cancer_treatment, treatment])

    # == create treatment onco drugs + non-onco drugs by patient_id and date
    treatment = treatment.groupby(by=['patient_id', 'fecinitto']).agg(lambda x: sum(x, [])).reset_index()
    return treatment


def preprocess_treatment(df_treatment, non_onco_drug):
    set_drugs = set()
    index_remove = []
    # == sort cancer treatment and remove duplicate ==
    for i in range(df_treatment.shape[0]):
        # == write the number of drug of the treatment
        drugs = df_treatment.drug_name[i]
        #if len(drugs) == 2 and len(set(drugs).intersection(non_onco_drug)) > 0:
        #    index_remove.append(i)
        #    continue
        # print(drugs, i)
        drugs.sort()
        df_treatment.at[i, 'n_drugs'] = len(drugs)
        df_treatment.at[i, 'drug_name'] = drugs
        set_drugs.update(set(drugs))

    #df_treatment.drop(index_remove, inplace=True)
    df_treatment = df_treatment.loc[df_treatment['n_drugs'] > 1]
    df_treatment = df_treatment.sort_values(by=['n_drugs'], ascending=False)
    df_treatment = df_treatment[['drug_name', 'n_drugs']]
    df_treatment = df_treatment.loc[df_treatment.astype(str).drop_duplicates().index].reset_index()
    df_treatment = df_treatment.drop(columns=['index'])
    return df_treatment, set_drugs


def generate_kg_treatment(df_treatment, drugBank_id, ddi, init, treatment_class):
    for i in range(df_treatment.shape[0]):
        list_d = df_treatment.drug_name[i]
        input_data = drugBank_id.loc[drugBank_id.DrugName.isin(list_d)].DrugBankID.values
        union, comorbidity_drug, set_DDIs = deductive_system.load_ddi(ddi, input_data)
        g1, g2, g1_classified, g2_classified = deductive_system.capture_knowledge(union, comorbidity_drug, set_DDIs)
        max_ddi = len(g1_classified.Effect_Impact.unique()) * len(list_d) * (len(list_d) - 1) / 2

        t = Treatment()
        t.set_value_treatment(input_data, g1, g2, g1_classified, g2_classified)
        if treatment_class == 'positive':
            t.set_effective()
            t.set_effective_2()
        else:
            # == treatment_class == 'negative':
            t.set_decrease_effectiveness()
            t.set_decrease_effectiveness_2()

        t.set_name('treatment' + str(init + i))

        graph_ttl, graph_2_ttl = create_turtle_file(t)
        save_ttl_file(graph_ttl, 'G1.ttl')
        save_ttl_file(graph_2_ttl, 'G2.ttl')


def get_drugbank_id(set_drugs, drugBank_id_name):
    drug_treatment = pd.DataFrame(set_drugs, columns=['DrugName'])
    return pd.merge(drugBank_id_name, drug_treatment, on='DrugName')


def create_turtle_file(treatment):
    graph_ttl = """@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix ex: <http://example/#> .
@prefix treatment_drug: <http://example/Treatment_Drug#> .
@prefix ddi: <http://example/DrugDrugInteraction#> .
ex:higher_toxicity	rdf:type	ex:HigherToxicity .
ex:lower_effectiveness	rdf:type	ex:LowerEffectiveness .\n"""
    graph_ttl = """"""
    treatment_class = """"""

    graph_ttl = graph_ttl + """<http://example/Treatment/""" + treatment.name + """>\trdf:type\tex:Treatment .\n"""

    if treatment.effective:
        graph_ttl = graph_ttl + """<http://example/Treatment/""" + treatment.name + """>\tex:hasClassificationEffect\tex:effective .\n"""
    if treatment.decrease_effectiveness:
        graph_ttl = graph_ttl + """<http://example/Treatment/""" + treatment.name + """>\tex:hasClassificationEffect\tex:decrease_effectiveness .\n"""
    if treatment.effective_2:
        treatment_class = treatment_class + """<http://example/Treatment/""" + treatment.name + """>\tex:hasClassificationEffect\tex:effective .\n"""
    if treatment.decrease_effectiveness_2:
        treatment_class = treatment_class + """<http://example/Treatment/""" + treatment.name + """>\tex:hasClassificationEffect\tex:decrease_effectiveness .\n"""

    drug_list = treatment.treatment
    for d in drug_list:
        treatment_drug = treatment.name + '_' + d
        graph_ttl = graph_ttl + """<http://example/Drug/""" + d + """>\trdf:type\tex:Drug .\n"""
        graph_ttl = graph_ttl + """treatment_drug:""" + treatment_drug + """\trdf:type\tex:Treatment_Drug .\n"""
        graph_ttl = graph_ttl + """treatment_drug:""" + treatment_drug + """\tex:related_to\t<http://example/Treatment/""" + treatment.name + """> .\n"""
        graph_ttl = graph_ttl + """treatment_drug:""" + treatment_drug + """\tex:related_to\t<http://example/Drug/""" + d + """> .\n"""
        cls = treatment.classified_drug.loc[treatment.classified_drug.objectDrug == d].Class.values
        for c in cls:
            if c == 'HigherToxicity':
                graph_ttl = graph_ttl + """treatment_drug:""" + treatment_drug + """\tex:hasHighToxicity\tex:higher_toxicity .\n"""
            else:
                graph_ttl = graph_ttl + """treatment_drug:""" + treatment_drug + """\tex:hasLowerEffect\tex:lower_effectiveness .\n"""

    ddi_g1 = get_ddi_triples(treatment, treatment.g1_classified)
    ddi_g2 = get_ddi_triples(treatment, treatment.g2_classified)

    graph_2_ttl = graph_ttl + treatment_class + ddi_g2
    graph_ttl = graph_ttl + ddi_g1
    return graph_ttl, graph_2_ttl


def get_ddi_triples(treatment, g):
    graph_ttl = ''
    for index, row in g.iterrows():
        graph_ttl = graph_ttl + """ddi:""" + treatment.name + row['precipitantDrug'] + row[
            'objectDrug'] + """\trdf:type\tex:DDI .\n"""
        # == adding effect and impact of a DDI
        #graph_ttl = graph_ttl + """ddi:""" + treatment.name + row['precipitantDrug'] + row[
        #    'objectDrug'] + """\trdf:ddiEffect\tex:""" + row['Effect_Impact'] + """ .\n"""

        graph_ttl = graph_ttl + """<http://example/Treatment/""" + treatment.name + """>\tex:related_to\tddi:""" + \
                    treatment.name + row['precipitantDrug'] + row['objectDrug'] + """ .\n"""

        # graph_ttl = graph_ttl + """ddi:""" + treatment.name + row['precipitantDrug'] + row[
        #    'objectDrug'] + """\tex:related_to\t<http://example/Treatment/""" + treatment.name + """> .\n"""

        graph_ttl = graph_ttl + """ddi:""" + treatment.name + row['precipitantDrug'] + row[
            'objectDrug'] + """\tex:precipitant_drug\t<http://example/Drug/""" + row['precipitantDrug'] + """> .\n"""
        graph_ttl = graph_ttl + """ddi:""" + treatment.name + row['precipitantDrug'] + row[
            'objectDrug'] + """\tex:object_drug\t<http://example/Drug/""" + row['objectDrug'] + """> .\n"""
    return graph_ttl


def save_ttl_file(graph_ttl, name):
    with open(name, 'a') as file:
        file.write(graph_ttl)


def main(*args):
    start = time.time()
    input_path = 'Input/'

    negative_treatment = input_path + 'treatment_negative_effect.csv'
    positive_treatment = input_path + 'treatment_positive_effect.csv'
    d_no_onco = input_path + 'patient_nonOncologycalTreatments.csv'

    negative_treatment = pd.read_csv(negative_treatment, delimiter=",")
    positive_treatment = pd.read_csv(positive_treatment, delimiter=",")
    drug_no_onco = pd.read_csv(d_no_onco, delimiter=",")
    ddi = deductive_system.load_dataset_ddi('../store_data/drug/Unsymmetric_DDI_corpus.csv')
    drugBank_id_name = pd.read_csv('../store_data/drug/drugBank_id_name.csv', delimiter=",")

    drug_no_onco = preprocessing_nonOncological_drugs(drug_no_onco)
    non_onco_drug = list(drug_no_onco.drug_name.unique())
    if args[1] == 'positive':
        treatment = preprocessing_oncological_drugs(positive_treatment)
    else:
        treatment = preprocessing_oncological_drugs(negative_treatment)
    treatment = create_treatment(drug_no_onco, treatment)
    df_treatment, set_drugs = preprocess_treatment(treatment, non_onco_drug)
    drugBank_id = get_drugbank_id(set_drugs, drugBank_id_name)
    generate_kg_treatment(df_treatment, drugBank_id, ddi, int(args[0]), args[1])
    # generate_kg_treatment(df_treatment, drugBank_id, ddi, 524, 'positive')

    end = time.time()
    print('get_treatment_list: ', end - start)

    # graph_ttl, graph_2_ttl = create_turtle_file(treatment_list)
    # graph_ttl = "\n".join(list(OrderedDict.fromkeys(graph_ttl.split("\n"))))
    # graph_2_ttl = "\n".join(list(OrderedDict.fromkeys(graph_2_ttl.split("\n"))))
    # save_ttl_file(graph_ttl, args[0])
    # save_ttl_file(graph_2_ttl, '02' + args[0])


if __name__ == '__main__':
    main(*sys.argv[1:])
