from pykeen.triples import TriplesFactory

from pykeen.pipeline import pipeline
from pykeen.models import predict
import pandas as pd
import numpy as np
import statistics
from scipy.ndimage import gaussian_filter1d
import math
import sys
import torch


def select_graph(n):
    # th_dec_eff = 9.6551724138  # means 524
    # th_eff = 90.3448275862  # means 56

    # th_lowEffect = 30.645161290322577 # means 430
    # th_effective = 69.35483870967742 # means 190
    th_lowEffect = 27.19266055 # means 399
    th_effective = 72.80733945 # means 149
    #n_sample = 124
    #n_sample_effective = [38, 34, 40, 35, 43]
    #n_sample_lowEffect = [n_sample - x for x in n_sample_effective]
    #th_effective = [1 - x / n_sample for x in n_sample_effective]
    #th_lowEffect = [1 - x / n_sample for x in n_sample_lowEffect]
    
    if n == 1:
        file_name = 'config_g1.csv'
    elif n == 2:
        file_name = 'config_g2.csv'
    else:
        file_name = 'config_g3.csv'
    return file_name, n, th_lowEffect, th_effective


# # Load Train data
def load_dataset(path, name):
    triple_data = open(path + name).read().strip()
    data = np.array([triple.split('\t') for triple in triple_data.split('\n')])
    tf_data = TriplesFactory.from_labeled_triples(triples=data)
    return tf_data, triple_data


def create_model(tf_training, tf_testing, embedding, n_epoch, path, fold):
    results = pipeline(
        training=tf_training,
        testing=tf_testing,
        model=embedding,  # 'TransE',  #'RotatE'
        # stopper='early',
        # stopper_kwargs=dict(frequency=5, patience=2, relative_delta=0.002),
        #training_loop='sLCWA',
        #negative_sampler='bernoulli',
        negative_sampler_kwargs=dict(
        filtered=True,
        ),
        # Training configuration
        training_kwargs=dict(
            num_epochs=n_epoch,
            use_tqdm_batch=False,
        ),
        # Runtime configuration
        random_seed=1235,
        device='gpu',
    )
    model = results.model
    results.save_to_directory(path + embedding + str(fold))
    return model, results


# # Predict links (Head prediction)
def predict_heads(model, prop, obj, tf_testing):  # triples_factory=results.training
    predicted_heads_df = predict.get_head_prediction_df(model, prop, obj, triples_factory=tf_testing)
    return predicted_heads_df


# Filter the prediction by the head 'treatment_drug:treatment'. We are not interested in predict another links
def filter_prediction(predicted_heads_df, constraint):
    predicted_heads_df = predicted_heads_df[predicted_heads_df.head_label.str.contains(constraint)]
    return predicted_heads_df


def save_statistics(path, line):
    with open(path + 'results_threshold.csv', 'a') as file:
        file.write(line)


def get_config(config_file):
    config = pd.read_csv(config_file, delimiter=";")  # 'config_G1.csv'
    models = config.model.values[0].split(',')
    epochs = config.epochs.values[0]
    k = config.k_fold.values[0]
    path = config.path.values[0]
    graph_name = config.graph_name.values[0]
    return models, epochs, k, path, graph_name


def load_testset_classes(path, name):
    r = pd.read_csv(path + name, delimiter='\t', header=None)
    r.columns = ['head_label', 'p', 'o']
    r['o'] = r['o'].str.replace(' .', '')
    r_tox = r.loc[r.o == 'ex:effective']
    head_tox = list(r_tox.head_label)
    r_eff = r.loc[r.o == 'ex:low_effect']
    head_eff = list(r_eff.head_label)
    return head_tox, head_eff


def adding_testset(predicted_heads, head):
    predicted_heads.loc[predicted_heads.head_label.isin(head), 'in_training'] = True

    predicted_heads.reset_index(inplace=True)
    predicted_heads.drop(columns=['index'], inplace=True)
    return predicted_heads


def get_threshold(predicted_heads, percentile):
    score_values = predicted_heads.score.values
    threshold = np.percentile(score_values, percentile)
    threshold_index = predicted_heads.loc[predicted_heads.score > threshold].shape[0]
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


def main(*args):
    file_name, n, th_dec_eff, th_eff = select_graph(int(args[0]))
    models, epochs, k, path, graph_name = get_config(file_name)
    # models = ['TransH','RotatE', 'TransE', 'TransD', 'HolE', 'TransR', 'ERMLP', 'QuatE', 'RESCAL', 'SE', 'UM']
    models = ['TransH','RotatE', 'SE', 'UM', 'TransE', 'TransD', 'HolE', 'TransR', 'ERMLP', 'QuatE', 'RESCAL']
    for m in models:
        precision = 0
        recall = 0
        f_measure = 0
        for i in range(0, k):
            tf_training, triple_train = load_dataset(path, 'train_' + str(i + 1) + '.ttl')
            tf_testing, triple_test = load_dataset(path, 'test_' + str(i + 1) + '.ttl')
            model, results = create_model(tf_training, tf_testing, m, epochs, path, i + 1)
            #model = torch.load(path + m + str(i + 1) + '/trained_model.pkl') # , map_location='cpu'
            predicted_heads_eff = predict_heads(model, 'ex:belong_to', 'ex:effective', tf_testing) #tf_training
            predicted_heads_dec_eff = predict_heads(model, 'ex:belong_to', 'ex:low_effect',tf_testing)

            threshold, threshold_index = get_threshold(predicted_heads_dec_eff, th_dec_eff)
            precision_dec_eff, tp = get_precision(predicted_heads_dec_eff, threshold_index)
            recall_dec_eff = get_recall(predicted_heads_dec_eff, tp)
            f_measure_dec_eff = 0
            if (precision_dec_eff + recall_dec_eff) > 0:
                f_measure_dec_eff = get_f_measure(precision_dec_eff, recall_dec_eff)
            
            
            threshold, threshold_index = get_threshold(predicted_heads_eff, th_eff)
            precision_eff, tp = get_precision(predicted_heads_eff, threshold_index)
            recall_eff = get_recall(predicted_heads_eff, tp)
            f_measure_eff = 0
            if (precision_eff + recall_eff) > 0:
                f_measure_eff = get_f_measure(precision_eff, recall_eff)
            
            
            precision += (precision_eff + precision_dec_eff) / 2
            recall += (recall_eff + recall_dec_eff) / 2
            f_measure += (f_measure_eff + f_measure_dec_eff) / 2
            # print(precision, recall, f_measure)

        avg_precision = precision / k
        avg_recall = recall / k
        avg_f_measure = f_measure / k
        line = m + ';' + str(avg_precision) + ';' + str(avg_recall) + ';' + str(avg_f_measure) + '\n'
        save_statistics(path, line)
        print(line)


if __name__ == '__main__':
    main(*sys.argv[1:])

