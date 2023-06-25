import random
import numpy as np
from goatools.obo_parser import GODag
from goatools.associations import read_gaf
from goatools.semantic import TermCounts, get_info_content
from goatools.semantic import resnik_sim
import pkg_resources

def score_cv(test_nodes, test_labelling, real_labelling):
    """Scores cross validation by counting the number of test nodes that
    were accurately labeled after their removal from the true
    labelling.
    """
    correct = 0
    total   = 0
    for node in test_nodes:
        if node not in test_labelling:
            continue
        test_label = test_labelling[node]
        if type(test_label) is list:
            for tl in test_label:
                if tl in real_labelling[node]:
                    correct += 1
                    break
        else:
            if test_label in real_labelling[node]:
                correct += 1
        total += 1
    return float(correct) / float(total)

def kfoldcv(k, 
            labels, 
            prediction_algorithm, 
            randomized=True, 
            reverse = False,
            filter_f_test = lambda x : True,
           filter_f_train = lambda x : True):
    """
    Performs k-fold cross validation.
    Args:
      - A number of folds k
      - A labeling for the nodes.
      - An algorithm that takes the training labels
      and outputs a predicted labelling.

    Output: 
      - A list where each element is the accuracy of the
      learning algorithm holding out one fold of the data.
    """
    nodes = list(labels.keys())
    if randomized:
        random.shuffle(nodes)
    accuracies = []
    for i in range(0, k):
        inc = int(len(nodes) / k)
        x = inc * i
        y = inc * (i + 1)
        if i + 1 == k:
            y = len(nodes)
        if not reverse:
            tr_nodes = nodes[:x] + nodes[y:]
            ts_nodes        = nodes[x:y]
        else:
            tr_nodes  = nodes[x:y]
            ts_nodes         = nodes[:x] + nodes[y:]
        test_nodes     = []
        training_nodes = []
        # Choose only the test nodes that have TRUE `filter_f_test` value.
        for node in ts_nodes:
            if filter_f_test(node):
                test_nodes.append(node)
            
        # Choose only the train nodes that have TRUE `filter_f_train` value
        for node in tr_nodes:
            if filter_f_train(node):
                training_nodes.append(node)
         
        training_labels = {n: labels[n] for n in training_nodes} # if filter_f(n)}
        test_labelling  = prediction_algorithm(training_labels)
        
        accuracy = score_cv(test_nodes, 
                            test_labelling, 
                            labels)
        accuracies.append(accuracy)
    return accuracies

def kfoldcv_with_pr(k, 
                    labels, 
                    prediction_algorithm, 
                    randomized=True, 
            filter_f_test = lambda x : True,
           filter_f_train = lambda x : True,
                    reverse = False,
                    use_go_parents = False,
                    parents = None):
    """Performs k-fold cross validation.

    Args:
      - A number of folds k
      - A labeling for the nodes.
      - An algorithm that takes the training labels
      and outputs a predicted labelling.

    Output: 
      - A list where each element is the accuracy of the
      learning algorithm holding out one fold of the data.
    """
    nodes = list(labels.keys())
    if randomized:
        random.shuffle(nodes)
    fscores = []
    for i in range(0, k):
        inc = int(len(nodes) / k)
        x = inc * i
        y = inc * (i + 1)
        if i + 1 == k:
            y = len(nodes)
        if not reverse:
            tr_nodes = nodes[:x] + nodes[y:]
            ts_nodes = nodes[x:y]
        else:
            tr_nodes  = nodes[x:y]
            ts_nodes  = nodes[:x] + nodes[y:]
        test_nodes     = []
        training_nodes = []
        # Choose only the test nodes that have TRUE `filter_f_test` value.
        for node in ts_nodes:
            if filter_f_test(node):
                test_nodes.append(node)
            
        # Choose only the train nodes that have TRUE `filter_f_train` value
        for node in tr_nodes:
            if filter_f_train(node):
                training_nodes.append(node)
        
        training_labels = {n: labels[n] for n in training_nodes}
        test_labelling  = prediction_algorithm(training_labels)
        
        fmax = score_cv_pr(test_nodes, 
                           test_labelling, 
                           labels, 
                           use_go_parents = use_go_parents, 
                           parents = parents)
        fscores.append(fmax)
    return fscores


def score_cv_pr(test_nodes, 
                test_labelling, 
                real_labelling, 
                ci = 1000, 
                use_go_parents = False, 
                parents = None):
    """Scores cross validation by counting the number of test nodes that
    were accurately labeled after their removal from the true
    labelling.
    """
    if use_go_parents:
        test_labelling, real_labelling = add_parent_labels(test_nodes, 
                                                           test_labelling, 
                                                           real_labelling, 
                                                           parent_dict = parents)

    def compute_fmax(precs, recalls):
        f1 = [2 * (p * r) / (p+r) if p+r != 0 else 0 for (p, r) in zip(precs, recalls)]
        return np.max(f1)

    cis = [i / ci for i in range(ci)]
    precision = []
    recall    = []
    for c in cis:
        prec_counter = 0
        rec_counter  = 0
        prs          = 0
        rcs          = 0
        for node in test_nodes:
            if node not in test_labelling:
                continue
            # print(test_labelling)
            pred_labels = set([t for (t,c1) in test_labelling[node] if c1 >= c])
            if len(pred_labels) != 0:
                prec_counter += 1
            true_labels = set(real_labelling[node])
            if len(true_labels) != 0:
                rec_counter += 1
            prs += len(pred_labels.intersection(true_labels)) / float(len(pred_labels)) if len(pred_labels) != 0 else 0 
            rcs += len(pred_labels.intersection(true_labels)) / float(len(true_labels)) if len(true_labels) != 0 else 0
        prs     = prs / prec_counter if prec_counter != 0 else 0
        rcs     = rcs / rec_counter  if rec_counter  != 0 else 0
        precision.append(prs)
        recall.append(rcs)
    fmax  = compute_fmax(precision, recall)
    return fmax 


def add_parent_labels(test_nodes, test_labelling, real_labelling, parent_dict):
    print("Adding Parent Labels")
    p_test_labelling = {}
    p_real_labelling = {}
    for node in test_nodes:
        if node not in test_labelling:
            continue
        parent_test = {}
        for (t, c) in test_labelling[node]:
            parent_test[t] = c
            for label in parent_dict[t]:
                if label not in parent_test:
                    parent_test[label] = c
                else:
                    parent_test[label] = max(c, parent_test[label])
        p_test_labelling[node] = [(t, c) for t, c in parent_test.items()]
        parent_real = set()
        for t in real_labelling[node]:
            parent_real.update([t] + parent_dict[t])
        p_real_labelling[node] = list(parent_real)
    print("Parent Labels Added")
    return p_test_labelling, p_real_labelling


def kfoldcv_sim(k, 
                labels, 
                prediction_algorithm, 
                randomized=True, 
                reverse = False,
                namespace = "MF",
                ci = 20,
                avg = False):
    """Performs k-fold cross validation.

    Args:
      - A number of folds k
      - A labeling for the nodes.
      - An algorithm that takes the training labels
      and outputs a predicted labelling.

    Output: 
      - A list where each element is the accuracy of the
      learning algorithm holding out one fold of the data.
    """
    gdagfile = pkg_resources.resource_filename('glide', 'data/go-basic.obo.dat')
    assoc_f  = pkg_resources.resource_filename('glide', 'data/go-human.gaf.dat')
    godag    = GODag(gdagfile)
    assoc    = read_gaf(assoc_f, namespace = namespace)
    t_counts = TermCounts(godag, assoc)
    nodes = list(labels.keys())
    if randomized:
        random.shuffle(nodes)
    fscores = []
    for i in range(0, k):
        inc = int(len(nodes) / k)
        x = inc * i
        y = inc * (i + 1)
        if i + 1 == k:
            y = len(nodes)
        if not reverse:
            training_nodes = nodes[:x] + nodes[y:]
            test_nodes = nodes[x:y]
        else:
            training_nodes  = nodes[x:y]
            test_nodes      = nodes[:x] + nodes[y:]
            
        training_labels = {n: labels[n] for n in training_nodes}
        test_labelling  = prediction_algorithm(training_labels)
        
        fmax = score_cv_sim(test_nodes, 
                           test_labelling, 
                            labels, 
                            godag, 
                            t_counts,
                            ci = ci,
                            avg = avg)
        fscores.append(fmax)
    return fscores



def score_cv_sim(test_nodes, 
                 test_labelling, 
                 real_labelling,
                 go_dag,
                 term_counts,
                 ci = 1000,
                 avg = False):
    """Scores cross validation by counting the number of test nodes that
    were accurately labeled after their removal from the true
    labelling.
    """
    def sem_similarity_(go_id, go_ids, avg = False):
        """
        If avg == True, compute the average Resnik Similarity Instead.
        """
        sims = [resnik_sim(go_id, go_i, go_dag, term_counts) for go_i in go_ids]
        if avg:
            return np.average(sims)
        return np.max(sims)
    def sem_similarity(gois_1, gois_2, avg = False):
        """
        If avg == True, use the average Resnik Similarity, provided in Pandey et. al.
        https://academic.oup.com/bioinformatics/article/24/16/i28/201569
        """
        if avg:
            sims = [sem_similarity_(g1, gois_2, avg) for g1 in gois_1]
            return np.average(sims)
        
        sims1 = [sem_similarity_(g1, gois_2) for g1 in gois_1]
        sims2 = [sem_similarity_(g2, gois_1) for g2 in gois_2]
        n_1   = len(sims1)
        n_2   = len(sims2)
        return (np.sum(sims1) + np.sum(sims2)) / float(n_1 + n_2)

    cis  = [i / ci for i in range(ci)]
    sims = []
    for c in cis:
        sim_counter  = 0
        sim          = 0
        for node in test_nodes:
            if node not in test_labelling:
                continue
            pred_labels = set([t for (t,c1) in test_labelling[node] if c1 >= c])
            true_labels = set(real_labelling[node])
            if len(true_labels) != 0:
                sim_counter += 1
            else:
                continue
            if len(pred_labels) != 0:
                sim    += sem_similarity(pred_labels, true_labels, avg = avg)
        sims.append(sim / float(sim_counter))
    return np.max(sims) 

