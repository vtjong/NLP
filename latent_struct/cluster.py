from typing import Set, Dict, List, Tuple
import sys
import numpy as np
import math
from numpy import random

def update_counts(*args):
    dict, key, count = args
    dict.setdefault(key, 0.0)
    dict[key] += count

def update_vals(*args):
    dict, key, count = args
    dict.setdefault(key, [])
    dict[key].append(count)

def parser(model_file: str) -> Tuple[List[Tuple[str, List[Tuple[str, str]]]],
                                    List[str], List[str], List[str], List[Tuple[str, str]]]:
    '''
    Parses a model file and generates semantic frames for specific dependency 
    relations.

    This function reads the `model_file` containing dependency relations and 
    processes them to create collapsed semantic frames based on specified 
    dependency relations. It returns the semantic frames and related data.

    Parameters:
    - model_file (str): The path to the model file containing dependency relations.

    Returns:
    - fs (tuple): Collapsed semantic frames per sentence. Each frame is represented 
      as a tuple containing a head token and a list of outgoing dependency labels 
      and their associated dependents.
    - u_heads (list): List of unique head tokens.
    - u_labs (list): List of unique dependency labels.
    - u_deps (list): List of unique dependent tokens.
    - u_lab_deps (list): List of unique pairs of dependency labels and their 
      associated dependent tokens.
    '''
    # Read in raw sentences to list [raw_sents]
    with open(model_file) as file:
        lines = file.read().strip().split("\n") 
    line_end = [i for i, x in enumerate(lines) if x == ''] + [len(lines)]
    line_beg = [0] + [idx+1 for idx in line_end][:-1]
    raw_sents = [lines[b:e] for b,e in list(zip(line_beg, line_end))]

    # Process sentences to get rid of unwanted relations
    relations = {'obj', 'ccomp', 'nsubj', 'iobj'}
    sents = [[word for word in sentence if word[:word.find('(')] in relations]
    for sentence in raw_sents if not []]

    # Create collapsed semantic frames per sentence
    fs, u_heads, u_labs, u_deps, u_lab_deps = init_vars(sents)
    fs = tuple((k, tuple(v)) for d in fs for (k, v) in d.items())

    # Convert to lists for easier iteration
    u_heads, u_labs = list(u_heads), list(u_labs)
    u_deps, u_lab_deps = list(u_deps), list(u_lab_deps)
    return fs, u_heads, u_labs, u_deps, u_lab_deps

def init_vars(
    sents: List[List[str]]
)-> Tuple[List[Dict[str, List[Tuple[str, str]]]],
    Set[str], Set[str], Set[str], Set[Tuple[str, str]]]:
    """
    Initializes variables and converts sentences into semantic frames.

    This function takes a list of sentences `sents`, where each sentence is 
    represented as a list of dependency relations in Stanford Dependency format. 
    It processes the sentences to create semantic frames and also creates sets 
    of unique heads, labels, dependents, and label-dependent pairs that
    appear in frames together.

    Parameters:
    - sents (list): A list of sentences, where each sentence is represented as a 
      list of dependency relations in Stanford Dependency format.

    Returns:
    - fs (list): A list of semantic frames, where each frame is represented as a 
      dictionary containing head tokens as keys and lists of outgoing dependency 
      labels and their associated dependents as values.
    - u_heads (set): A set of unique head tokens.
    - u_labs (set): A set of unique dependency labels.
    - u_deps (set): A set of unique dependent tokens.
    - u_lab_deps (set): A set of unique pairs of dependency labels and their \
      associated dependent tokens.
    """
    fs= []
    u_heads, u_labs, u_deps, u_lab_deps = set(), set(), set(), set()
    for sentence in sents:
        frame, s_heads = dict(), set()
        for word in sentence:
            head_beg, head_end = word.find("("), word.find(",")
            head = word[head_beg+1:head_end]
            head_noidx = head[: head.rfind("-")]
            label, dep = word[:head_beg], word[head_end+1:-1].strip()
            dep_noidx = dep[: dep.rfind("-")]
            lab_dep = (label,dep)
            lap_dep_nodix = (label, dep_noidx)
            update_vals(frame, head, lab_dep)

            # Add to head, lab, dep sets
            u_lab_deps.add(lap_dep_nodix)
            u_heads.add(head_noidx)
            u_labs.add(label), u_deps.add(dep_noidx)
            s_heads.add(head)
        fs.append(frame)
    return fs, u_heads, u_labs, u_deps, u_lab_deps

def heads_2_idx(
    fs: List[Dict[str, List[Tuple[str, str]]]],
    u_heads: Set[str]
) -> Dict[str, List[int]]:
    """
    Creates a dictionary with heads as keys and a list of frames they appear 
    in as values.

    This function takes a list of semantic frames `fs` and a set of unique 
    head tokens `u_heads`.
    It creates a dictionary where head tokens are the keys, 
    and the values are lists of frame indices in which each head token appears.

    Parameters:
    - fs (list): A list of semantic frames, where each frame is represented as 
      a tuple containing a head token and a list of outgoing dependency labels 
      and their associated dependents.
    - u_heads (set): A set of unique head tokens.

    Returns:
    - heads_to_idx (dict): A dictionary where head tokens are keys, and 
      the values are lists of frame indices.

    Example Usage:
    ```
    frames = [('like-5', [('nsubj', 'I-3'), ('obj', 'apples-6')]), 
    ('eat-2', [('nsubj', 'She-1'), ('obj', 'pizza-3')])]
    unique_heads = {'like-5', 'eat-2'}

    heads_indices = heads_2_idx(frames, unique_heads)
    ```
    This example creates a dictionary mapping head tokens to the frames they appear in.
    """
    heads_to_idx = {}
    for head in u_heads:
        heads_to_idx[head] = [i for i, frame in enumerate(fs) if head ==
        frame[0][: frame[0].rfind("-")]]
    return heads_to_idx

def labs_2_idx(
    fs: List[Dict[str, List[Tuple[str, str]]]],
    num_fs: int,
    u_labs: Set[str]
) -> Dict[str, List[int]]:
    """
    Creates a dictionary with labels as keys and a list of frames each label 
    appears in as values.

    This function takes a list of semantic frames `fs`, the total number of 
    frames `num_fs`, and a set of unique dependency labels `u_labs`. It creates 
    a dictionary where labels are the keys, and the values are lists of frame 
    indices in which each label appears. Since labels can appear multiple times 
    in a frame, the list of frame indices may contain repeated indices corresponding 
    to the number of times a label appears in a frame.

    Parameters:
    - fs (list): A list of semantic frames, where each frame is represented as 
      a tuple containing a head token and a list of outgoing dependency labels 
      and their associated dependents.
    - num_fs (int): The total number of frames.
    - u_labs (set): A set of unique dependency labels.

    Returns:
    - labs_to_idx (dict): A dictionary where labels are keys, and the values are 
      lists of frame indices.
    """
    labs_to_idx = {}
    for lab in u_labs:
        for idx in range(num_fs):
            for lab_dep in fs[idx][1]:
                if lab == lab_dep[0]: 
                    update_vals(labs_to_idx, lab, idx)
    return labs_to_idx

def lab_deps_2_idx(
    fs: List[Dict[str, List[Tuple[str, str]]]],
    num_fs: int,
    u_lab_deps: Set[Tuple[str, str]]
) -> Dict[Tuple[str, str], List[int]]:
    """
    Parameters:
    - fs (list): A list of semantic frames, where each frame is represented as a 
      tuple containing a head token and a list of outgoing dependency labels and 
      their associated dependents.
    - num_fs (int): The total number of frames.
    - u_lab_deps (set): A set of unique pairs of dependency labels and their 
      associated dependent tokens.

    Returns:
    - lab_deps_to_idx (dict): A dictionary where label and dependent pairs are 
      keys, and the values are lists of frame indices.
    """
    lab_deps_to_idx = {}
    for u_lab_dep in u_lab_deps:
        for idx in range(num_fs):
            for lab_dep in fs[idx][1]:
                lab, dep = lab_dep[0], lab_dep[1]
                dep = dep[: dep.rfind("-")]
                lab_dep = (lab,dep)
                if u_lab_dep == lab_dep: 
                    update_vals(lab_deps_to_idx, lab_dep, idx)
    return lab_deps_to_idx

def init_model(
    fs: List[Dict[str, List[Tuple[str, str]]]],
    u_H: List[str],
    u_L: List[str],
    u_D: List[str],
    u_LD: List[Tuple[str, str]],
    seed: int,
    num_clusters: int
) -> Tuple[
    np.ndarray, np.ndarray,
    Dict[Tuple[int, str], float], Dict[Tuple[int, str], float],
    Dict[Tuple[int, str, str], float],
    Dict[str, int], Dict[str, List[int]], Dict[Tuple[str, str], List[int]]
]:
    """
    Calculates initial model probabilities for frames, CH, CL, and CLD.

    This function takes the following inputs:
    - fs (list): A list of frames.
    - u_H (set): A set of unique head tokens.
    - u_L (set): A set of unique dependency labels.
    - u_D (set): A set of unique dependent tokens.
    - u_LD (set): A set of unique label-dependent pairs.
    - seed (int): A random seed for reproducibility.
    - num_clusters (int): The number of clusters.

    Returns:
    - p_mat (ndarray): An initial probability matrix.
    - priors (ndarray): Initial priors.
    - CH (dict): An initial CH matrix.
    - CL (dict): An initial CL matrix.
    - CLD (dict): An initial CLD matrix.
    - H_idx (dict): A dictionary mapping head tokens to frame indices.
    - L_idx (dict): A dictionary mapping labels to frame indices.
    - LD_idx (dict): A dictionary mapping label-dependent pairs to frame indices.
    """

    frames = fs
    num_frames = len(frames)
    random.seed(seed) 
    p_mat = random.random_sample((num_frames, num_clusters))
    p_mat /= p_mat.sum(axis=1)[:, np.newaxis]
    C_mat = np.sum(p_mat, axis=0, dtype=np.float64)
    
    # CH init
    H_idx = heads_2_idx(frames, u_H)
    CH = update_CH(u_H, H_idx, p_mat, C_mat, num_clusters)

    # CL init (can be repeated so need to add idx the number of times this 
    # shows up)
    L_idx = labs_2_idx(frames, num_frames, u_L)
    CL = update_CL(u_L, L_idx, p_mat, C_mat, num_clusters)

    # CLD init  
    LD_idx = lab_deps_2_idx(frames, num_frames, u_LD)
    CLD = update_CLD(u_LD, LD_idx, p_mat, u_L, u_D, L_idx, num_clusters)

    priors = C_mat / np.sum(C_mat) if np.sum(C_mat) != 0 else C_mat
    return p_mat, priors, CH, CL, CLD, H_idx, L_idx, LD_idx

def update_gen(
    p_mat: np.ndarray,
    C_mat: np.ndarray,
    num_clusters: int,
    uniq: List[str],
    dat_to_idx: Dict[str, List[int]]
) -> Dict[Tuple[int, str], float]:
    """
    Computes and updates a matrix with information related to unique data elements.

    This function takes the following inputs:
    - p_mat (np.ndarray): A probability matrix.
    - C_mat (np.ndarray): A matrix that stores clustering information.
    - num_clusters (int): The number of clusters.
    - args (tuple): A variable number of arguments containing unique 
    data elements and their associated indices.

    It computes and updates a matrix using the provided information.

    Returns:
    - mat (dict): An updated matrix where keys represent (cluster, data) pairs, 
    and values are computed ratios.

    Example Usage:
    ```
    probability_matrix = [[0.1, 0.2], [0.3, 0.4]]
    clustering_matrix = [1, 1]
    num_clusters = 2
    unique_data = {'like-5', 'eat-2'}
    data_indices = {'like-5': [0], 'eat-2': [1]}

    updated_matrix = update_gen(probability_matrix, clustering_matrix, 
                        num_clusters, unique_data, data_indices)
    ```
    This example computes and updates a matrix based on unique data elements.
    """
    mat = dict()
    for col in range(num_clusters):
        for dat in uniq:
            num = math.fsum([p_mat[row][col] for row in dat_to_idx[dat]])
            dem = C_mat[col]
            rat = num/dem     
            mat[(col, dat)] = 0.0
            if not np.isnan(rat):
                mat[(col, dat)] = rat
    return mat

def update_CH(
    uniq_heads: List[str],
    heads_to_idx: Dict[str, List[int]],
    p_mat: np.ndarray,
    C_mat: np.ndarray,
    num_clusters: int
) -> Dict[Tuple[int, str], float]:
    """
    Updates the CH matrix with information related to unique heads.

    This function takes the following inputs:
    - uniq_heads (set): A set of unique head tokens.
    - heads_to_idx (dict): A dictionary mapping head tokens to the list of 
      frame indices where each head appears.
    - p_mat (np.ndarray): A probability matrix.
    - C_mat (np.ndarray): A matrix that stores clustering information.
    - num_clusters (int): The number of clusters.

    It computes and updates the CH matrix using the provided information.

    Returns:
    - CH (dict): An updated CH matrix.

    Example Usage:
    ```
    unique_heads = {'like-5', 'eat-2'}
    heads_indices = {'like-5': [0], 'eat-2': [1]}
    probability_matrix = [[0.1, 0.2], [0.3, 0.4]]
    clustering_matrix = [[1, 0], [0, 1]]
    num_clusters = 2

    updated_CH = update_CH(unique_heads, heads_indices, probability_matrix, 
    clustering_matrix, num_clusters)
    ```
    This example updates the CH matrix based on unique head tokens.
    """
    return update_gen(p_mat, C_mat, num_clusters, uniq_heads, heads_to_idx)

    
def update_CL(
    uniq_labs: List[str],
    labs_to_idx: Dict[str, List[int]],
    p_mat: np.ndarray,
    C_mat: np.ndarray,
    num_clusters: int
) -> Dict[Tuple[int, str], float]:
    """
    Updates the CL matrix with information related to unique labels.

    This function takes the following inputs:
    - uniq_labs (set): A set of unique dependency labels.
    - labs_to_idx (dict): A dictionary mapping labels to the list of frame indices 
      where each label appears.
    - p_mat (np.ndarray): A probability matrix.
    - C_mat (np.ndarray): A matrix that stores clustering information.
    - num_clusters (int): The number of clusters.

    It computes and updates the CL matrix using the provided information.

    Returns:
    - CL (dict): An updated CL matrix.
    """
    CL = update_gen(p_mat, C_mat, num_clusters, uniq_labs, labs_to_idx)
    return CL

def update_CLD(
    uniq_lab_deps: Set[Tuple[str, str]],
    lab_deps_to_idx: Dict[Tuple[str, str], List[int]],
    p_mat: np.ndarray,
    uniq_labs: Set[str],
    uniq_deps: Set[str],
    labs_to_idx: Dict[str, List[int]],
    num_clusters: int
) -> Dict[Tuple[int, str, str], float]:

    """
    Computes and updates the CLD matrix with information related to unique 
    label-dependent pairs.

    This function takes the following inputs:
    - uniq_lab_deps (set): A set of unique pairs of dependency labels and their 
      associated dependent tokens.
    - lab_deps_to_idx (dict): A dictionary mapping pairs to the list of frame 
      indices where each pair appears.
    - p_mat (np.ndarray): A probability matrix.
    - uniq_labs (set): A set of unique dependency labels.
    - uniq_deps (set): A set of unique dependent tokens.
    - labs_to_idx (dict): A dictionary mapping labels to the list of frame 
      indices where each label appears.
    - num_clusters (int): The number of clusters.

    Returns:
    - CLD (dict): An updated CLD matrix.
    """
    # Init CLD
    CLD = {}
    for col in range(num_clusters):
        for dep in uniq_deps:
            for lab in uniq_labs:
                update_counts(CLD, (col, lab, dep), 0.0) 
    # Calculate denominators for cld
    cld_denom = {}
    for col in range(num_clusters):
        cld_denom[col] = {}
        for lab in labs_to_idx:
            cld_denom[col][lab] = math.fsum([p_mat[idx][col] for idx in 
            labs_to_idx[lab]])
    # CLD step
    for col in range(num_clusters):
        for lab_dep in uniq_lab_deps:
            num = math.fsum([p_mat[row][col] for row in lab_deps_to_idx[lab_dep]])
            lab, dep = lab_dep[0], lab_dep[1]
            dem = cld_denom[col][lab]
            try: 
                rat = num/dem  
            except ZeroDivisionError:
                rat = 0.0
            if not np.isnan(rat):
                CLD[(col,lab, dep)] = rat   
    return CLD

def calc_LL(p_mat: np.ndarray) -> float:
    """
    Calculates the log likelihood of the probability matrix.

    Args:
        p_mat (numpy.ndarray): The probability matrix.

    Returns:
        float: The log likelihood of the probability matrix.
    """
    return np.sum(np.log(np.sum(p_mat, axis=1)))

def expectation(
    frames: List[Tuple[str, List[Tuple[str, str]]]],
    p_mat: np.ndarray,
    priors: np.ndarray,
    CH: Dict[Tuple[int, str], float],
    CL: Dict[Tuple[int, str], float],
    CLD: Dict[Tuple[int, str, str], float]
) -> np.ndarray:
    """
    Performs the expectation step of the EM algorithm and returns the updated 
    p-matrix.

    Args:
    - frames(list): A list of frames, where each frame is represented as a tuple 
      containing a head token and a list of 
      label-dependent pairs.
    - p_mat (np.ndarray): The probability matrix.
    - priors (np.ndarray): The prior probabilities.
    - CH (dict): The CH matrix.
    - CL (dict): The CL matrix.
    - CLD (dict): The CLD matrix.

    Returns:
    - np.ndarray: The updated probability matrix.
    """
    new_p_mat = np.zeros_like(p_mat)
    n_frames, n_clusters = np.shape(p_mat) 
    for frame_idx, frame in enumerate(frames):
        # Need to get the info. out from the frame_idx, all heads, all POS, all deps 
        # new_p_mat[col][frame_idx] = np.sum(p_mat[:, col])*CH[uniq_to_idx[dat]]
        head = frame[0] 
        head = head[: head.rfind("-")]
        lab_deps = frame[1]
        for col in range(n_clusters):
            temp = priors[col]*CH[(col, head)]
            for lab, dep in lab_deps:
                temp *= CL[(col, lab)]
                dep_noidx = dep[: dep.rfind("-")]
                temp *= CLD[(col, lab, dep_noidx)]
            new_p_mat[frame_idx][col] = temp
    return new_p_mat

def maximization(
    p_mat: np.ndarray,
    u_H: Set[str],
    H_idx: Dict[str, List[int]],
    u_L: Set[str],
    L_idx: Dict[str, List[int]],
    u_D: Set[str],
    u_LD: Set[Tuple[str, str]],
    LD_idx: Dict[Tuple[str, str], List[int]]
) -> Tuple[
    np.ndarray,
    np.ndarray,
    Dict[Tuple[int, str], float],
    Dict[Tuple[int, str], float],
    Dict[Tuple[int, str, str], float]
]:
    """
    Performs the maximization step of the EM algorithm and returns the updated 
    p-matrix.

    Args:
    - p_mat (ndarray): The probability matrix.
    - u_H (set): A set of unique head tokens.
    - H_idx (dict): A dictionary mapping head tokens to frame indices.
    - u_L (set): A set of unique dependency labels.
    - L_idx (dict): A dictionary mapping labels to frame indices.
    - u_D (set): A set of unique dependent tokens.
    - u_LD (set): A set of unique label-dependent pairs.
    - LD_idx (dict): A dictionary mapping label-dependent pairs to frame indices.

    Returns:
    - p_mat (ndarray): The updated probability matrix.
    - priors (ndarray): Updated priors.
    - CH (dict): Updated CH matrix.
    - CL (dict): Updated CL matrix.
    - CLD (dict): Updated CLD matrix.
    """
    num_frames, num_clusters = np.shape(p_mat)
    row_sum_p_mat = p_mat.sum(axis=1)[:, np.newaxis]
    p_mat /= row_sum_p_mat
    C_mat = np.sum(p_mat, axis=0)
    
    CH = update_CH(u_H, H_idx, p_mat, C_mat, num_clusters)
    CL = update_CL(u_L, L_idx, p_mat, C_mat, num_clusters)
    CLD = update_CLD(u_LD, LD_idx, p_mat, u_L, u_D, L_idx, num_clusters)

    priors = C_mat / np.sum(C_mat) if np.sum(C_mat) != 0 else C_mat
    return p_mat, priors, CH, CL, CLD
        
def EM(
    frames: List[Tuple[str, List[Tuple[str, str]]]],
    p_mat: List[List[float]],
    CH: Dict[Tuple[int, str], float],
    CL: Dict[Tuple[int, str], float],
    CLD: Dict[Tuple[int, str, str], float],
    u_H: Set[str],
    H_idx: Dict[str, List[int]],
    u_L: Set[str],
    L_idx: Dict[str, List[int]],
    u_D: Set[str],
    u_LD: Set[Tuple[str, str]],
    LD_idx: Dict[Tuple[str, str], List[int]]
) -> Tuple[
    List[List[float]],
    Dict[Tuple[int, str], float],
    Dict[Tuple[int, str], float],
    Dict[Tuple[int, str, str], float]
]:
    """
    Performs the EM algorithm, updating matrices iteratively until convergence 
    and saving the model.

    Args:
    - frames (list): A list of frames.
    - p_mat (ndarray): The probability matrix.
    - CH (dict): The CH matrix.
    - CL (dict): The CL matrix.
    - CLD (dict): The CLD matrix.
    - u_H (set): A set of unique head tokens.
    - H_idx (dict): A dictionary mapping head tokens to frame indices.
    - u_L (set): A set of unique dependency labels.
    - L_idx (dict): A dictionary mapping labels to frame indices.
    - u_D (set): A set of unique dependent tokens.
    - u_LD (set): A set of unique label-dependent pairs.
    - LD_idx (dict): A dictionary mapping label-dependent pairs to frame indices.

    Returns:
    - new_p_mat (ndarray): The updated probability matrix.
    - CH (dict): Updated CH matrix.
    - CL (dict): Updated CL matrix.
    - CLD (dict): Updated CLD matrix.
    """

    LLs = [calc_LL(p_mat)]
    num_iter = len(LLs)
    while num_iter < 4 or LLs[num_iter-1] - LLs[num_iter-4] > 0.1:
        print("Iteration: ", num_iter)
        p_mat, priors, CH, CL, CLD = maximization(p_mat, u_H, H_idx, u_L, L_idx, u_D, u_LD, LD_idx)
        p_mat = expectation(frames, p_mat, priors, CH, CL, CLD)
        LLs.append(calc_LL(p_mat))
        num_iter = len(LLs)
    save_it_up("model.sem", CH, CL, CLD)
    return p_mat, CH, CL, CLD

def best_heads(
    CH: Dict[Tuple[int, str], float],
    num_clusters: int
) -> None:
    """
    Writes the top head tokens for each cluster to the 'best.heads' file.

    Args:
    - CH (Dict[Tuple[int, str], float]): The CH matrix.
    - num_clusters (int): The number of clusters.

    Returns:
    None
    """
    with open('best.heads', 'w') as f:
        for c_i in range(num_clusters):
            temp = {head: value for (idx, head), value in CH.items() if idx == c_i}
            sorted_temp = sorted(temp.items(), key=lambda item: item[1], reverse=True)[:15]
            
            for head, value in sorted_temp:
                f.write(f"{c_i} : {head} {value}\n")

def best_frames(
    p_mat: np.ndarray,
    fs: List[Tuple[str, List[Tuple[str, str]]]],
    classes_to_output: List[int]
) -> None:
    """
    Writes the top frames for each class to the 'best.frames' file.

    Args:
    - p_mat (np.ndarray): The probability matrix.
    - fs (List[Tuple[str, List[Tuple[str, str]]]]): List of frames and their info.
    - classes_to_output (List[int]): List of class indices to output.

    Returns:
    None
    """
    num_out = 10
    with open('best.frames', 'w') as f:
        for c in classes_to_output:
            best_frames = (-p_mat[:, c]).argsort()[:num_out]
            for i in range(num_out):
                f_idx = best_frames[i]
                f_info = fs[f_idx][1]
                f_info_str = ", ".join([f"'{lab} {dep[:dep.rfind('-')]}'" for lab, dep in f_info])
                f.write(f"{c} {p_mat[f_idx][c]} : {fs[f_idx][0]} [{f_info_str}]\n")

def save_it_up(
    filename: str,
    CH: Dict[Tuple[int, str], float],
    CL: Dict[Tuple[int, str], float],
    CLD: Dict[Tuple[int, str, str], float]
) -> None:
    """
    Saves the CH, CL, and CLD matrices to a file.

    Args:
    - filename (str): The name of the output file.
    - CH (dict): The CH matrix.
    - CL (dict]): The CL matrix.
    - CLD (dict): The CLD matrix.

    Returns:
    None
    """

    with open(filename, "w", encoding="utf-8") as file:
        for (col, head) in CH:
            file.write(f"CH {col} : {head} {CH[(col, head)]}\n")

        for (col, lab) in CL:
            file.write(f"CL {col} : {lab} {CL[(col, lab)]}\n")

        for (col, lab, dep) in CLD:
            file.write(f"CLD {col} {lab} : {dep} {CLD[(col, lab, dep)]}\n")

model_file, num_clusters, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
classes_to_output = list(map(int, sys.argv[4:]))
fs, u_H, u_L, u_D, u_LD = parser(model_file)
p_mat, priors, CH, CL, CLD, H_idx, L_idx, LD_idx = init_model(fs, u_H, u_L, u_D, u_LD, seed, num_clusters)
p_mat = expectation(fs, p_mat, priors, CH, CL, CLD)
ll = calc_LL(p_mat)
p_mat, CH, CL, CLD = EM(fs, p_mat, CH, CL, CLD, u_H, H_idx, u_L, L_idx, u_D, u_LD, LD_idx)
best_heads(CH, num_clusters)
best_frames(p_mat, fs, classes_to_output)
