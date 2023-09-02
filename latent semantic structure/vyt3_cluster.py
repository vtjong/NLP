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

def parser(model_file):
    '''
    [parser(model_file)] reads [model_file], a number of latent classes, a 
    random seed (123 here), and three class names/indices from sys.argv. The 
    three class names denote the frames to print. 
    - Discard any relations beyond the following: obj, ccomp, nsubj, iobj
    - All dependencies stemming from the same head token should be collapsed 
    into a single semantic frame consisting of a head token and a list/set of 
    all the outgoing dependency labels and their associated dependents. 
        - For example:
            - nsubj(like-5, I-3)
            - obj(like-5, apples-6)
        - Would be something like:
            - like-5 [(nsubj, I-3),(obj apples-6)]
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

def init_vars(sents):
    """
    [init_vars(sents)] converts sentences [sents] into semantic frames and creates
    sets of unique heads, labels, dependents, as well as label-dependent pairs 
    that appear in frames together.
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

def heads_2_idx(fs, u_heads):
    """
    [heads_2_idx(fs, u_heads)] creates a dictionary with heads as keys and 
    a list of frames they appear in as values. 
    """
    heads_to_idx = {}
    for head in u_heads:
        heads_to_idx[head] = [i for i, x in enumerate(fs) if head == 
        x[0][:x[0].rfind("-")]]
    return heads_to_idx

def labs_2_idx(fs, num_fs, u_labs):
    """
    [labs_2_idx(fs, num_frames, uniq_labs)] creates a dictionary with labels
    as keys and a list of frames each label appear in as values; since labels
    can appear >=1 time(s) in a frame, the list of frames has repeated frame 
    indices equal to the number of times each label appears in that given frame.
    """
    labs_to_idx = {}
    for lab in u_labs:
        for idx in range(num_fs):
            for lab_dep in fs[idx][1]:
                if lab == lab_dep[0]: 
                    update_vals(labs_to_idx, lab, idx)
    return labs_to_idx

def lab_deps_2_idx(fs, num_fs, u_lab_deps):
    """
    [lab_deps_to_idx(fs, num_fs, u_lab_deps)] creates a dictionary with 
    label and dependent pairs as keys and a list of frames each pair appear in 
    as values; since label and dependent pairs can appear >=1 time(s) in a frame, 
    the list of frames has repeated frame indices equal to the number of times 
    each of these pairs appears in that given frame.
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

def init_model(fs, u_H, u_L, u_D, u_LD, seed, num_clusters):
    '''
    [init_model(frames)] calculates probabilities of initial models, CH, CL, CLD. 
    '''
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

def update_gen(p_mat, C_mat, num_clusters, *args):
    mat = dict()
    uniq, dat_to_idx = args
    for col in range(num_clusters):
        for dat in uniq:
            num = math.fsum([p_mat[row][col] for row in dat_to_idx[dat]])
            dem = C_mat[col]
            rat = num/dem     
            mat[(col, dat)] = 0.0
            if not np.isnan(rat):
                mat[(col, dat)] = rat
    return mat

def update_CH(uniq_heads, heads_to_idx, p_mat, C_mat, num_clusters):
    """
    [update_CH(CH, p_mat, frames_noidx, uniq_heads)] finds a list of indices of 
    frames each head in [uniq_heads] appears in and, for each col c_i, sums the 
    elements of [p_mat] with r_j's using the values in the list of indices of 
    frames the head is in and updates the CH matrix accordingly. 
    """
    return update_gen(p_mat, C_mat, num_clusters, uniq_heads, heads_to_idx)
    
def update_CL(uniq_labs, labs_to_idx, p_mat, C_mat, num_clusters):
    """
    [update_CL(CL, p_mat, frames_noidx, uniq_heads)] finds a list of indices of 
    frames each lab in [uniq_labs] appears in and, for each col c_i, sums the 
    elements of [p_mat] with r_j's using the values in the list of indices of 
    frames the labels are in and updates the CH matrix accordingly. 
    """
    CL = update_gen(p_mat, C_mat, num_clusters, uniq_labs, labs_to_idx)
    return CL

def update_CLD(uniq_lab_deps, lab_deps_to_idx, p_mat, uniq_labs, uniq_deps, labs_to_idx, num_clusters):
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

def calc_LL(p_mat):
    return np.sum(np.log(np.sum(p_mat, axis=1)))

def expectation(frames, p_mat, priors, CH, CL, CLD):
    """
    expectation((fs_sts, p_mat, priors, CH, CL, CLD)) carries out the expectation step 
    of the EM algorithm and returns the updated p-matrix. 
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

def maximization(p_mat, u_H, H_idx, u_L, L_idx, u_D, u_LD, LD_idx):
    """
    maximization(p_mat) carries out the expectation step of the EM algorithm and 
    returns the updated p-matrix. 
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
        
def EM(frames, p_mat, CH, CL, CLD, u_H, H_idx, u_L, L_idx, u_D, u_LD, LD_idx):
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

def best_heads(CH, num_clusters):
    f = open('best.heads', 'w')
    for c_i in range(num_clusters):
        temp = dict()
        for (idx, head) in CH.keys():
            if idx == c_i: temp[head] = CH[(idx, head)]
        sort_temp = sorted(temp.items(), key=lambda y: y[1], reverse=True)
        sort_temp = sort_temp[:15]
        
        for i in range(15):
            f.write(str(c_i) + ' : ' + sort_temp[i][0] + ' ' + str(sort_temp[i][1]) + '\n')
    f.close()

def best_frames(p_mat, fs, classes_to_output):
    num_out = 10
    f = open('best.frames', 'w')
    for c in classes_to_output:
        best_frames = (-p_mat[:, c]).argsort()[:num_out]
        for i in range(num_out):
            f_idx = best_frames[i]
            str_f_info = str()
            f_info = fs[f_idx][1]
            for idx, (lab, dep) in enumerate(f_info):
                if len(f_info) == 1:
                    dep_noidx = dep[: dep.rfind("-")]
                    str_f_info += "'" + lab + " " + dep_noidx + "'"
                    break
                elif len(f_info) > 1 and idx < len(f_info) - 1:
                    dep_noidx = dep[: dep.rfind("-")]
                    str_f_info += "'" + lab + " " + dep_noidx + "'"
                    str_f_info += ", " 
            f.write(str(c) + ' ' + str(p_mat[f_idx][c]) + ' : ' + fs[f_idx][0] + 
            " [" + str_f_info + "]" + '\n')
    f.close()

def save_it_up(filename, CH, CL, CLD):
    file = open(filename, "w", encoding="utf-8")
    for (col,head) in CH:    
        file.write("CH " + str(col) + " : " + head + " " + 
            str(CH[(col, head)]) + "\n")
    
    for (col, lab) in CL:
        file.write("CL " + str(col) + " : " + lab + " " + 
            str(CL[(col, lab)]) + "\n")
    
    for (col, lab, dep) in CLD:
        file.write("CLD " + str(col) + " " + lab + " : " + 
            str(dep) + " " + str(CLD[(col, lab, dep)]) + "\n")

model_file, num_clusters, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
classes_to_output = list(map(int, sys.argv[4:]))
fs, u_H, u_L, u_D, u_LD = parser(model_file)
p_mat, priors, CH, CL, CLD, H_idx, L_idx, LD_idx = init_model(fs, u_H, u_L, u_D, u_LD, seed, num_clusters)
p_mat = expectation(fs, p_mat, priors, CH, CL, CLD)
ll = calc_LL(p_mat)
p_mat, CH, CL, CLD = EM(fs, p_mat, CH, CL, CLD, u_H, H_idx, u_L, L_idx, u_D, u_LD, LD_idx)
best_heads(CH, num_clusters)
best_frames(p_mat, fs, classes_to_output)
