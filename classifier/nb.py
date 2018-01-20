import utils
import numpy as np

from collections import defaultdict

CLASS = ['1st', '2nd', '3rd', 'crew']
AGE = ['adult', 'child']
GENDER = ['female', 'male']
SURVIVED = ['yes', 'no']

DATA_PATH = '../data/titanic.txt'
joint_probabilities = {}
conditional_probabilities = {}
nb_cond_probabilities = defaultdict(dict)
nb_probabilities = {}


def format_data(data):
    formatted_data = []
    for feature in data:
        class_w, age, gender, survived = feature
        formatted_data.append([CLASS.index(class_w), AGE.index(age),
                               GENDER.index(gender), SURVIVED.index(survived)])

    return formatted_data


def get_count_data(search_comb, data):
    count = 0
    for x in data:

        if search_comb == x:
            count += 1

    return count


def get_joint_distribution_probabilities(data, total, joint_probs):
    for i in range(len(CLASS)):
        for j in range(len(AGE)):
            for k in range(len(GENDER)):
                for l in range(len(SURVIVED)):
                    joint_probs[i, j, k, l] = get_count_data([i, j, k, l], data) / total

    return joint_probs


def get_count_prob(pattern, prob):
    sum_prob = 0
    for key in prob:
        if list(key) == pattern:
            sum_prob += prob[key]
        elif pattern[3] == 'x' and pattern[:3] == list(key[:3]):
            sum_prob += prob[key]

    return sum_prob


def get_count_survived(data):
    count_surv = 0
    count_dead = 0
    for line in data:
        if line[3] == 0:
            count_surv += 1
        else:
            count_dead += 1

    return count_surv, count_dead


def get_prior_survived(data, total):
    count_surv, count_dead = get_count_survived(data)
    return count_surv/total, count_dead/total


def get_joint_prob_nb(data, feature_index, index, total):
    count_feature_d_joint = 0
    count_feature_s_joint = 0

    for line in data:
        if line[feature_index] == index:
            if line[3] == 0:
                count_feature_s_joint += 1
            else:
                count_feature_d_joint += 1

    return count_feature_s_joint/total, count_feature_d_joint/total


def get_conditional_probabilities(joint_prob, conditional_probs):
    for i in range(len(CLASS)):
        for j in range(len(AGE)):
            for k in range(len(GENDER)):
                denom = get_count_prob([i, j, k, 'x'], joint_prob)
                if denom > 0:
                    conditional_probs[i, j, k, 1] = get_count_prob([i, j, k, 1], joint_prob) / denom
                else:
                    conditional_probs[i, j, k, 1] = 0

    return conditional_probs


def get_nb_cond_prob(data, total, nb_cond_prob, nb_prob):
    prior_surv, prior_dead = get_prior_survived(data, total)

    for i in range(len(CLASS)):
        class_s_joint, class_d_join = get_joint_prob_nb(data, 0, i, total)
        nb_cond_prob[0][(i, 0)] = class_s_joint/prior_surv
        nb_cond_prob[0][(i, 1)] = class_d_join/prior_dead

    for i in range(len(AGE)):
        age_s_joint, age_d_join = get_joint_prob_nb(data, 1, i, total)
        nb_cond_prob[1][(i, 0)] = age_s_joint / prior_surv
        nb_cond_prob[1][(i, 1)] = age_d_join / prior_dead

    for i in range(len(GENDER)):
        gender_s_joint, gender_d_join = get_joint_prob_nb(data, 2, i, total)
        nb_cond_prob[2][(i, 0)] = gender_s_joint / prior_surv
        nb_cond_prob[2][(i, 1)] = gender_d_join / prior_dead

    for i in range(len(CLASS)):
        for j in range(len(AGE)):
            for k in range(len(GENDER)):
                survival_prob = nb_cond_prob[0][(i, 0)] * nb_cond_prob[1][(j, 0)] * nb_cond_prob[2][(k, 0)] * prior_surv
                death_prob = nb_cond_prob[0][(i, 1)] * nb_cond_prob[1][(j, 1)] * nb_cond_prob[2][(k, 1)] * prior_dead
                denom = survival_prob + death_prob
                if denom > 0:
                    nb_prob[i, j, k, 1] = death_prob / denom
                else:
                    nb_prob[i, j, k, 1] = 0

    return nb_prob


if __name__ == "__main__":
    data = utils.read_file_in_list(DATA_PATH)
    data = format_data(data)
    total_data = len(data)
    joint_prob = get_joint_distribution_probabilities(data, total_data, joint_probabilities)
    conditional_prob = get_conditional_probabilities(joint_prob, conditional_probabilities)
    nb_prob = get_nb_cond_prob(data, total_data, nb_cond_probabilities, nb_probabilities)
