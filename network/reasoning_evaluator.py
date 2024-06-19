import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_
import copy
import random
import logging
from collections import defaultdict

from embedding import RotatE

class Reasoning_Evaluator(torch.nn.Module):
    def __init__(self):
        super(Reasoning_Evaluator, self).__init__()

        self.num_entities = self.entity_size
        self.num_relations = self.relation_size

    def set_rules(self, input):
        self.rules = list()
        if type(input) == list:
            for rule in input:
                rule_ = (rule[0], rule[1:])
                self.rules.append(rule_)
            logging.info('Evaluator: read {} rules from list.'.format(len(self.rules)))
        elif type(input) == str:
            with open(input, 'r') as fi:
                for line in fi:
                    rule = line.strip().split()
                    rule = [int(_) for _ in rule]
                    rule_ = (rule[0], rule[1:])
                    self.rules.append(rule_)
            logging.info('Evaluator: read {} rules from file.'.format(len(self.rules)))
        else:
            raise ValueError
        self.num_rules = len(self.rules)

        self.relation2rules = [[] for r in range(self.num_relations)]
        for index, rule in enumerate(self.rules):
            relation = rule[0]
            self.relation2rules[relation].append([index, rule])
        
        self.rule_weights = torch.nn.parameter.Parameter(torch.zeros(self.num_rules))

    def forward(self, all_h, all_r, edges_to_remove):
        query_r = all_r[0].item()
        assert (all_r != query_r).sum() == 0
        device = all_r.device

        score = torch.zeros(all_r.size(0), self.num_entities, device=device)
        mask = torch.zeros(all_r.size(0), self.num_entities, device=device)
        for index, (r_head, r_body) in self.relation2rules[query_r]:
            assert r_head == query_r

            x = self.graph.grounding(all_h, r_head, r_body, edges_to_remove)
            score += x * self.rule_weights[index]
            mask += x
        
        if mask.sum().item() == 0:
            if self.entity_feature == 'bias':
                return mask + self.bias.unsqueeze(0), (1 - mask).bool()
            else:
                return mask - float('-inf'), mask.bool()
        
        if self.entity_feature == 'bias':
            score = score + self.bias.unsqueeze(0)
            mask = torch.ones_like(mask).bool()
        else:
            mask = (mask != 0)
            score = score.masked_fill(~mask, float('-inf'))
        
        return score, mask

    def compute_score(self, all_h, all_r, all_t, edges_to_remove):
        query_r = all_r[0].item()
        assert (all_r != query_r).sum() == 0
        device = all_r.device

        rule_score = list()
        rule_index = list()
        mask = torch.zeros(all_r.size(0), self.num_entities, device=device)
        for index, (r_head, r_body) in self.relation2rules[query_r]:
            assert r_head == query_r

            x = self.graph.grounding(all_h, r_head, r_body, edges_to_remove)
            score = x * self.rule_weights[index]
            mask += x

            rule_score.append(score)
            rule_index.append(index)

        rule_index = torch.tensor(rule_index, dtype=torch.long, device=device)
        pos_index = F.one_hot(all_t, self.num_entities).bool()
        if device.type == "cuda":
            pos_index = pos_index.cuda(device)
        neg_index = (mask != 0)

        if len(rule_score) == 0:
            return None, None

        rule_score = list()
        for score in rule_score:
            pos_score = (score * pos_index).sum(1) / torch.clamp(pos_index.sum(1), min=1)
            neg_score = (score * neg_index).sum(1) / torch.clamp(neg_index.sum(1), min=1)
            H_score = pos_score - neg_score
            rule_score.append(H_score.unsqueeze(-1))

        rule_score = torch.cat(rule_score, dim=-1)
        rule_score = torch.softmax(rule_score, dim=-1).sum(0)

        return rule_score, rule_index
