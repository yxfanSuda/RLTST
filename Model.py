import torch
import numpy as np
from utils import *
from tqdm import tqdm
from transformers import  AdamW
from collections import defaultdict
from TST_Learning import AdamWTSTLearning
from sklearn import metrics
import torch.nn as nn
import torch.nn.functional as F
class BaseNetwork(nn.Module):
    def __init__(self, pretrained_model = None):
        super(BaseNetwork, self).__init__()
        self.pretrained_model = pretrained_model

    def forward(self, texts, input_mask, segment_ids):
        output = self.pretrained_model(texts,input_mask, segment_ids)
        return output

class ParsingTask(nn.Module):
    def __init__(self, params):
        super(ParsingTask, self).__init__()
        self.params = params
        self.link_classifier = Classifier(params.path_hidden_size * 2, params.path_hidden_size,
                                          1)
        self.label_classifier = Classifier(params.path_hidden_size * 2,
                                           params.path_hidden_size,
                                           params.relation_type_num)

    def forward(self, predicted_path, batch_size, node_num):
        return self.link_classifier(predicted_path).reshape(batch_size, node_num, node_num), \
               self.label_classifier(predicted_path)

class ARTask(nn.Module):
    def __init__(self, params):
        super(ARTask, self).__init__()
        self.params = params
        self.link_classifier = Classifier(params.path_hidden_size * 2, params.path_hidden_size,
                                          1)
        self.label_classifier = Classifier(params.path_hidden_size * 2,
                                           params.path_hidden_size,
                                           params.relation_type_num)

    def forward(self, predicted_path, batch_size, node_num):
        return self.link_classifier(predicted_path).reshape(batch_size, node_num, node_num), \
               self.label_classifier(predicted_path)

class TaskSpecificNetwork1(nn.Module):
    def __init__(self, params, pretained_model ):
        super(TaskSpecificNetwork1, self).__init__()
        self.base_network = BaseNetwork(pretained_model)
        self.SSAModule  = SSAModule(params)
        self.ParsingNetwork = ParsingTask(params)
        self.HuARNetwork = ARTask(params)
        self.Ou5ARNetwork = ARTask(params)
        self.Ou10ARNetwork = ARTask(params)
        self.Ou15ARNetwork = ARTask(params)

    def forward(self, tasktype, texts,input_mask, segment_ids, sep_index_list, edu_nums, speakers, turns):
        rep_x = self.base_network(texts, input_mask,  segment_ids)
        predict_path, batch, node_num = self.SSAModule(rep_x, sep_index_list,
                                    edu_nums, speakers, turns)
        if tasktype == 'parsing':
            link_scores, label_scores = \
                self.ParsingNetwork(predict_path, batch, node_num)
            output = (link_scores, label_scores)
        elif tasktype == 'hu_ar':
            link_scores, label_scores = \
                self.HuARNetwork(predict_path, batch, node_num)
            output = (link_scores, label_scores)
        elif tasktype == 'ou5_ar':
            link_scores, label_scores = \
                self.Ou5ARNetwork(predict_path, batch, node_num)
            output = (link_scores, label_scores)
        elif tasktype == 'ou10_ar':
            link_scores, label_scores = \
                self.Ou10ARNetwork(predict_path, batch, node_num)
            output = (link_scores, label_scores)
        elif tasktype == 'ou15_ar':
            link_scores, label_scores = \
                self.Ou15ARNetwork(predict_path, batch, node_num)
            output = (link_scores, label_scores)
        return output

class ActorNetwork(nn.Module):
    def __init__(self, args):
        super(ActorNetwork, self).__init__()
        self.args = args
        self.actor = nn.Linear(args.state_dim, 2)
        nn.init.xavier_uniform_(self.actor.weight)
        self.nl = nn.Tanh()

    def forward(self, x):
        action_probs = F.softmax(self.actor(x), dim=-1)
        return action_probs

class CriticNetwork(nn.Module):
    def __init__(self, args, pretrained_model):
        super(CriticNetwork, self).__init__()
        self.args = args
        self.task_model = TaskSpecificNetwork1(args, pretrained_model)
        self.ff1 = nn.Linear(args.state_dim, args.hdim)
        nn.init.xavier_uniform_(self.ff1.weight)
        self.critic_layer = nn.Linear(args.hdim, 1)
        nn.init.xavier_uniform_(self.critic_layer.weight)
        self.nl = nn.Tanh()

    def forward(self, x, att_mask, segment_ids):
        x_out = self.task_model.base_network(x,att_mask,segment_ids)[0][:, 0, :].detach()
        c_in = self.nl(self.ff1(x_out))
        out = torch.sigmoid(self.critic_layer(c_in))
        out = torch.mean(out)
        return x_out, out

    def task_output(self, tasktype, texts, input_mask, segment_ids, sep_index_list, edu_nums, speakers, turns):
        return self.task_model(tasktype, texts, input_mask, segment_ids, sep_index_list, edu_nums, speakers, turns)

class PolicyNetwork(nn.Module):
    def __init__(self, args, pretrained_model):
        super(PolicyNetwork, self).__init__()
        self.args = args
        self.actor = ActorNetwork(args)
        self.critic = CriticNetwork(args, pretrained_model)
        self.task_optims = {}
        self.loss_fns = {}
        #hu_ar
        param_groups = [{'params': [p for p in self.critic.task_model.base_network.parameters() if p.requires_grad],
                         'lr':args.pretrained_model_learning_rate}]
        param_groups.append(
            {'params': [p for p in self.critic.task_model.SSAModule.parameters() if p.requires_grad],
             'lr': args.learning_rate})
        param_groups.append({'params': [p for p in self.critic.task_model.HuARNetwork.parameters() if p.requires_grad],
                         'lr':args.learning_rate})
        if args.TST_Learning_Mode:
            optimizer_cls = AdamWTSTLearning
            optimizer_kwargs = {
                "betas": (.9, .999),
                "eps": 1e-6,
            }
            optimizer_kwargs["lr"] = args.learning_rate
            optimizer = optimizer_cls(param_groups, **optimizer_kwargs)
            self.task_optims['hu_ar'] = optimizer
        else:
            self.task_optims['hu_ar'] = AdamW(param_groups, lr=args.learning_rate)
        self.loss_fns['hu_ar'] = nn.CrossEntropyLoss()
        #ou5_ar
        param_groups = [{'params': [p for p in self.critic.task_model.base_network.parameters() if p.requires_grad],
                         'lr': args.pretrained_model_learning_rate}]
        param_groups.append(
            {'params': [p for p in self.critic.task_model.SSAModule.parameters() if p.requires_grad],
             'lr': args.learning_rate})
        param_groups.append(
            {'params': [p for p in self.critic.task_model.Ou5ARNetwork.parameters() if p.requires_grad],
             'lr': args.learning_rate})
        if args.TST_Learning_Mode:
            optimizer_cls = AdamWTSTLearning
            optimizer_kwargs = {
                "betas": (.9, .999),
                "eps": 1e-6,
            }
            optimizer_kwargs["lr"] = args.learning_rate
            optimizer = optimizer_cls(param_groups, **optimizer_kwargs)
            self.task_optims['ou5_ar'] = optimizer
        else:
            self.task_optims['ou5_ar'] = AdamW(param_groups, lr=args.learning_rate)
        self.loss_fns['ou5_ar'] = nn.CrossEntropyLoss()
        #ou10_ar
        param_groups = [{'params': [p for p in self.critic.task_model.base_network.parameters() if p.requires_grad],
                         'lr': args.pretrained_model_learning_rate}]
        param_groups.append(
            {'params': [p for p in self.critic.task_model.SSAModule.parameters() if p.requires_grad],
             'lr': args.learning_rate})
        param_groups.append(
            {'params': [p for p in self.critic.task_model.Ou10ARNetwork.parameters() if p.requires_grad],
             'lr': args.learning_rate})
        if args.TST_Learning_Mode:
            optimizer_cls = AdamWTSTLearning
            optimizer_kwargs = {
                "betas": (.9, .999),
                "eps": 1e-6,
            }
            optimizer_kwargs["lr"] = args.learning_rate
            optimizer = optimizer_cls(param_groups, **optimizer_kwargs)
            self.task_optims['ou10_ar'] = optimizer
        else:
            self.task_optims['ou10_ar'] = AdamW(param_groups, lr=args.learning_rate)
        self.loss_fns['ou10_ar'] = nn.CrossEntropyLoss()
        #ou15_ar
        param_groups = [{'params': [p for p in self.critic.task_model.base_network.parameters() if p.requires_grad],
                         'lr': args.pretrained_model_learning_rate}]
        param_groups.append(
            {'params': [p for p in self.critic.task_model.SSAModule.parameters() if p.requires_grad],
             'lr': args.learning_rate})
        param_groups.append(
            {'params': [p for p in self.critic.task_model.Ou15ARNetwork.parameters() if p.requires_grad],
             'lr': args.learning_rate})
        if args.TST_Learning_Mode:
            optimizer_cls = AdamWTSTLearning
            optimizer_kwargs = {
                "betas": (.9, .999),
                "eps": 1e-6,
            }
            optimizer_kwargs["lr"] = args.learning_rate
            optimizer = optimizer_cls(param_groups, **optimizer_kwargs)
            self.task_optims['ou15_ar'] = optimizer
        else:
            self.task_optims['ou15_ar'] = AdamW(param_groups, lr=args.learning_rate)
        self.loss_fns['ou15_ar'] = nn.CrossEntropyLoss()
        #parsing
        param_groups = [{'params': [p for p in self.critic.task_model.base_network.parameters() if p.requires_grad],
                         'lr': args.pretrained_model_learning_rate}]
        param_groups.append(
            {'params': [p for p in self.critic.task_model.SSAModule.parameters() if p.requires_grad],
             'lr': args.learning_rate})
        param_groups.append(
            {'params': [p for p in self.critic.task_model.ParsingNetwork.parameters() if p.requires_grad],
             'lr': args.learning_rate})
        if args.TST_Learning_Mode:
            optimizer_cls = AdamWTSTLearning
            optimizer_kwargs = {
                "betas": (.9, .999),
                "eps": 1e-6,
            }
            optimizer_kwargs["lr"] = args.learning_rate
            optimizer = optimizer_cls(param_groups, **optimizer_kwargs)
            self.task_optims['parsing'] = optimizer
        else:
            self.task_optims['parsing'] = AdamW(param_groups, lr=args.learning_rate)
        self.loss_fns['parsing'] = nn.CrossEntropyLoss()
        self.saved_actions = []
        self.rewards = []

    def set_gradient_mask(self, mask, type):
        self.task_optims[type].set_gradient_mask(mask)

    def forward(self, batch_x, text_mask, segmend_ids):
        batch_rep, exp_reward = self.critic(batch_x, text_mask, segmend_ids)
        action_probs = self.actor(batch_rep)
        return action_probs, batch_rep, exp_reward

    def compute_f1_and_loss_reward(self, tasktype, eval_dataloader):
        eval_matrix = {
            'hypothesis': None,
            'reference': None,
            'edu_num': None
        }
        accum_eval_link_loss, accum_eval_label_loss = [], []
        for batch in eval_dataloader:
            texts, input_mask, segment_ids, labels, sep_index, pairs, graphs, speakers, turns, edu_nums,ids = batch
            texts, input_mask, segment_ids, graphs, speakers, turns, edu_nums = \
                texts.cuda(), input_mask.cuda(), segment_ids.cuda(), graphs.cuda(), speakers.cuda(), turns.cuda(), edu_nums.cuda()
            mask = get_mask(node_num=edu_nums + 1, max_edu_dist=self.args.max_edu_dist).cuda()
            with torch.no_grad():
                link_scores, label_scores = self.critic.task_output(tasktype, texts, input_mask, segment_ids,  sep_index,
                                                                edu_nums, speakers, turns)

            eval_link_loss, eval_label_loss = compute_loss(link_scores, label_scores, graphs, mask)
            accum_eval_link_loss.append((eval_link_loss.sum(), eval_link_loss.size(-1)))
            accum_eval_label_loss.append((eval_label_loss.sum(), eval_label_loss.size(-1)))
            batch_size = link_scores.size(0)
            max_len = edu_nums.max()
            link_scores[~mask] = -1e9
            predicted_links = torch.argmax(link_scores, dim=-1)
            predicted_labels = torch.argmax(label_scores.reshape(-1, max_len + 1, self.args.relation_type_num)[
                                                torch.arange(batch_size * (max_len + 1)), predicted_links.reshape(
                                                    -1)].reshape(batch_size, max_len + 1, self.args.relation_type_num),
                                            dim=-1)
            predicted_links = predicted_links[:, 1:] - 1
            predicted_labels = predicted_labels[:, 1:]
            for i in range(batch_size):
                hp_pairs = {}
                step = 1
                while step < edu_nums[i]:
                    link = predicted_links[i][step].item()
                    label = predicted_labels[i][step].item()
                    hp_pairs[(link, step)] = label
                    step += 1
                predicted_result = {'hypothesis': hp_pairs,
                                    'reference': pairs[i],
                                    'edu_num': step}
                record_eval_result(eval_matrix, predicted_result)
        f1_bi, f1_multi = tsinghua_F1(eval_matrix)
        a, b = zip(*accum_eval_link_loss)
        c, d = zip(*accum_eval_label_loss)
        eval_link_loss, eval_label_loss = sum(a) / sum(b), sum(c) / sum(d)
        if tasktype == 'hu_rs' or tasktype == 'ou5_ar' or tasktype == 'ou10_ar' or tasktype == 'ou15_ar':
            total_loss = eval_link_loss
            total_f1 = f1_bi
        elif tasktype == 'parsing':
            total_loss = eval_link_loss + eval_label_loss
            total_f1 = f1_bi + f1_multi
        
        print('tasktype {}, link f1 is {}, rel f1 is {}'.format(tasktype, f1_bi, f1_multi))
        print('tasktype {}, link loss is {}, rel loss is {}'.format(tasktype, eval_link_loss, eval_label_loss))
        return total_loss, total_f1
    def compute_f1_and_loss_reward_rl(self, tasktype, eval_dataloader):
        eval_matrix = {
            'hypothesis': None,
            'reference': None,
            'edu_num': None
        }
        accum_eval_link_loss, accum_eval_label_loss = [], []
        for batch in eval_dataloader:
            texts, input_mask, segment_ids, labels, sep_index, pairs, graphs, speakers, turns, edu_nums,ids = batch
            texts, input_mask, segment_ids, graphs, speakers, turns, edu_nums = \
                texts.cuda(), input_mask.cuda(), segment_ids.cuda(), graphs.cuda(), speakers.cuda(), turns.cuda(), edu_nums.cuda()
            mask = get_mask(node_num=edu_nums + 1, max_edu_dist=self.args.max_edu_dist).cuda()
            with torch.no_grad():
                link_scores, label_scores = self.critic.task_output(tasktype, texts, input_mask, segment_ids,  sep_index,
                                                                edu_nums, speakers, turns)

            eval_link_loss, eval_label_loss = compute_loss(link_scores, label_scores, graphs, mask)
            accum_eval_link_loss.append((eval_link_loss.sum(), eval_link_loss.size(-1)))
            accum_eval_label_loss.append((eval_label_loss.sum(), eval_label_loss.size(-1)))
            batch_size = link_scores.size(0)
            max_len = edu_nums.max()
            link_scores[~mask] = -1e9
            predicted_links = torch.argmax(link_scores, dim=-1)
            predicted_labels = torch.argmax(label_scores.reshape(-1, max_len + 1, self.args.relation_type_num)[
                                                torch.arange(batch_size * (max_len + 1)), predicted_links.reshape(
                                                    -1)].reshape(batch_size, max_len + 1, self.args.relation_type_num),
                                            dim=-1)
            predicted_links = predicted_links[:, 1:] - 1
            predicted_labels = predicted_labels[:, 1:]
            for i in range(batch_size):
                hp_pairs = {}
                step = 1
                while step < edu_nums[i]:
                    link = predicted_links[i][step].item()
                    label = predicted_labels[i][step].item()
                    hp_pairs[(link, step)] = label
                    step += 1
                predicted_result = {'hypothesis': hp_pairs,
                                    'reference': pairs[i],
                                    'edu_num': step}
                record_eval_result(eval_matrix, predicted_result)
        f1_bi, f1_multi = tsinghua_F1(eval_matrix)
        a, b = zip(*accum_eval_link_loss)
        c, d = zip(*accum_eval_label_loss)
        eval_link_loss, eval_label_loss = sum(a) / sum(b), sum(c) / sum(d)
        if tasktype == 'hu_rs' or tasktype == 'ou5_ar' or tasktype == 'ou10_ar' or tasktype == 'ou15_ar':
            total_loss = eval_link_loss
            total_f1 = f1_bi
        elif tasktype == 'parsing':
            total_loss = eval_link_loss 
            total_f1 = f1_bi 
        
        print('tasktype {}, link f1 is {}'.format(tasktype, f1_bi))
        print('tasktype {}, link loss is {}'.format(tasktype, eval_link_loss))
        return total_loss, total_f1
    def train_minibatch(self, task_type, batch):
        accum_train_link_loss = accum_train_label_loss = 0
        for mini_batch in batch:
            texts, input_mask, segment_ids, labels, sep_index, pairs, graphs, speakers, turns, edu_nums = mini_batch
            texts, input_mask, segment_ids, graphs, speakers, turns, edu_nums = \
                    texts.cuda(), input_mask.cuda(), segment_ids.cuda(), graphs.cuda(), speakers.cuda(), turns.cuda(), edu_nums.cuda()
            mask = get_mask(node_num=edu_nums + 1, max_edu_dist=self.args.max_edu_dist).cuda()
            link_scores, label_scores = self.critic.task_output(task_type, texts, input_mask, segment_ids,  sep_index,
                                                                edu_nums, speakers, turns)
            link_loss, label_loss = compute_loss(link_scores.clone(), label_scores.clone(), graphs, mask )
            link_loss = link_loss.mean()
            label_loss = label_loss.mean()
            if task_type=='hu_ar' or task_type=='ou5_ar' or task_type=='ou10_ar' or task_type=='ou15_ar':
                loss = link_loss
            elif task_type =='parsing':
                loss = link_loss + label_loss
            self.critic.task_model.zero_grad()
            loss.backward()
            self.task_optims[task_type].step()
            accum_train_link_loss += link_loss.item()
            accum_train_label_loss += label_loss.item()
        return accum_train_link_loss, accum_train_label_loss

    def train_minibatch_rl(self, task_type, batch):
        accum_train_link_loss = accum_train_label_loss = 0
        texts, input_mask, segment_ids, labels, sep_index, pairs, graphs, speakers, turns, edu_nums = batch
        texts, input_mask, segment_ids, graphs, speakers, turns, edu_nums = \
                texts.cuda(), input_mask.cuda(), segment_ids.cuda(), graphs.cuda(), speakers.cuda(), turns.cuda(), edu_nums.cuda()
        mask = get_mask(node_num=edu_nums + 1, max_edu_dist=self.args.max_edu_dist).cuda()
        link_scores, label_scores = self.critic.task_output(task_type, texts, input_mask, segment_ids,  sep_index,
                                                            edu_nums, speakers, turns)
        link_loss, label_loss = compute_loss(link_scores.clone(), label_scores.clone(), graphs, mask)
        link_loss = link_loss.mean()
        label_loss = label_loss.mean()
        if task_type == 'hu_ar' or task_type == 'ou5_ar' or task_type == 'ou10_ar' or task_type == 'ou15_ar':
            loss = link_loss
        elif task_type =='parsing':
            loss = link_loss + label_loss
        self.critic.task_model.zero_grad()
        loss.backward()
        self.task_optims[task_type].step()
        accum_train_link_loss += link_loss.item()
        accum_train_label_loss += label_loss.item()
        return accum_train_link_loss, accum_train_label_loss

    def train_minibatch_optim_link_rl(self, task_type, batch):
        accum_train_link_loss = accum_train_label_loss = 0
        texts, input_mask, segment_ids, labels, sep_index, pairs, graphs, speakers, turns, edu_nums,ex_ids = batch
        texts, input_mask, segment_ids, graphs, speakers, turns, edu_nums = \
            texts.cuda(), input_mask.cuda(), segment_ids.cuda(), graphs.cuda(), speakers.cuda(), turns.cuda(), edu_nums.cuda()
        mask = get_mask(node_num=edu_nums + 1, max_edu_dist=self.args.max_edu_dist).cuda()
        link_scores, label_scores = self.critic.task_output(task_type, texts, input_mask, segment_ids, sep_index,
                                                            edu_nums, speakers, turns)
        link_loss, label_loss = compute_loss(link_scores.clone(), label_scores.clone(), graphs, mask)
        link_loss = link_loss.mean()
        label_loss = label_loss.mean()
        loss = link_loss
        self.critic.task_model.zero_grad()
        loss.backward()
        self.task_optims[task_type].step()
        accum_train_link_loss += link_loss.item()
        accum_train_label_loss += label_loss.item()
        return accum_train_link_loss, accum_train_label_loss

class SSAModule(nn.Module):
    def __init__(self, params):
        super(SSAModule, self).__init__()
        self.params = params
        self.gru = nn.GRU(params.hidden_size, params.hidden_size // 2, batch_first=True, bidirectional=True)
        self.path_emb = PathEmbedding(params)
        self.path_update = PathUpdateModel(params)
        self.gnn = StructureAwareAttention(params.hidden_size, params.path_hidden_size, params.num_heads,
                                           params.dropout)
        self.layer_num = params.num_layers
        self.norm = nn.LayerNorm(params.hidden_size)
        self.dropout = nn.Dropout(params.dropout)
        self.hidden_size = params.hidden_size
        self.root = nn.Parameter(torch.zeros(params.hidden_size), requires_grad=False)

    def __fetch_sep_rep2(self, ten_output, seq_index):
        batch, seq_len, hidden_size = ten_output.shape
        shift_sep_index_list = self.get_shift_sep_index_list(seq_index, seq_len)
        ten_output = torch.reshape(ten_output, (batch * seq_len, hidden_size))
        sep_embedding = ten_output[shift_sep_index_list, :]
        sep_embedding = torch.reshape(sep_embedding, (batch, len(seq_index[0]), hidden_size))
        return sep_embedding

    def get_shift_sep_index_list(self, pad_sep_index_list, seq_len):
        new_pad_sep_index_list = []
        for index in range(len(pad_sep_index_list)):
            new_pad_sep_index_list.extend([item + index * seq_len for item in pad_sep_index_list[index]])
        return new_pad_sep_index_list

    def padding_sep_index_list(self, sep_index_list):

        max_edu = max([len(a) for a in sep_index_list])
        total_new_sep_index_list = []
        for index_list in sep_index_list:
            new_sep_index_list = []
            gap = max_edu - len(index_list)
            new_sep_index_list.extend(index_list)
            for i in range(gap):
                new_sep_index_list.append(index_list[-1])
            total_new_sep_index_list.append(new_sep_index_list)
        return max_edu, total_new_sep_index_list

    def forward(self, SentenceEmbedding, sep_index_list, edu_nums, speakers, turns):
        sentences = SentenceEmbedding[0]
        batch_size = sentences.shape[0]
        edu_num, pad_sep_index_list = self.padding_sep_index_list(sep_index_list)
        node_num = edu_num + 1
        sentences = self.__fetch_sep_rep2(sentences, pad_sep_index_list)
        nodes = torch.cat((self.root.expand(batch_size, 1, sentences.size(-1)),
                           sentences.reshape(batch_size, edu_num, -1)), dim=1)
        dialogue, _ = self.gru(nodes)
        nodes = self.dropout(dialogue)
        edu_nums = edu_nums + 1

        edu_attn_mask = torch.arange(edu_nums.max()).expand(len(edu_nums), edu_nums.max()).cuda() < edu_nums.unsqueeze(
            1)
        edu_attn_mask = self.gnn.masking_bias(edu_attn_mask)
        const_path = self.path_emb(speakers, turns)
        struct_path = torch.zeros_like(const_path)
        for _ in range(self.layer_num):
            nodes = self.gnn(nodes, edu_attn_mask, struct_path + const_path)
            struct_path = self.path_update(nodes, const_path, struct_path)
            struct_path = self.dropout(struct_path)
        predicted_path = torch.cat((struct_path, struct_path.transpose(1, 2)), -1)
        return predicted_path, batch_size, node_num
#cited from wang et al 2021
class StructureAwareAttention(nn.Module):
    def __init__(self, hidden_size, path_hidden_size, head_num, dropout):
        super(StructureAwareAttention, self).__init__()
        self.q_transform = nn.Linear(hidden_size, hidden_size)
        self.k_transform = nn.Linear(hidden_size, hidden_size)
        self.v_transform = nn.Linear(hidden_size, hidden_size)
        self.struct_k_transform = nn.Linear(path_hidden_size, hidden_size // head_num)
        self.struct_v_transform = nn.Linear(path_hidden_size, hidden_size // head_num)
        self.o_transform = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.ReLU()
        self.hidden_size = hidden_size
        self.head_num = head_num
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_size)
        self.path_norm = nn.LayerNorm(path_hidden_size)

    def forward(self, nodes, bias, paths):
        q, k, v = self.q_transform(nodes), self.k_transform(nodes), self.v_transform(nodes)
        q = self.split_heads(q, self.head_num)
        k = self.split_heads(k, self.head_num)
        v = self.split_heads(v, self.head_num)
        paths = self.path_norm(paths)
        struct_k, struct_v = self.struct_k_transform(paths), self.struct_v_transform(paths)
        q = q * (self.hidden_size // self.head_num) ** -0.5
        w = torch.matmul(q, k.transpose(-1, -2)) + torch.matmul(q.transpose(1, 2),
                                                                struct_k.transpose(-1, -2)).transpose(1, 2) + bias
        w = torch.nn.functional.softmax(w, dim=-1)
        output = torch.matmul(w, v) + torch.matmul(w.transpose(1, 2), struct_v).transpose(1, 2)
        output = self.activation(self.o_transform(self.combine_heads(output)))
        return self.norm(nodes + self.dropout(output))

    @staticmethod
    def split_heads(x, heads):
        batch = x.shape[0]
        length = x.shape[1]
        channels = x.shape[2]

        y = torch.reshape(x, [batch, length, heads, channels // heads])
        return torch.transpose(y, 2, 1)

    @staticmethod
    def combine_heads(x):
        batch = x.shape[0]
        heads = x.shape[1]
        length = x.shape[2]
        channels = x.shape[3]

        y = torch.transpose(x, 2, 1)

        return torch.reshape(y, [batch, length, heads * channels])

    @staticmethod
    def masking_bias(mask, inf=-1e9):
        ret = ~mask * inf
        return torch.unsqueeze(torch.unsqueeze(ret, 1), 1)

class PathEmbedding(nn.Module):
    def __init__(self, params):
        super(PathEmbedding, self).__init__()
        self.speaker = nn.Embedding(2, params.path_hidden_size // 4)
        self.turn = nn.Embedding(2, params.path_hidden_size // 4)
        self.valid_dist = params.valid_dist
        self.position = nn.Embedding(self.valid_dist * 2 + 3, params.path_hidden_size // 2)

        self.tmp = torch.arange(200)
        self.path_pool = self.tmp.expand(200, 200) - self.tmp.unsqueeze(1)
        self.path_pool[self.path_pool > self.valid_dist] = self.valid_dist + 1
        self.path_pool[self.path_pool < -self.valid_dist] = -self.valid_dist - 1
        self.path_pool += self.valid_dist + 1

    def forward(self, speaker, turn):
        batch_size, node_num, _ = speaker.size()
        speaker = self.speaker(speaker)
        turn = self.turn(turn)
        position = self.position(self.path_pool[:node_num, :node_num].cuda())
        position = position.expand(batch_size, node_num, node_num, position.size(-1))
        return torch.cat((speaker, turn, position), dim=-1)

class Classifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_class):
        super().__init__()
        self.input_transform = nn.Sequential(nn.Linear(input_size, hidden_size), nn.Tanh())
        self.output_transform = nn.Linear(hidden_size, num_class)

    def forward(self, x):
        return self.output_transform(self.input_transform(x))

class PathUpdateModel(nn.Module):
    def __init__(self, params):
        super(PathUpdateModel, self).__init__()
        self.x_dim = params.hidden_size
        self.h_dim = params.path_hidden_size

        self.r = nn.Linear(2*self.x_dim + self.h_dim, self.h_dim, True)
        self.z = nn.Linear(2*self.x_dim + self.h_dim, self.h_dim, True)

        self.c = nn.Linear(2*self.x_dim, self.h_dim, True)
        self.u = nn.Linear(self.h_dim, self.h_dim, True)

    def forward(self, nodes, bias, hx, mask=None):
        batch_size, node_num, hidden_size = nodes.size()
        nodes = nodes.unsqueeze(1).expand(batch_size, node_num, node_num, hidden_size)
        nodes = torch.cat((nodes, nodes.transpose(1, 2)),dim=-1)  # B N N H
        if mask is not None:
            nodes, bias = nodes[mask], bias[mask]
        if hx is None:
            hx = torch.zeros_like(bias)

        rz_input = torch.cat((nodes, hx), -1)
        r = torch.sigmoid(self.r(rz_input))
        z = torch.sigmoid(self.z(rz_input))

        u = torch.tanh(self.c(nodes) + r * self.u(hx))

        new_h = z * hx + (1 - z) * u
        return new_h
