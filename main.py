import argparse
import random
import os
from torch.distributions import Categorical
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, AdamW
from Model import PolicyNetwork
from dialogue_dataset import DialogueDataset
from tqdm import tqdm
import torch.nn.functional as F
from utils import *

def seed_everything(seed=256):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_mol_file', type=str)
    parser.add_argument('--eval_mol_file', type=str)
    parser.add_argument('--test_mol_file', type=str)
    parser.add_argument('--train_hu_file', type=str)
    parser.add_argument('--train_ou5_file', type=str)
    parser.add_argument('--train_ou10_file', type=str)
    parser.add_argument('--train_ou15_file', type=str)
    parser.add_argument('--dataset_dir', type=str, default='dataset')
    parser.add_argument('--model_name_or_path', type=str, default='./pretrained_model/bert-base-uncased/')
    parser.add_argument('--remake_dataset', action="store_true")
    parser.add_argument('--remake_tokenizer', action="store_true")
    parser.add_argument('--max_edu_dist', type=int, default=20)
    parser.add_argument('--path_hidden_size', type=int, default=384)
    parser.add_argument('--hidden_size', type=int, default=768)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--speaker', action='store_true')
    parser.add_argument('--valid_dist', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--pretrained_model_learning_rate', type=float, default=1e-5)
    parser.add_argument('--ST_epoches', type=int, default=5)
    parser.add_argument('--TST_epoches', type=int, default=3)
    parser.add_argument('--RL_epoches', type=int, default=3)
    parser.add_argument('--mol_pool_size', type=int, default=100)
    parser.add_argument('--hu_pool_size', type=int, default=100)
    parser.add_argument('--ou5_pool_size', type=int, default=100)
    parser.add_argument('--ou10_pool_size', type=int, default=100)
    parser.add_argument('--ou15_pool_size', type=int, default=100)
    parser.add_argument('--eval_mol_pool_size', type=int, default=10)
    parser.add_argument('--mol_batch_size', type=int, default=100)
    parser.add_argument('--hu_batch_size', type=int, default=100)
    parser.add_argument('--ou5_batch_size', type=int, default=100)
    parser.add_argument('--ou10_batch_size', type=int, default=100)
    parser.add_argument('--ou15_batch_size', type=int, default=100)
    parser.add_argument('--hu_batch_size_rl', type=int, default=10000)
    parser.add_argument('--ou5_batch_size_rl', type=int, default=10000)
    parser.add_argument('--ou10_batch_size_rl', type=int, default=5000)
    parser.add_argument('--ou15_batch_size_rl', type=int, default=4000)
    parser.add_argument('--ST_model_path', type=str, default='model.pt')
    parser.add_argument('--TST_model_path', type=str, default='model.pt')
    parser.add_argument('--hu_select_id_file', type=str, default='hu_select_id_file.txt')
    parser.add_argument('--ou5_select_id_file', type=str, default='ou5_select_id_file.txt')
    parser.add_argument('--ou10_select_id_file', type=str, default='ou10_select_id_file.txt')
    parser.add_argument('--ou15_select_id_file', type=str, default='ou15_select_id_file.txt')
    parser.add_argument('--hu_selected_data_file', type=str, default='hu_selected_data.json')
    parser.add_argument('--ou5_selected_data_file', type=str, default='ou5_selected_data.json')
    parser.add_argument('--ou10_selected_data_file', type=str, default='ou10_selected_data.json')
    parser.add_argument('--ou15_selected_data_file', type=str, default='ou15_selected_data.json')
    parser.add_argument('--do_train', action="store_true")
    parser.add_argument('--report_step', type=int, default= 20)
    parser.add_argument('--early_stop', type=int, default=1000)
    parser.add_argument('--utt_max_len', type=int, default= 24)
    parser.add_argument('--max_mol_text_len', type=int, default= 380)
    parser.add_argument('--max_hu_text_len', type=int, default= 180)
    parser.add_argument('--max_ou5_text_len', type=int, default= 130)
    parser.add_argument('--max_ou10_text_len', type=int, default= 260)
    parser.add_argument('--max_ou15_text_len', type=int, default= 380)
    parser.add_argument('--TST_Learning_Mode', type=bool, default=False)
    parser.add_argument('--alpha', type=float, default=0.7)
    parser.add_argument('--state_dim', type=int, default= 768)
    parser.add_argument('--hdim', type=int, default= 384)
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--seed', type=int, default= 512)
    args = parser.parse_args()
    seed_everything(args.seed)
    args.n_gpu = torch.cuda.device_count()
    args.device = torch.device("cuda")

    if not os.path.isdir(args.dataset_dir):
        os.mkdir(args.dataset_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    train_hu_file = os.path.join(args.dataset_dir, 'train_hu.pt')
    train_ou5_file = os.path.join(args.dataset_dir, 'train_ou5.pt')
    train_ou10_file = os.path.join(args.dataset_dir, 'train_ou10.pt')
    train_ou15_file = os.path.join(args.dataset_dir, 'train_ou15.pt')

    train_mol_file = os.path.join(args.dataset_dir, 'train_mol.pt')
    test_mol_file = os.path.join(args.dataset_dir, 'test_mol.pt')
    eval_mol_file = os.path.join(args.dataset_dir, 'eval_mol.pt')

    if os.path.exists(train_mol_file) and not args.remake_dataset:
        print('loading dataset..')
        if args.do_train:
            train_dataset_hu = torch.load(train_hu_file)
            train_dataset_ou5 = torch.load(train_ou5_file)
            train_dataset_ou10 = torch.load(train_ou10_file)
            train_dataset_ou15 = torch.load(train_ou15_file)
            eval_dataset_mol = torch.load(eval_mol_file)
        train_dataset_mol = torch.load(train_mol_file)
        relations, type2ids, id2types = train_dataset_mol.relations, train_dataset_mol.type2ids, train_dataset_mol.id2types
        if not args.do_train:
            test_dataset_mol = DialogueDataset(args=args, filename=args.test_mol_file, tokenizer=tokenizer,
                                                   mode='test', text_max_sep_len=args.utt_max_len,
                                                   total_seq_len=args.max_mol_text_len)
            test_dataset_mol.get_relations(relations, type2ids, id2types)
            test_dataset_mol.get_discourse_graph()
    else:
        train_dataset_hu = DialogueDataset(args=args, filename= args.train_hu_file, tokenizer=tokenizer, mode='train',text_max_sep_len=args.utt_max_len,
                                           total_seq_len = args.max_hu_text_len)

        train_dataset_ou5 = DialogueDataset(args=args, filename=args.train_ou5_file, tokenizer=tokenizer,
                                                 mode='train', text_max_sep_len=args.utt_max_len,
                                                 total_seq_len=args.max_ou5_text_len)

        train_dataset_ou10 = DialogueDataset(args=args, filename=args.train_ou10_file, tokenizer=tokenizer,
                                            mode='train', text_max_sep_len=args.utt_max_len,
                                            total_seq_len=args.max_ou10_text_len)

        train_dataset_ou15 = DialogueDataset(args=args, filename=args.train_ou15_file, tokenizer=tokenizer,
                                            mode='train', text_max_sep_len=args.utt_max_len,
                                            total_seq_len=args.max_ou15_text_len)

        train_dataset_mol = DialogueDataset(args=args, filename=args.train_mol_file, tokenizer=tokenizer, mode='train',
                                        text_max_sep_len=args.utt_max_len,
                                        total_seq_len=args.max_mol_text_len)

        eval_dataset_mol = DialogueDataset(args=args, filename=args.eval_mol_file, tokenizer=tokenizer, mode='eval',
                                           text_max_sep_len=args.utt_max_len,
                                           total_seq_len = args.max_mol_text_len)


        relations = train_dataset_mol.relations | train_dataset_mol.relations
        type2ids, id2types = DialogueDataset.format_relations(relations)
        train_dataset_mol.get_relations(relations, type2ids, id2types)
        train_dataset_mol.get_discourse_graph()

        eval_dataset_mol.get_relations(relations, type2ids, id2types)
        eval_dataset_mol.get_discourse_graph()

        train_dataset_hu.get_relations(relations, type2ids, id2types)
        train_dataset_hu.get_discourse_graph()

        train_dataset_ou5.get_relations(relations, type2ids, id2types)
        train_dataset_ou5.get_discourse_graph()

        train_dataset_ou10.get_relations(relations, type2ids, id2types)
        train_dataset_ou10.get_discourse_graph()

        train_dataset_ou15.get_relations(relations, type2ids, id2types)
        train_dataset_ou15.get_discourse_graph()

        print('saving dataset..')
        torch.save(train_dataset_mol, train_mol_file)
        torch.save(eval_dataset_mol, eval_mol_file)
        torch.save(train_dataset_hu, train_hu_file)
        torch.save(train_dataset_ou5, train_ou5_file)
        torch.save(train_dataset_ou10, train_ou10_file)
        torch.save(train_dataset_ou15, train_ou15_file)
    args.relation_type_num = len(id2types)
    pretrained_model = AutoModel.from_pretrained(args.model_name_or_path)

    def train_collate_fn_mol(examples):

        def pool(d):
            d = sorted(d, key=lambda x: x[9])
            edu_nums = [x[9] for x in d]
            buckets = []
            i, j, t = 0, 0, 0
            for edu_num in edu_nums:
                if t + edu_num > args.mol_batch_size:
                    buckets.append((i, j))
                    i, t = j, 0
                t += edu_num
                j += 1
            buckets.append((i, j))

            for bucket in buckets:
                batch = d[bucket[0]:bucket[1]]

                texts, input_mask, segment_ids, _,sep_index, pairs,graphs, speakers, turns, edu_nums, _ = zip(*batch)
                texts = torch.stack(texts, dim=0)
                segment_ids = torch.stack(segment_ids, dim=0)
                input_mask = torch.stack(input_mask, dim=0)
                assert texts.shape[0] == segment_ids.shape[0] == input_mask.shape[0] == len(sep_index)
                speakers = ints_to_tensor(list(speakers))
                turns = ints_to_tensor(list(turns))
                graphs = ints_to_tensor(list(graphs))
                edu_nums = torch.tensor(edu_nums)
                yield texts, input_mask, segment_ids, _, sep_index,pairs, graphs, speakers, turns, edu_nums

        return pool(examples)

    def eval_collate_fn_mol(examples):
        texts, input_mask, segment_ids, _, sep_index,pairs, graphs, speakers, turns, edu_nums, ids = zip(*examples)
        texts = torch.stack(texts, dim=0)
        segment_ids = torch.stack(segment_ids, dim=0)
        input_mask = torch.stack(input_mask, dim=0)
        assert texts.shape[0] == segment_ids.shape[0] == input_mask.shape[0] == len(sep_index)
        speakers = ints_to_tensor(list(speakers))
        turns = ints_to_tensor(list(turns))
        graphs = ints_to_tensor(list(graphs))
        edu_nums = torch.tensor(edu_nums)
        return texts, input_mask, segment_ids, _, sep_index, pairs,graphs, speakers, turns, edu_nums, ids

    def train_collate_fn_hu(examples):

        def pool(d):
            d = sorted(d, key=lambda x: x[9])
            edu_nums = [x[9] for x in d]
            buckets = []
            i, j, t = 0, 0, 0
            for edu_num in edu_nums:
                if t + edu_num > args.hu_batch_size:
                    buckets.append((i, j))
                    i, t = j, 0
                t += edu_num
                j += 1
            buckets.append((i, j))

            for bucket in buckets:
                batch = d[bucket[0]:bucket[1]]

                texts, input_mask, segment_ids, _,sep_index, pairs,graphs, speakers, turns, edu_nums, _ = zip(*batch)
                texts = torch.stack(texts, dim=0)
                segment_ids = torch.stack(segment_ids, dim=0)
                input_mask = torch.stack(input_mask, dim=0)
                assert texts.shape[0] == segment_ids.shape[0] == input_mask.shape[0] == len(sep_index)
                speakers = ints_to_tensor(list(speakers))
                turns = ints_to_tensor(list(turns))
                graphs = ints_to_tensor(list(graphs))
                edu_nums = torch.tensor(edu_nums)
                yield texts, input_mask, segment_ids, _, sep_index,pairs, graphs, speakers, turns, edu_nums

        return pool(examples)

    def train_collate_fn_ou_len5(examples):

        def pool(d):
            d = sorted(d, key=lambda x: x[9])
            edu_nums = [x[9] for x in d]
            buckets = []
            i, j, t = 0, 0, 0
            for edu_num in edu_nums:
                if t + edu_num > args.ou5_batch_size:
                    buckets.append((i, j))
                    i, t = j, 0
                t += edu_num
                j += 1
            buckets.append((i, j))

            for bucket in buckets:
                batch = d[bucket[0]:bucket[1]]

                texts, input_mask, segment_ids, _,sep_index, pairs,graphs, speakers, turns, edu_nums, _ = zip(*batch)
                texts = torch.stack(texts, dim=0)
                segment_ids = torch.stack(segment_ids, dim=0)
                input_mask = torch.stack(input_mask, dim=0)
                assert texts.shape[0] == segment_ids.shape[0] == input_mask.shape[0] == len(sep_index)
                speakers = ints_to_tensor(list(speakers))
                turns = ints_to_tensor(list(turns))
                graphs = ints_to_tensor(list(graphs))
                edu_nums = torch.tensor(edu_nums)
                yield texts, input_mask, segment_ids, _, sep_index,pairs, graphs, speakers, turns, edu_nums

        return pool(examples)

    def train_collate_fn_ou_len10(examples):

        def pool(d):
            d = sorted(d, key=lambda x: x[9])
            edu_nums = [x[9] for x in d]
            buckets = []
            i, j, t = 0, 0, 0
            for edu_num in edu_nums:
                if t + edu_num > args.ou10_batch_size:
                    buckets.append((i, j))
                    i, t = j, 0
                t += edu_num
                j += 1
            buckets.append((i, j))

            for bucket in buckets:
                batch = d[bucket[0]:bucket[1]]

                texts, input_mask, segment_ids, _,sep_index, pairs,graphs, speakers, turns, edu_nums, _ = zip(*batch)
                texts = torch.stack(texts, dim=0)
                segment_ids = torch.stack(segment_ids, dim=0)
                input_mask = torch.stack(input_mask, dim=0)
                assert texts.shape[0] == segment_ids.shape[0] == input_mask.shape[0] == len(sep_index)
                speakers = ints_to_tensor(list(speakers))
                turns = ints_to_tensor(list(turns))
                graphs = ints_to_tensor(list(graphs))
                edu_nums = torch.tensor(edu_nums)
                yield texts, input_mask, segment_ids, _, sep_index,pairs, graphs, speakers, turns, edu_nums

        return pool(examples)

    def train_collate_fn_ou_len15(examples):

        def pool(d):
            d = sorted(d, key=lambda x: x[9])
            edu_nums = [x[9] for x in d]
            buckets = []
            i, j, t = 0, 0, 0
            for edu_num in edu_nums:
                if t + edu_num > args.ou15_batch_size:
                    buckets.append((i, j))
                    i, t = j, 0
                t += edu_num
                j += 1
            buckets.append((i, j))

            for bucket in buckets:
                batch = d[bucket[0]:bucket[1]]

                texts, input_mask, segment_ids, _,sep_index, pairs,graphs, speakers, turns, edu_nums, _ = zip(*batch)
                texts = torch.stack(texts, dim=0)
                segment_ids = torch.stack(segment_ids, dim=0)
                input_mask = torch.stack(input_mask, dim=0)
                assert texts.shape[0] == segment_ids.shape[0] == input_mask.shape[0] == len(sep_index)
                speakers = ints_to_tensor(list(speakers))
                turns = ints_to_tensor(list(turns))
                graphs = ints_to_tensor(list(graphs))
                edu_nums = torch.tensor(edu_nums)
                yield texts, input_mask, segment_ids, _, sep_index,pairs, graphs, speakers, turns, edu_nums

        return pool(examples)


    def MultiTaskLearning(mtl_model, train_hu_dataloader, train_ou5_dataloader,
                                        train_ou10_dataloader,train_ou15_dataloader, train_mol_dataloader):
        step = 0
        total_hu_loss = total_ou5_loss = total_ou10_loss = total_ou15_loss = total_molweni_link_loss = total_molweni_rel_loss = 0

        
        #train ou5
        for ou5_data_batch in tqdm(train_ou5_dataloader):
            ou5_link_rs_loss, _ = \
                mtl_model.train_minibatch('ou5_ar', ou5_data_batch)
            total_ou5_loss += ou5_link_rs_loss
            step += 1
            if step % args.report_step == 0:
                print('\t{} step ou5 loss: {:.4f} '.format(step, total_ou5_loss / args.report_step))
                total_ou5_loss = 0
            if args.debug:
                break 
        #train ou10
        for ou10_data_batch in tqdm(train_ou10_dataloader):
            ou10_link_rs_loss, _ = \
                mtl_model.train_minibatch('ou10_ar', ou10_data_batch)
            total_ou10_loss += ou10_link_rs_loss
            step += 1
            if step % args.report_step == 0:
                print('\t{} step ou10 loss: {:.4f} '.format(step, total_ou10_loss / args.report_step))
                total_ou10_loss = 0
            if args.debug:
                break 
        #train ou15
        for ou15_data_batch in tqdm(train_ou15_dataloader):
            ou15_link_rs_loss, _ = \
                mtl_model.train_minibatch('ou15_ar', ou15_data_batch)
            total_ou15_loss += ou15_link_rs_loss
            step += 1
            if step % args.report_step == 0:
                print('\t{} step ou15 loss: {:.4f} '.format(step, total_ou15_loss / args.report_step))
                total_ou15_loss = 0
            if args.debug:
                break 
        #train hu
        for hu_data_batch in tqdm(train_hu_dataloader):
            hu_link_rs_loss, _ = \
                mtl_model.train_minibatch('hu_ar', hu_data_batch)
            total_hu_loss += hu_link_rs_loss
            step += 1
            if step % args.report_step == 0:
                print('\t{} step hu loss: {:.4f} '.format(step, total_hu_loss / args.report_step))
                total_hu_loss = 0
            if args.debug:
                break 
        # train mol
        for mol_data_batch in tqdm(train_mol_dataloader):
            temp_link_mol_loss, temp_rel_mol_loss = \
                mtl_model.train_minibatch('parsing', mol_data_batch)
            total_molweni_link_loss += temp_link_mol_loss
            total_molweni_rel_loss += temp_rel_mol_loss
            step += 1
            if step % args.report_step == 0:
                print(
                    '\t{} mol link loss {:.4f}, rel loss {:.4f} '.format(step,
                                  total_molweni_link_loss / args.report_step,
                                  total_molweni_rel_loss / args.report_step))
                total_molweni_link_loss = total_molweni_rel_loss = 0
            if args.debug:
                break 
            
    def generate_TST_mask(args, model, task_type, train_dataloader):
        gradient_mask = dict()
        model.train()
        for name, params in model.named_parameters():
            if 'SSAModule.gnn' in name:
                gradient_mask[params] = params.new_zeros(params.size())
        N = len(train_dataloader)
        for batch in tqdm(train_dataloader):
            for mini_batch in batch:
                texts, input_mask, segment_ids, labels, sep_index, pairs, graphs, speakers, turns, edu_nums = mini_batch
                texts, input_mask, segment_ids, graphs, speakers, turns, edu_nums = \
                    texts.cuda(), input_mask.cuda(), segment_ids.cuda(), graphs.cuda(), speakers.cuda(), turns.cuda(), edu_nums.cuda()
                mask = get_mask(node_num=edu_nums + 1, max_edu_dist=args.max_edu_dist).cuda()
                link_scores, label_scores = model.critic.task_output(task_type, texts, input_mask, segment_ids,
                                                                    sep_index,
                                                                    edu_nums, speakers, turns)
                link_loss, label_loss = compute_loss(link_scores.clone(), label_scores.clone(), graphs, mask)
                link_loss = link_loss.mean()
                label_loss = label_loss.mean()
                if task_type == 'hu_ar' or task_type == 'ou5_ar' or task_type == 'ou10_ar' or task_type == 'ou15_ar':

                    loss = link_loss
                elif task_type == 'parsing':
                    loss = link_loss + label_loss
                loss.backward()
                for name, params in model.named_parameters():
                    if 'SSAModule.gnn' in name:
                        torch.nn.utils.clip_grad_norm_(params, 1.0)
                        gradient_mask[params] += (params.grad ** 2) / N
                model.critic.task_model.zero_grad()
                if args.debug:
                    break
            if args.debug:
                break
        r = None
        for k, v in gradient_mask.items():
            v = v.view(-1).cpu().numpy()
            if r is None:
                r = v
            else:
                r = np.append(r, v)
        polar = np.percentile(r, args.alpha * 100)
        for k in gradient_mask:
            gradient_mask[k] = gradient_mask[k] >= polar
        
        return gradient_mask

    eps = np.finfo(np.float32).eps.item()
   
    
    if args.do_train:
        train_dataloader_mol = DataLoader(dataset=train_dataset_mol, batch_size=args.mol_pool_size,
                                          shuffle=True,
                                          collate_fn=train_collate_fn_mol)

        eval_dataloader_mol = DataLoader(dataset=eval_dataset_mol, batch_size=args.eval_mol_pool_size,
                                         shuffle=False,
                                         collate_fn=eval_collate_fn_mol)

        train_dataloader_hu = DataLoader(dataset=train_dataset_hu, batch_size=args.hu_pool_size, shuffle=True,
                                            collate_fn=train_collate_fn_hu)


        train_dataloader_ou5 = DataLoader(dataset=train_dataset_ou5, batch_size=args.ou5_pool_size,
                                               shuffle=True,
                                               collate_fn=train_collate_fn_ou_len5)

        train_dataloader_ou10 = DataLoader(dataset=train_dataset_ou10, batch_size=args.ou10_pool_size,
                                               shuffle=True,
                                               collate_fn=train_collate_fn_ou_len10)

        train_dataloader_ou15 = DataLoader(dataset=train_dataset_ou15, batch_size=args.ou15_pool_size,
                                               shuffle=True,
                                               collate_fn=train_collate_fn_ou_len15)
        
        #TST
        args.TST_Learning_Mode = True
        model = PolicyNetwork(args=args, pretrained_model=pretrained_model)
        model = model.to(args.device)
        state_dict = torch.load(args.ST_model_path+'.pt')
        model.load_state_dict(state_dict, strict=False)
        print('generate task mask')
        parsing_mask = generate_TST_mask(args, model, 'parsing', train_dataloader_mol)
        hu_mask = generate_TST_mask(args, model, 'hu_ar', train_dataloader_hu)
        
        ou5_mask = generate_TST_mask(args, model, 'ou5_ar', train_dataloader_ou5)
        ou10_mask = generate_TST_mask(args, model, 'ou10_ar', train_dataloader_ou10)
        ou15_mask = generate_TST_mask(args, model, 'ou15_ar', train_dataloader_ou15)
        
        model.set_gradient_mask(parsing_mask, 'parsing')
        model.set_gradient_mask(hu_mask, 'hu_ar')
        model.set_gradient_mask(ou5_mask, 'ou5_ar')
        model.set_gradient_mask(ou10_mask, 'ou10_ar')
        model.set_gradient_mask(ou15_mask, 'ou15_ar')
        

        max_reward = 10000
        max_epoch = -1
        for epoch in range(args.TST_epoches):
            print('{} epoch TST finetuning..'.format(epoch + 1))
            model.train()
            MultiTaskLearning(model, train_dataloader_hu,
                              train_dataloader_ou5,
                              train_dataloader_ou10,
                              train_dataloader_ou15,
                              train_dataloader_mol)
            mol_linkandrel_loss, _ = model.compute_f1_and_loss_reward(tasktype='parsing',
                                                                      eval_dataloader=eval_dataloader_mol)
            print('eval mol link loss {}'.format(mol_linkandrel_loss))
            total_eval_loss = mol_linkandrel_loss
            print('total eval loss {}'.format(total_eval_loss))
            if total_eval_loss < max_reward:
                torch.save(model.state_dict(), args.TST_model_path + '.pt')
                max_reward = total_eval_loss
                max_epoch = epoch


    else:
        test_dataloader_mol = DataLoader(dataset=test_dataset_mol, batch_size=args.eval_mol_pool_size,
                                         shuffle=False,
                                         collate_fn=eval_collate_fn_mol)

        model = PolicyNetwork(args=args, pretrained_model=pretrained_model)
        model = model.to(args.device)
        state_dict = torch.load(args.TST_model_path+'.pt')
        model.load_state_dict(state_dict,strict=False)
        model.eval()
        total_loss, total_f1 = model.compute_f1_and_loss_reward(tasktype='parsing',
                                                          eval_dataloader=test_dataloader_mol)
