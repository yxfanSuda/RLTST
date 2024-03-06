import json
from tqdm import tqdm

class Hu_File:
    def __init__(self):
        self.output_dir = './Hu_Dataset/Hu_AR_dir/'
        self.train_file = './Hu_Dataset/train.json'
        self.valid_file = './Hu_Dataset/dev.json'
        self.test_file = './Hu_Dataset/test.json'

class Ou5_File:
    def __init__(self):
        self.output_dir = './Ou_Dataset/Ou5_AR_dir/'
        self.train_file = './Ou_Dataset/5_train.json'
        self.valid_file = './Ou_Dataset/5_dev.json'
        self.test_file = './Ou_Dataset/5_test.json'

class Ou10_File:
    def __init__(self):
        self.output_dir = './Ou_Dataset/Ou10_AR_dir/'
        self.train_file = './Ou_Dataset/10_train.json'
        self.valid_file = './Ou_Dataset/10_dev.json'
        self.test_file = './Ou_Dataset/10_test.json'

class Ou15_File:
    def __init__(self):
        self.output_dir = './Ou_Dataset/Ou15_AR_dir/'
        self.train_file = './Ou_Dataset/15_train.json'
        self.valid_file = './Ou_Dataset/15_dev.json'
        self.test_file = './Ou_Dataset/15_test.json'

def load_dataset(fname):
    dataset = []
    with open(fname, 'r') as f:
        for line in f:
            data = json.loads(line)
            ctx = data['context']
            ctx_spk = data['ctx_spk']
            ctx_adr = data['ctx_adr']
            rsp = data['answer']
            rsp_spk = data['ans_spk']
            rsp_adr = data['ans_adr']
            integrate_ctx = ctx + [rsp]
            integrate_ctx_spk = ctx_spk + [rsp_spk]
            integrate_ctx_adr = ctx_adr + [rsp_adr]
            assert len(integrate_ctx) == len(integrate_ctx_spk)
            assert len(integrate_ctx) == len(integrate_ctx_adr)
            label = []
            for utr_id_adr, utr_adr in enumerate(integrate_ctx_adr):
                label_utr = [0 for _ in range(len(integrate_ctx))]
                for cand_utr_id_spk, cand_utr_spk in enumerate(integrate_ctx_spk[:utr_id_adr]):  # consider only the preceding utterances
                    if cand_utr_spk == utr_adr:
                        label_utr[cand_utr_id_spk] = 1
                label.append(label_utr)
            dataset.append((ctx, ctx_spk, rsp, rsp_spk, label))
    print("dataset_size: {}".format(len(dataset)))
    return dataset

def create_examples(lines, set_type, data_type):
    examples = []
    for (i, line) in enumerate(lines):
        guid = "%s-%s-%s" % (data_type, set_type, str(i))
        ctx =line[0]
        ctx_spk = line[1]
        rsp = line[2]
        rsp_spk = line[3]
        label = line[-1]
        examples.append([guid, ctx, ctx_spk, rsp, rsp_spk, label])
    return examples

def write_des(data, des_file):
    with open(des_file, 'w', encoding='utf8') as fw:
        for da in tqdm(data):
            fw.write(json.dumps(da,ensure_ascii=False)+'\n')


class Convert_ARlabel2_Link:
    def __init__(self):
        pass

    def load_json(self, file):
        with open(file,'r',encoding='utf8')as fr:
            lines  = fr.readlines()
        datas = [ ]
        for line in lines:
            line = line.strip()
            datas.append(json.loads(line))
        return datas

    def convert_labelMatrix_2_relations(self, LabelMatrix, multi_edge=False):
        relations = []
        for j, templabel in enumerate(LabelMatrix):
            i_list = []
            for i, label in enumerate(templabel):
                if label ==1:
                    i_list.append(i)
            if i_list:
                if multi_edge:
                    for last_i in i_list:
                        relations.append({'type': "Result",
                                          'x': last_i,
                                          'y': j})
                else:
                    last_i = i_list[-1]
                    relations.append({'type':"Result",
                                      'x': last_i,
                                      'y': j})
        return relations

    def write_des(self, source_file, des_file,multi_edge = False):
        """

        """
        source_datas = self.load_json(source_file)
        des_datas = []
        for da in tqdm(source_datas):
            # print(da)
            id = da[0]
            edus = da[1]+[da[3]]
            speakers = da[2]+[da[4]]
            LabelMatrix = da[5]
            relations = self.convert_labelMatrix_2_relations(LabelMatrix, multi_edge = multi_edge)
            edu_speakers = []
            for edu, speaker in zip(edus, speakers):
                temp_dic = {'speaker': str(speaker),
                            'text': edu}
                edu_speakers.append(temp_dic)
            des_datas.append({'id':id,
                              'edus': edu_speakers,
                              'relations': relations})
        with open(des_file,'w',encoding='utf8')as fw:
            json.dump(des_datas, fw, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    
    # hu = Hu_File()
    # ou5 = Ou5_File()
    # ou10 = Ou10_File()
    ou15 = Ou15_File()
    # for dataset_file in [hu, ou5, ou10, ou15]:
    dataset_name ='ou15'
    for dataset_file in [ou15]:
        filenames = [dataset_file.train_file, dataset_file.valid_file, dataset_file.test_file]
        filetypes = ["train"]
        for (filename, filetype) in zip(filenames, filetypes):
            dataset = load_dataset(filename)
            examples = create_examples( dataset, filetype, dataset_name)
            new_filename = dataset_file.output_dir + "/{}_ar.json".format(filetype)
            write_des(examples, des_file=new_filename)
    
    conaddParsingType  = Convert_ARlabel2_Link()
    # conaddParsingType.write_des('./Hu_Dataset/Hu_AR_dir/valid_ar.json',
    #                             './Hu_Dataset/Hu_Link_Dir/valid.json')

    conaddParsingType.write_des('./Ou_Dataset/Ou15_AR_dir/train_ar.json',
                                './Ou_Dataset/Ou15_Link_Dir/train.json')

    # conaddParsingType.write_des('./Ou_Dataset/Ou10_AR_dir/train_ar.json',
    #                             './Ou_Dataset/Ou10_Link_Dir/train.json')

    # conaddParsingType.write_des('./Ou_Dataset/Ou15_AR_dir/train_ar.json',
    #                             './Ou_Dataset/Ou15_Link_Dir/train.json')
