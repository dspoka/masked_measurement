import hydra
from hydra import utils as hydra_utils
import os, sys
import socket
from saver import Saver
from logger import LOGGER
from omegaconf import DictConfig,OmegaConf
import torch
from termcolor import colored
from pathlib import Path
from transformers import RobertaTokenizerFast
# from models import GenNumSIUnit
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel
from transformers import RobertaModel
import json
import numpy as np
import copy
import pickle
from tqdm import tqdm
import signal
from torch import nn
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader

def mask_sent(text, match):
    '''Used for a single wiki-convert sentence
        input:  Some cars drive 200 mph fast on race tracks.
        output: Some cars drive [#NUM] [#UNIT] fast on race tracks.
    '''
    num_start = match['num_start'] 
    num_end = match['num_end'] 
    unit_start = match['unit_start'] 
    unit_end = match['unit_end'] 
    assert num_end < unit_start
    s_unit_token = UNIT_TOKEN
    s_num_token = NUM_TOKEN
    
    # mask_out_unit=True and mask_out_dimension=True
    text = text[:unit_start] + s_unit_token + text[unit_end:]
    
    # mask out num (depend on flag)
    text = text[:num_start] + s_num_token + text[num_end:]
    return text

class BaseUnitTransformer(RobertaPreTrainedModel):
    def __init__(self, config, args, class_buckets, dimID_2_derivedID):
        super().__init__(config)#Roberta
        self.sem_hz = args.sem_hz
        self.do_weighted = args.do_weighted
        self.c_hz = config.hidden_size #hidden_dim output of roberta
        self.n_y1 = len(class_buckets['y1']) #|D| = num_rows
        self.do_zsl = args.model_zsl
        self.mask_out_number = args.mask_out_number
        self.mask_out_unit = args.mask_out_unit
        self.mask_out_dimension = args.mask_out_dimension
        self.args = args
        self.roberta = RobertaModel(config, add_pooling_layer=True)
        self.dimID_2_derivedID = dimID_2_derivedID
        self.num_derived = sum([len(v) for k,v in dimID_2_derivedID.items()])
        self.do_derived = args.do_derived
        self.supervised_derived = args.supervised_derived

    def set_special_ids(self, tokenizer):
        self.unit_token_id = tokenizer.convert_tokens_to_ids(UNIT_TOKEN)
        self.num_token_id = tokenizer.convert_tokens_to_ids(NUM_TOKEN)
        self.mask_token_id = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
        self.out_vocab_size = self.config.vocab_size - 2

    def set_unit_mask(self):
        weights = torch.zeros(self.n_y1, self.num_derived)
        for k,v in self.dimID_2_derivedID.items():
            if k < self.n_y1:
                for derived_id in v:
                    weights[k,derived_id] = 1.0

        # print(weights.sum(dim=1))[ 3., 10.,  6.,  6., 38.,  2.,  2.]
        self.unit_mask = torch.nn.Embedding.from_pretrained(weights, freeze=True)

    def transformer_forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled_output = outputs[1] #[B,c_hz]
        return pooled_output
    
    def model_forward(self, inputs, gzsl):
        pooled_output = self.transformer_forward(**inputs)
        outputs = self.model_loss(pooled_output, gzsl, **inputs)
        return outputs['total_loss']

    def evaluate(self, inputs, gzsl, analysis=False):
        pooled_output = self.transformer_forward(**inputs)
        outputs = self.model_loss(pooled_output, gzsl, do_eval=True, analysis=analysis,**inputs)
        return outputs

    def forward(self, inputs, mode, gzsl):
        if mode == 'loss':
            return self.model_forward(inputs, gzsl)
        elif mode == 'eval':
            return self.evaluate(inputs, gzsl)
        elif mode == 'analysis':
            return self.evaluate(inputs, gzsl, analysis=True)
    
    def model_loss(self, pooled_output, gzsl, do_eval, **kwargs):
        raise NotImplementedError

class GenNumSIUnit(BaseUnitTransformer):
    # 3. p(x,D|C) = p(D|C)*p(x|C,D): mask number and mask unit
    # 4. p(x,D,U|C) = p(D|C)*p(U|C,D)*p(x|U,C,D): mask number and mask unit
    # +add unit table in discrimnantive way
    # eval: numeracy. Unknown dim: argmax P(D|C) -> d
    # argmax p(x|C,d=D): MAE
    # ---- given D=d
    # argmax p(x|C,d=D): MAE

    def __init__(self, config, args, class_buckets, dimID_2_derivedID):
        super().__init__(config, args, class_buckets, dimID_2_derivedID)
        # self.proj_c = torch.nn.Linear(self.c_hz, self.d_hz)
        if self.do_weighted:
            weights = torch.tensor([1, 3.02,   88.4,  18.1,   79.4,  20.4,  24.3])
            self.dim_crossentropy = nn.CrossEntropyLoss(reduction='none', weight=weights)
        else:
            self.dim_crossentropy = nn.CrossEntropyLoss(reduction='none')
        self.crossentropy = nn.CrossEntropyLoss(reduction='none')
        self.num_criterion = torch.nn.L1Loss(reduction='none')
        self.proj_semantic = torch.nn.Linear(self.c_hz, self.sem_hz)
        
        if self.do_zsl:
            self.unit_embedder = UnitEmbedder(args, class_buckets, self.sem_hz)
            self.W_num = torch.nn.Linear(self.sem_hz*2, self.n_y1)#todo: judy
        else:
            self.si_classifier = torch.nn.Linear(self.sem_hz, self.n_y1)
            self.W_num = torch.nn.Linear(self.sem_hz, self.n_y1)
        
        if self.do_derived:
            self.derived_classifier = torch.nn.Linear(self.sem_hz, self.num_derived)
            self.derived_W_num = torch.nn.Linear(self.sem_hz, self.num_derived)


    def model_loss(self, pooled_output, gzsl, do_eval=False, analysis=False, label_dim=None, label_num=None, tgt_dimensions=None, tgt_derived=None,label_rec=None, **kwargs):
    # def forward(self, texts, tgt_dimensions, tgt_numbers=None):
        """
        Args:
            texts (Tensor): input tensor of tokens [Batch, Seq-Len]
            tgt_dimensions (Tensor): True index of dimension or SI-unit since its 1 to 1 correspondence
            tgt_numbers (Tensor, optional): Tensor of float values for each sentence. Defaults to None.
            label_num (Tensor): True numbers for each example
            tgt_derived (Tensor, ): True derived units for each example

        Returns:
            Dict: 
        """
        # texts [B,S]
        # tgt_dimensions [B]
        ################SI_Units################
        #P(D|C)
        sem_context = self.proj_semantic(pooled_output)
        if self.do_zsl:
            si_logits = self.unit_embedder(sem_context, gzsl=gzsl) #[B, num_si]
        else:
            si_logits = self.si_classifier(sem_context) #[B, num_si]
        si_loss = self.dim_crossentropy(si_logits, tgt_dimensions) #[B]
        si_log_ll = -si_loss
        ########################################
        
        ################ Derived Units ################
        if self.do_derived:
            ###### p(U|D,C) ######
            
            # Condition on C
            derived_logits = self.derived_classifier(sem_context) #[B, num_derived]
            
            if self.supervised_derived:
                # Condition on D
                unit_mask = self.unit_mask(tgt_dimensions) #[B, num_derived]            
                masked_derived_logits = derived_logits - ((1 - unit_mask) * 1e10) #[B, num_derived] 
                derived_loss = self.crossentropy(masked_derived_logits, tgt_derived) #[B]
            else:
                # Get negative log-likelihoods
                derived_loss = self.crossentropy(derived_logits, tgt_derived) #[B]

            # Get log-likelihoods
            derived_log_ll = -derived_loss #[B]
        ########################################

        ################number###################

        if self.do_derived:
            # p(x|UDC) = p(x|UC)

            # Condition on C
            d_pred_mu = self.derived_W_num(sem_context) #[B, num_derived]
            
            # Condition on U
            tgt_derived = tgt_derived.view(-1, 1) #[B,1]
            pred_mu = d_pred_mu.gather(1, tgt_derived) #[B,1]
            pred_mu = pred_mu.squeeze(dim=1) #[B]
            
            distance = self.num_criterion(torch.log(label_num), pred_mu)
            num_loss = distance
            constant = torch.log(torch.abs(1/label_num))
            num_log_ll = -distance + constant
        else:
            # p(X|D,C)
            if self.do_zsl:
                # todo: decide how to parametrize this
                d_pred_mu = self.unit_embedder(pooled_output, gzsl) #[B, num_si]
            else:
                d_pred_mu = self.W_num(sem_context) #[B, num_si]
            # tgt_dimensions = tgt_dimensions.view(-1, 1) #[B,1]
            pred_mu = d_pred_mu.gather(1, tgt_dimensions.view(-1, 1)) #[B,1]
            pred_mu = pred_mu.squeeze(dim=1) #[B]
            distance = self.num_criterion(torch.log(label_num), pred_mu) # input and target are switched but this shouldn't matter
            num_loss = distance
            constant = torch.log(torch.abs(1/label_num))
            num_log_ll = -distance + constant
        ########################################
            
            
        if self.do_derived:
            log_ll = num_log_ll + si_log_ll + derived_log_ll #[B]
            total_loss = -log_ll #[B]
            outputs = {'total_loss': total_loss, 'si_loss': si_loss, 'num_loss': num_loss, 'U_loss': derived_loss}
        else:
            log_ll = num_log_ll + si_log_ll #[B]
            total_loss = -log_ll #[B]
            outputs = {'total_loss': total_loss, 'si_loss': si_loss, 'num_loss': num_loss}
        
        ################# Evaluate #################
        if do_eval:
            outputs['classification'] = {}
            outputs['regression'] = {}
            # +: means marginalized
            # *: means argmax

            # Get D*
            pred_si = torch.argmax(si_logits, dim=1) #[B]
            # tgt_dimensions = tgt_dimensions.squeeze(dim=1) #[B]
            outputs['classification']['D|C'] = {'pred': pred_si, 'logits': si_logits, 'true': tgt_dimensions}
                        
            if self.do_derived:
                # U|DC, U|XDC = ground truth D

                if self.supervised_derived:
                    tgt_derived = tgt_derived.squeeze(dim=1) #[B]
                    pred_derived = torch.argmax(masked_derived_logits, dim=1) #[B]
                    outputs['classification']['U|DC'] = {'pred': pred_derived, 'logits': masked_derived_logits, 'true': tgt_derived}
                
                    derived_logits_u_given_xdc = self.calc_u_given_dxc(d_pred_mu, masked_derived_logits, label_num)
                    pred_derived_given_xdc = torch.argmax(derived_logits_u_given_xdc, dim=1) #[B]
                    outputs['classification']['U|XDC'] = {'pred': pred_derived_given_xdc, 'logits': derived_logits_u_given_xdc, 'true': tgt_derived}
                
                    outputs['regression']['X|UDC'] = {'pred': torch.exp(pred_mu), 'true': label_num}
                
                # Condition on D*. pred_si is D* in p(X|U*D*C)
                unit_mask = self.unit_mask(pred_si) #[B, num_derived]        
                derived_logits = derived_logits - ((1 - unit_mask) * 1e10) #[B, num_derived]
                
                # This is U* in p(X|U*D*C)
                pred_derived_argmax_d = torch.argmax(derived_logits, dim=1) #[B]

                # print('pred_derived_argmax_d', pred_derived_argmax_d.min(), pred_derived_argmax_d.max(), pred_derived_argmax_d.size())
                # print('d_pred_mu', d_pred_mu.min(), d_pred_mu.max(), d_pred_mu.size())
                
                # Condition on U*
                # d_pred_mu [B, num_derived]
                pred_mu = d_pred_mu.gather(1, pred_derived_argmax_d.view(-1, 1)) #[B,1]
                pred_mu = pred_mu.squeeze(dim=1) #[B]
                outputs['regression']['X|U*D*C'] = {'pred': torch.exp(pred_mu), 'true': label_num}
                
                # outputs['regression']['X|U+DC']
                # outputs['regression']['X|U*DC']
                # outputs['regression']['X|UDC'] = {'pred': torch.exp(pred_mu), 'true': label_num}
                
                # p(U|C) = sigma_D p(UD|C) = sigma_D p(U|DC) * p(D|C)
                # p(U|DXC) proportional to p(X|UDC) * P(U|DC)

                # outputs['classification']['U|D*C'] = {'pred': , 'logits': , 'true': tgt_derived}
                # outputs['classification']['U|D+C'] = {'pred': , 'logits': , 'true': tgt_derived}          
            else:
                outputs['regression']['X|DC'] = {'pred': torch.exp(pred_mu), 'true': label_num} 

                # p(D|XC) proportional to p(X|CD) * P(D|C)            
                si_logits_d_given_xc = self.calc_d_given_xc(d_pred_mu, si_logits, label_num) #[B,num_si]
                if torch.any(si_logits_d_given_xc != si_logits_d_given_xc):
                    print('si_logits_d_given_xc', si_logits_d_given_xc.size())
                    print(si_logits_d_given_xc.min(), si_logits_d_given_xc.max())
                    foohere
                pred_si_given_xc = torch.argmax(si_logits_d_given_xc, dim=1) #[B]
                outputs['classification']['D|XC'] = {'pred': pred_si_given_xc, 'logits': si_logits_d_given_xc, 'true': tgt_dimensions}                      

                pred_mu = d_pred_mu.gather(1, pred_si.view(-1, 1)) #[B,1]
                pred_mu = pred_mu.squeeze(dim=1) #[B]
                outputs['regression']['X|D*C'] = {'pred': torch.exp(pred_mu), 'true': label_num}
                
        # Store semantic context for t-SNE visualization
        if analysis:     
            outputs['sem_context'] = sem_context            

        return outputs

    def calc_u_given_c():
        # TODO
        pass

    def calc_u_given_dxc(self, d_pred_mu, derived_logits, tgt_numbers):
        # derived_logits [B, num_derived] P(U|DC)
        # d_pred_mu [B, num_derived] P(X|UC)
        # tgt_numbers [B]

        num_derived = derived_logits.size()[1]
        tgt_numbers = tgt_numbers.repeat(num_derived, 1).permute(1,0)#[B,num_derived]

        distance = self.num_criterion(torch.log(tgt_numbers), d_pred_mu)#[B,num_derived]
        num_loss = distance
        constant = torch.log(torch.abs(1/tgt_numbers))#[B,num_derived]

        num_log_ll = -distance + constant #[B,num_derived]

        log_softmax = torch.nn.LogSoftmax(dim=1)
        derived_log_ll = log_softmax(derived_logits) #[B,num_derived]
        log_ll = num_log_ll + derived_log_ll #[B, num_derived]
        # print('derived_log_ll', derived_log_ll.size())
        # print('num_log_ll', num_log_ll.size())
        return log_ll

    def calc_d_given_xc(self, d_pred_mu, si_logits, tgt_numbers):
        # print('d_pred_mu', d_pred_mu.size()) #[B,num_si]
        # print('si_logits', si_logits.size()) #[B,num_si]
        # print('tgt_numbers', tgt_numbers.size()) #[B]
        num_si = si_logits.size()[1]
        tgt_numbers = tgt_numbers.repeat(num_si, 1).permute(1,0)#[B,num_si]
        # if torch.any(tgt_numbers != tgt_numbers):
        #     print('nan-tgt_numbers', tgt_numbers.size())
        # print('tgt_numbers', tgt_numbers.size(), tgt_numbers.min(), tgt_numbers.max())
        distance = self.num_criterion(torch.log(tgt_numbers), d_pred_mu)#[B,num_si]
        # if torch.any(distance != distance):
        #     print('nan-distance', distance.size())
        # print('distance', distance.size())
        num_loss = distance
        constant = torch.log(torch.abs(1/tgt_numbers))#[B,num_si]
        # if torch.any(constant != constant):
        #     print('nan-constant', constant.size())
        # print('constant', constant.size())

        num_log_ll = -distance + constant #[B,num_si]
        # print('num_log_ll', num_log_ll.size())#[B,num_si]
        log_softmax = torch.nn.LogSoftmax(dim=1)
        si_log_ll = log_softmax(si_logits) #[B,num_si]
        # if torch.any(si_log_ll != si_log_ll):
        #     print('nan-si_log_ll', si_log_ll.size())
        # print('si_log_ll', si_log_ll.size())
        log_ll = num_log_ll + si_log_ll #[B, num_si]
        # if torch.any(log_ll != log_ll):
        #     print('nan-log_ll', log_ll.size())
        # print('log_ll', log_ll.size())
        return log_ll

def seed(args=None, num=None):
    if args is None:
        n_seed = num
    else:
        n_seed = args.seed
    np.random.seed(n_seed)
    torch.manual_seed(n_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(n_seed)
        torch.backends.cudnn.deterministic = True



def get_paths():
    set_paths = {
        'dataset_path':'/home/dspokoyn/links/datasets/units',
        'unece_path':'/projects/dspokoyn/units/assets',
        'wiki_convert_path':'/projects/judyjin/datasets/wiki-convert/zsl_splits_2_way_manual',
        'load_root': '/trunk/dspokoyn/units',
        'save_root': '/trunk/dspokoyn/units'
    }
	return set_paths

def build_roberta(all_cfg, sys_argv, tokenizer, class_buckets, dimID_2_derivedID, train_dataset):
    # model = RobertaForProbabilisticSequenceClassification.from_pretrained('roberta-base', args=all_cfg, return_dict=True)
    #model = NumUnitRoberta.from_pretrained('roberta-base', args=all_cfg, return_dict=True)
    if all_cfg.model_name == 'DownGen':
        foohere
        model = DownGen.from_pretrained('roberta-base', args=all_cfg, return_dict=True, class_buckets=class_buckets, dimID_2_derivedID=dimID_2_derivedID)
    elif all_cfg.model_name == 'Gen':
        model = GenNumSIUnit.from_pretrained("dspoka/units-gen-d-u", use_auth_token=True, args=all_cfg, return_dict=True, class_buckets=class_buckets, dimID_2_derivedID=dimID_2_derivedID)
        model.set_unit_mask()
        sum_ = 0
        # print()
        # for k,v in model.named_parameters():
        #     if 'roberta' not in k:
        #         print(k,v.shape)
        #     sum_ += torch.sum(v)
        # print('sum', sum_)
        # print()

        # model = GenNumSIUnit.from_pretrained('roberta-base', args=all_cfg, return_dict=True, class_buckets=class_buckets, dimID_2_derivedID=dimID_2_derivedID)
    else:
        print('Error model_name not defined')

    if all_cfg.do_derived and all_cfg.supervised_derived:
        # this is here because these models get initialized after they are built
        model.set_unit_mask()
    
    model.resize_token_embeddings(len(tokenizer))
    model.set_special_ids(tokenizer)    
    optimizer = torch.optim.AdamW(model.parameters(), lr=all_cfg.optim.lr)

    seed(num=all_cfg.optim.seed)
    return model, optimizer

def build_name(all_cfg, sys_args):    
    model_name = ''
    if all_cfg.debug:
        model_name += 'debug-'

    model_name += all_cfg.downstream
    model_name += all_cfg.model_name
    
    model_name += f'sem:{all_cfg.sem_hz}-'
    
    lr = all_cfg.optim.lr
    model_name += f'-lr:{lr}_'

    model_name += f'sd:{all_cfg.optim.seed}'
    return model_name



def wiki_load_deduped(all_cfg, split_type=None):
    '''Loads the wiki-convert data without any duplicates
    Note: Name Change from load_wiki_clean_mode to wiki_load_deduped
    '''
    load_dir = '/trunk/dspokoyn/datasets/climate/raw/wikicon'
    if all_cfg.debug:
        debug_ = 'debug_'
    else:
        debug_ = ''
    load_path = os.path.join(load_dir, f'{debug_}wikicon_deduped.pkl')
    # check if file exists
    if not os.path.exists(load_path):
        return None, None, None

    with open(load_path, 'rb') as f:
        data = pickle.load(f)
    all_texts = data['texts']
    all_masked_texts = data['masked_texts']
    all_metadata = data['metadata']
    if split_type is not None:
        split_texts = []
        split_masked_texts = []
        split_metadata = []
        for datum in all_metadata:
            if datum['split'] == split_type:
                split_texts.append(all_texts[datum['corpus_id']])
                split_masked_texts.append(all_masked_texts[datum['corpus_id']])
                split_metadata.append(datum)
        all_texts = split_texts
        all_masked_texts = split_masked_texts
        all_metadata = split_metadata
    
    for text,metadata in zip(all_texts, all_metadata):
        metadata['comment'] = text
    
    all_texts = all_texts[:1024]
    all_masked_texts = all_masked_texts[:1024]
    all_metadata = all_metadata[:1024]

    return all_texts, all_masked_texts, all_metadata

class TimeoutError(Exception):
    pass

class timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message
    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)
    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)
    def __exit__(self, type, value, traceback):
        signal.alarm(0)

def filter_2_nums(train_seen):
    re_number = r'(?:0|(?:[1-9](?:\d*|\d{0,2}(?:,\d{3})*)))+(?:\.\d*)?(?!\S)'    
    import re
    re_number = re.compile(re_number)
    data = []
    num_with_2_nums = 0
    filtered_data = []
    for sent in tqdm(train_seen):
        try:
            with timeout(seconds=1):
                num_matches = 0
                for match in re_number.finditer(sent['comment']):
                    try:
                        float(match.group())
                    except:
                        print(match.group())
                    # print(dir(match))
                    # print(match.span())
                    # foohere
                    num_matches += 1
                if num_matches >= 2:
                    num_with_2_nums += 1
                if num_with_2_nums % 10000 == 0:
                    print(num_with_2_nums)
        except TimeoutError:
            pass
        if num_matches >= 2:
            filtered_data.append(sent)
    return filtered_data

def load_model(all_cfg):
    all_cfg.paths.root = hydra_utils.get_original_cwd()
    paths = get_paths()
    all_cfg.paths.dataset_path = paths['dataset_path']
    all_cfg.paths.unece_path = paths['unece_path']
    all_cfg.paths.save_root = paths['save_root']
    all_cfg.paths.load_root = paths['load_root']
    all_cfg.paths.wiki_convert_path = paths['wiki_convert_path']
    
    model_name = build_name(all_cfg, sys.argv)

    print(OmegaConf.to_yaml(all_cfg))
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # device = torch.device('cpu')



    # train_seen_dataset,val_seen_dataset,val_unseen_dataset,test_seen_dataset,test_unseen_dataset, class_buckets, tokenizer, dimID_2_derivedID = setup_wiki_task(all_cfg, device)
    # print('class_buckets', class_buckets)
    # print('dimID_2_derivedID', dimID_2_derivedID)
    
    class_buckets = {'y1': [0, 1, 2, 3, 4, 5, 6], 'y2': [7, 8, 9, 10, 11, 12], 'id': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12}}
    dimID_2_derivedID = {6: [3, 34], 0: [17, 52, 76, 67, 22, 37, 59, 4, 9, 0], 5: [61, 79, 81, 6, 20], 1: [28, 25, 78, 43, 5], 3: [47, 13, 60, 14, 41, 49, 32, 58, 57, 11, 63, 21, 53, 64, 56, 44, 54, 42, 36, 24, 38, 50, 80, 70, 55, 68, 66, 71, 19, 82, 29, 1, 27, 73, 40, 15, 7], 4: [77], 2: [2, 48], 9: [72], 7: [45, 39], 8: [33, 30, 26, 35, 18, 75, 62, 31, 10], 10: [12, 16, 74, 69], 11: [65, 51, 23, 8], 12: [46]}
    train_seen_dataset = None
    # tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    tokenizer = AutoTokenizer.from_pretrained("dspoka/units-gen-d-u", use_auth_token=True)

    model, optimizer = build_roberta(all_cfg, sys.argv, tokenizer, class_buckets, dimID_2_derivedID, train_seen_dataset)
    model = model.to(device)

    all_cfg.paths.save_path = os.path.join(all_cfg.paths.save_root, all_cfg.paths.save_dir, all_cfg.downstream, model_name)
    if all_cfg.mode_inference:
        all_cfg.paths.load_path = os.path.join(all_cfg.paths.load_root, all_cfg.paths.load_path, model_name)
    
    LOGGER.set(os.path.join(all_cfg.paths.save_path, 'log.txt'), reopen_to_flush=True)
    saver = Saver({"model": model, "optimizer": optimizer}, keep_every_n=all_cfg.optim['keep_every_n'])
    
    if all_cfg.model_name == 'DownGen':
        #todo Down has to do this but not Gen
        model.init_downstream()
        
    model.to(device)
    return model, class_buckets, dimID_2_derivedID, tokenizer, device

def has_num(word):
    for char in word:
        if char in ['0','1','2','3','4','5','6','7','8','9']:
            return True
    return False

def load_vocab():
    f = open('vocab.json')
    data = json.load(f)
    for datum in data:
        if has_num(datum):
            # print(datum)
            if datum in ['0','1','2','3','4','5','6','7','8','9']:
                print('\n\n', datum)

@hydra.main(config_path="conf", config_name="config")
def main(all_cfg: DictConfig):
    model, class_buckets, dimID_2_derivedID, tokenizer, device = load_model(copy.deepcopy(all_cfg.pretrained))
    print('loaded-model')
    # derived_2_id, dimID_2_derivedID, id_2_derived
    
    train_text, _, train_metadata  = wiki_load_deduped(all_cfg, split_type='train_seen')
    val_text, _, val_metadata  = wiki_load_deduped(all_cfg, split_type='val_seen')
    all_text, _, all_metadata  = wiki_load_deduped(all_cfg, split_type=None)
    print('val-text', val_text[0])
    print('metadata', val_metadata[0])
    train_metadata = filter_2_nums(train_metadata)
    train_seen_dataset = WikiConvertDataset(copy.deepcopy(all_cfg.pretrained), train_metadata, tokenizer, class_buckets, 'train', device)
    
    train_loader = DataLoader(train_seen_dataset, batch_size=all_cfg['optim']['eval_batch_size'], shuffle=False, drop_last=False, num_workers=0, pin_memory=True)
    with torch.no_grad():
        for batch in train_loader:
            batch = {k:v.to(device) for k,v in batch.items()}
            metrics = batch_infer(batch, model, tokenizer, device)
            b_size = batch['input_ids'].size()[0]
            for i in range(b_size):
                for k,_ in metrics.items():
                    print(metrics[k][i])
                print()
            foohere
    # sent = 'My sailboat is 10 feet long and weights [#NUM] [#UNIT].'
    # number = 10
    # dim_id = dim_id_2_name['Mass']
    # unit_id = derived_2_id['tonnes']
    
    metrics = infer(sent, number, dim_id, unit_id, model, tokenizer, device)
    for k,v in metrics.items():
        print(k, v)



def decode_input_ids(input_ids, tokenizer):
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    tokens = [t for t in tokens if t != tokenizer.pad_token]
    return tokens

def batch_infer(batch, model, tokenizer, device):
    # label_num = torch.tensor(number, dtype=torch.long).to(device)
    # tgt_dimensions = torch.tensor(dim, dtype=torch.long).to(device)
    # tgt_derived = torch.tensor(unit, dtype=torch.long).to(device)
    # encodings = tokenizer(sent, padding='max_length', max_length=64)
    print('batch', batch.keys())
    
    outputs = model.evaluate(inputs=batch, gzsl=None)
    metrics = {}

    # outputs['classification']['D|C']['pred']
    metrics['D|C'] = outputs['classification']['D|C']['pred']
    metrics['U|DC'] = outputs['classification']['U|DC']['pred']
    metrics['U|XDC'] = outputs['classification']['U|XDC']['pred']
    metrics['X|UDC'] = outputs['regression']['X|UDC']['pred']
    metrics['X|U*D*C'] = outputs['regression']['X|U*D*C']['pred']
    
    print('input_ids', batch['input_ids'].size())
    decoded_text = []
    for sent in batch['input_ids']:
        tokens = decode_input_ids(sent, tokenizer)
        clean_sent = tokenizer.convert_tokens_to_string(tokens)
        decoded_text.append(clean_sent)
    # decoded_text = tokenizer.convert_tokens_to_string([decode_input_ids(sent, tokenizer) for sent in batch['input_ids']])

    # print(metrics['D|C']['pred'])
    # print(metrics['U|DC']['pred'][0].size())
    metrics['dim_pred'] = [dim_id_2_name[m.item()] for m in metrics['D|C']]
    # = dim_id_2_name[metrics['D|C'][0].item()]

    # metrics['unit_pred'] = id_2_derived[metrics['U|DC'][0].item()]
    metrics['unit_pred'] = [id_2_derived[m.item()] for m in metrics['U|DC']]
    metrics['text'] = decoded_text
    #  = {'pred': torch.exp(pred_mu), 'true': label_num}
    return metrics

def infer(sent, number, dim, unit, model, tokenizer, device):    
    label_num = torch.tensor(number, dtype=torch.long).to(device)
    tgt_dimensions = torch.tensor(dim, dtype=torch.long).to(device)
    tgt_derived = torch.tensor(unit, dtype=torch.long).to(device)
    encodings = tokenizer(sent, padding='max_length', max_length=64)

    item = {key: torch.as_tensor(val).to(device) for key, val in encodings.items()}
    item['label_num'] = label_num
    item['tgt_dimensions'] = tgt_dimensions
    item['tgt_derived'] = tgt_derived
    for k,v in item.items():
        item[k] = v.unsqueeze(0)
    
    
    outputs = model.evaluate(inputs=item, gzsl=None)
    metrics = {}
    print(outputs['classification'].keys())
    # outputs['classification']['D|C']['pred']
    metrics['D|C'] = outputs['classification']['D|C']['pred']
    #  = {'pred': pred_si, 'logits': si_logits, 'true': tgt_dimensions}
    # outputs['classification']['U|DC']['pred']
    metrics['U|DC'] = outputs['classification']['U|DC']['pred']
    #  = {'pred': pred_derived, 'logits': masked_derived_logits, 'true': tgt_derived}
    # outputs['classification']['U|XDC']['pred']
    metrics['U|XDC'] = outputs['classification']['U|XDC']['pred']
    #  = {'pred': pred_derived_given_xdc, 'logits': derived_logits_u_given_xdc, 'true': tgt_derived}
    # outputs['regression']['X|UDC']['pred']
    metrics['X|UDC'] = outputs['regression']['X|UDC']['pred']
    metrics['X|U*D*C'] = outputs['regression']['X|U*D*C']['pred']
    
    decoded_text = tokenizer.convert_tokens_to_string(decode_input_ids(item['input_ids'][0], tokenizer))
    
    
    # dim_id_2_name = ['Length', 'Area', 'Temperature', 'Velocity', 'Electric charge', 'Mass', 'Power','Volume', 'Mass flow rate', 'Capacitance', 'Kinematic viscosity', 'Volume flow rate', 'Dimensionless']
    # print(metrics['D|C']['pred'])
    # print(metrics['U|DC']['pred'][0].size())
    metrics['dim_pred'] = dim_id_2_name[metrics['D|C'][0].item()]
    metrics['unit_pred'] = id_2_derived[metrics['U|DC'][0].item()]
    metrics['text'] = decoded_text

    return metrics

def load_fixed_assets():
    #### derived unit string to integer id ###
    derived_2_id = {'kilometers per pixel': 0, 'miles per year': 1, '°F': 2, 'kilowatts': 3, 'yards': 4, 'sqft': 5, 'kilograms': 6, 'feet per week': 7, 'litres per hour': 8, 'meters per pixel': 9, 'kilograms per day': 10, 'meters per day': 11, 'sqmi per year': 12, 'kmph': 13, 'mm per year': 14, 'mm per hour': 15, 'acres per year': 16, 'meters': 17, 'grams per day': 18, 'cms per month': 19, 'long tons': 20, 'miles per hour': 21, 'mm': 22, 'litres per day': 23, 'cms per minute': 24, 'hectares': 25, 'pounds per year': 26, 'miles per second': 27, 'sqmi': 28, 'miles per month': 29, 'pounds per hour': 30, 'tonnes per hour': 31, 'meters per year': 32, 'pounds per day': 33, 'horsepower': 34, 'kilograms per year': 35, 'feet per year': 36, 'feet': 37, 'yards per minute': 38, 'litres': 39, 'cms per second': 40, 'cms per year': 41, 'kilometers per hour': 42, 'acres': 43, 'meters per second': 44, 'cc': 45, 'feet per mile': 46, 'mph': 47, '°C': 48, 'feet per minute': 49, 'miles per day': 50, 'litres per minute': 51, 'miles': 52, 'inches per hour': 53, 'mm per day': 54, 'cms per hour': 55, 'kilometers per second': 56, 'feet per hour': 57, 'inches per year': 58, 'inches': 59, 'knots': 60, 'pounds': 61, 'tonnes per day': 62, 'feet per day': 63, 'feet per second': 64, 'litres per second': 65, 'kilometers per year': 66, 'kilometers': 67, 'inches per day': 68, 'sqkm per year': 69, 'meters per minute': 70, 'cms per day': 71, 'Farad': 72, 'mm per month': 73, 'hectares per year': 74, 'tonnes per year': 75, 'cms': 76, 'Coulomb': 77, 'sqkm': 78, 'grams': 79, 'inches per month': 80, 'tonnes': 81, 'kilometers per day': 82}
    id_2_derived = {item:key for key,item in derived_2_id.items()}
    ### derived unit string id to train label ###
    derivedID_2_dimID = {3: 6, 17: 0, 52: 0, 76: 0, 61: 5, 28: 1, 34: 6, 67: 0, 22: 0, 79: 5, 47: 3, 13: 3, 25: 1, 37: 0, 78: 1, 59: 0, 43: 1, 60: 3, 5: 1, 81: 5, 6: 5, 4: 0, 77: 4, 20: 5, 2: 2, 14: 3, 48: 2, 41: 3, 49: 3, 32: 3, 58: 3, 57: 3, 11: 3, 63: 3, 9: 0, 21: 3, 53: 3, 64: 3, 56: 3, 44: 3, 54: 3, 42: 3, 36: 3, 24: 3, 38: 3, 50: 3, 80: 3, 70: 3, 55: 3, 68: 3, 66: 3, 71: 3, 19: 3, 72: 9, 45: 7, 39: 7, 33: 8, 30: 8, 12: 10, 65: 11, 16: 10, 26: 8, 46: 12, 35: 8, 18: 8, 74: 10, 75: 8, 62: 8, 31: 8, 69: 10, 51: 11, 23: 11, 8: 11, 10: 8, 82: 3, 29: 3, 0: 0, 1: 3, 27: 3, 73: 3, 40: 3, 15: 3, 7: 3}
    
    dimID_2_derivedID = {}
    for k,v in derivedID_2_dimID.items():
        if v in dimID_2_derivedID:
            dimID_2_derivedID[v].append(k)
        else:
            dimID_2_derivedID[v] = [k]
    
    dim_name_2_dim_id= {'Length':0, 'Area':1, 'Temperature':2, 'Velocity':3, 'Electric charge':4, 'Mass':5, 'Power':6,'Volume':7, 'Mass flow rate':8, 'Capacitance':9, 'Kinematic viscosity':10, 'Volume flow rate':11, 'Dimensionless':12}
    dim_id_2_name = {item:key for key,item in dim_name_2_dim_id.items()}

    return derived_2_id, dimID_2_derivedID, id_2_derived, dim_id_2_name

class WikiConvertDataset(Dataset):
    def __init__(self, all_cfg, data, tokenizer, class_buckets, split, device):
        """[summary]

        Args:
            all_cfg ([type]): [description]
            data ([type]): [description]
            tokenizer ([type]): [description]
            device ([type]): [description]
        """     
        
        self.split = split
        self.class_buckets = class_buckets
        self.class_2_id = self.class_buckets['id']
        self.data = data
        self.all_cfg = all_cfg
        
        # self.mask_out_number = all_cfg.mask_out_number
        # self.mask_out_unit = all_cfg.mask_out_unit
        # self.mask_out_dimension = all_cfg.mask_out_dimension

        # self.mode_inference = all_cfg.mode_inference
        # self.tokenizer = tokenizer
        self.device = device
        self.max_seq_length = all_cfg.data_cfg['max_seq_length']
        self.s_unit_token = UNIT_TOKEN
        self.s_num_token = NUM_TOKEN
        if all_cfg.do_derived:
            self.do_derived = True
            self.derived_2_id, self.dimID_2_derivedID, _ ,_ = load_fixed_assets()
        else: 
            self.do_derived = True
        self.do_num_in_SI = all_cfg.do_num_in_SI
        self.prep(tokenizer)
        

    def __len__(self):
        return len(self.masked_text)
    

    def get_base(self,num_unit):
        label = num_unit['base']

    def prep(self, tokenizer):
        self.masked_text = []
        self.label_num = []
        self.label_dim = []
        self.tgt_derived = []
        max_num = 0
        min_num = 100000000
        count = 0
        count_neg = 0
        len_text = []
        len_token = []
        
        for datum in self.data:
            text = datum['comment']
            base_unit = datum['base_unit']
            num_start = datum['offset']
            num_end = num_start+ datum['length']
            if self.do_num_in_SI:
                num_SI = float(datum['number_SI'])
            else:
                #not SI here just for naming
                num_SI = float(datum['number'])
            unit_start = datum['unit_start']
            unit_end = unit_start +datum['unit_length']
            dim_label = datum['train_label']
            
            if self.all_cfg.do_derived:
                derived_label = self.derived_2_id[datum['unit']]
        
            # tgt_derived = datum[]
            match = {'num_start':num_start,'num_end':num_end,'unit_start':unit_start,'unit_end':unit_end}

            if num_SI < 1e-6:
                # todo: skips negative number
                count_neg+=1
                continue
            
            
            masked_sent = mask_sent(text, match)
            tokens = tokenizer(masked_sent)['input_ids']
            if len(tokens) > self.max_seq_length:
                continue

            max_num = max(max_num, num_SI)
            min_num = min(min_num, num_SI)
            len_text.append(len(text))
            len_token.append(len(tokens))            
                
            self.label_num += [num_SI]
            self.masked_text.append(masked_sent)
            self.label_dim += [dim_label]
            if self.all_cfg.do_derived:
                self.tgt_derived += [derived_label]

                # self.label_deribed +=

            # if num_SI <0:
            #     count_neg +=1
                # print(text)

        self.encodings = tokenizer(self.masked_text, padding='max_length', max_length=self.max_seq_length)
        # short_text.sort(key = len)
        # # # print(short_text)
        # print('negative', count_neg)
        # print('shorten to', self.max_seq_length, ":",count)
        # print(len(self.label_dim) + count)
        print('average_len', np.mean(len_text))
        print('median_len', np.median(len_text))
        print('average_token_len', np.mean(len_token))
        print('median_token_len', np.median(len_token))
        print('max_num', max_num)
        print('min_num', min_num)
        print('self.label_dim', set(self.label_dim))

        
    
    def __getitem__(self, idx):
        item = {key: torch.as_tensor(val[idx]) for key, val in self.encodings.items()}
        # item['tgt_dimensions'] = torch.as_tensor(self.label_dim[idx], dtype=torch.int64) 
        item['tgt_dimensions'] = torch.as_tensor(self.class_2_id[self.label_dim[idx]], dtype=torch.int64) 
        item['label_num'] = torch.as_tensor(self.label_num[idx], dtype=torch.float32)
        if self.all_cfg.do_derived:
            item['tgt_derived'] = torch.as_tensor(self.tgt_derived[idx], dtype=torch.int64)
        
        # item = {key: torch.tensor(val[idx], device=self.device) for key, val in self.encodings.items()}
        # item['tgt_dimensions'] = torch.tensor(self.label_dim[idx], dtype=torch.int64, device=self.device) 
        # item['label_num'] = torch.tensor(self.label_num[idx], dtype=torch.float32, device=self.device)
        return item


derived_2_id, dimID_2_derivedID, id_2_derived, dim_id_2_name = load_fixed_assets()
UNIT_TOKEN = '[#UNIT]'
NUM_TOKEN = '[#NUM]'

if __name__ == '__main__':
    main()
    # load_model()
    # load_vocab()
