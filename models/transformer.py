from os import name
from transformers import RobertaTokenizer, RobertaModel
# from transformers.modeling_roberta import RobertaPreTrainedModel
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel
# from .unit_embedder import UnitEmbedder
# from ..wiki_convert_dataset import get_unit_conv_factors_as_tensor


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

    return derived_2_id, dimID_2_derivedID

def get_unit_conv_factors():    
    ureg  = pint.UnitRegistry()
    unit_id_2_si_conv_factor = {}    
    unit_name_2_unit_id, _ = load_fixed_assets()

    pint_fail = {"sqmi":"square miles", "sqkm":"square kilometers",
    "cms":"centimeters","Coulomb":"coulomb","sqft":"square feet",
    "Farad":"farad","long tons": "tons"}

    for unit, unit_id in unit_name_2_unit_id.items():
        # print(unit)
        for fail_unit, fixed_unit in pint_fail.items():
            if fail_unit in unit:
                unit = unit.replace(fail_unit, fixed_unit)
                # print('replaced with', unit)
        unit_id_2_si_conv_factor[unit_id] = ureg(unit).to_base_units().magnitude

    return unit_id_2_si_conv_factor

def get_unit_conv_factors_as_tensor():
    unit_id_2_si_conv_factor = get_unit_conv_factors()
    unit_id_2_si_conv_factor_as_tensor = torch.ones(len(unit_id_2_si_conv_factor))
    for unit_id, si_conv_factor in unit_id_2_si_conv_factor.items():
        unit_id_2_si_conv_factor_as_tensor[unit_id] = si_conv_factor
    return unit_id_2_si_conv_factor_as_tensor


import torch
import torch.nn as nn
from torch import Tensor
# todo:
# !!! set somewhere
UNIT_TOKEN = '[#UNIT]'
NUM_TOKEN = '[#NUM]'

""" class UnitEmbedder(nn.Module):
    def __init__(self, n_unit, hz):
        super(UnitEmbedder, self).__init__()
        self.unit_embedding_table = torch.nn.Embedding(n_unit, hz)
    
    def forward(self, pooled_output, si_logits=None, number=None):
        # print('pooled_output', pooled_output.size())
        # print('table', self.unit_embedding_table.weight.size())
        si_logits = torch.einsum('bz,sz->bs', pooled_output, self.unit_embedding_table.weight)
        return si_logits """


class RobertaForProbabilisticSequenceClassification(RobertaPreTrainedModel):
    authorized_missing_keys = [r"position_ids"]

    def __init__(self, config, args):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        

        self.hz = config.hidden_size
        self.classifier = nn.Sequential(torch.nn.Linear(self.hz, 1))
        # pos_weight = torch.tensor([1., 12.])
        # pos_weight = torch.tensor([10.])
        self.pos_weight = torch.tensor([args.optim.pos_weight])
        self.criterion = nn.BCEWithLogitsLoss(reduction='none', pos_weight=self.pos_weight)
        # self.criterion = nn.BCELoss(reduction='none')
        self.init_weights()

    def forward(self, inputs, mode):
        if mode == 'loss':
            return self.model_loss(**inputs)
        elif mode == 'eval':
            #return all metrics
            return self.evaluate(inputs)
        # elif mode == 'pred':
        #     #return all argmaxes
        #     return self.argmax(**inputs)

    def evaluate(self, inputs):
        loss, logits = self.model_loss(**inputs)        
        return loss, logits

    def model_loss(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
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
        sequence_output = outputs[0]
        cls_output = sequence_output[:,0,:]

        logits = self.classifier(cls_output).squeeze()
        loss = self.criterion(logits, labels)
        return loss, logits

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



class Majority(nn.Module):
    def __init__(self, config, args, train_dataset, class_buckets):
        super().__init__()
        self.n_y1 = len(class_buckets['y1'])
        self.n_y2 = len(class_buckets['y2'])
        #self.n_y3 = len(class_buckets['y3'])
        
        print('y1', class_buckets['y1'])
        print('y2', class_buckets['y2'])
        tgt_dimensions = [d['tgt_dimensions'] for d in train_dataset]
        label_num = [d['label_num'] for d in train_dataset]
        # print('label_unit', label_unit, type(label_unit))
        # print('label_num', label_num, type(label_num))
        tgt_dimensions = torch.stack(tgt_dimensions)
        # gzsl = y1
        bins_y1 = torch.bincount(tgt_dimensions, minlength=self.n_y1)
        self.si_logits_y1 = torch.zeros_like(bins_y1, requires_grad=False)
        values, indices = torch.topk(bins_y1, 5)
        for i,idx in enumerate(indices):
            self.si_logits_y1[idx] = 10-i #[num_si]
        self.pred_si_y1 = torch.argmax(self.si_logits_y1) #[1]
        
        # gzsl = y1+y2
        bins_y1y2 = torch.bincount(tgt_dimensions, minlength=self.n_y1+self.n_y2)
        self.si_logits_y1y2 = torch.zeros_like(bins_y1y2, requires_grad=False)
        values, indices = torch.topk(bins_y1y2, 5)
        for i,idx in enumerate(indices):
            self.si_logits_y1y2[idx] = 10-i #[num_si]
        self.pred_si_y1y2 = torch.argmax(self.si_logits_y1y2) #[1]
        """ # gzsl = y1+y3
        bins_y1y3 = torch.bincount(tgt_dimensions, minlength=self.n_y1+self.n_y3)
        self.si_logits_y1y3 = torch.zeros_like(bins_y1y3, requires_grad=False)
        values, indices = torch.topk(bins_y1y3, 5)
        for i,idx in enumerate(indices):
            self.si_logits_y1y3[idx] = 10-i #[num_si]
        self.pred_si_y1y3 = torch.argmax(self.si_logits_y1y3) #[1] """


        self.fake_param = torch.nn.Embedding(3,7)#so that optimizer doesnt complain
        # print('si_logits', self.si_logits.size(), self.si_logits)
        # print('pred_si', self.pred_si.size(), self.pred_si)

        # print('label_unit', label_unit.size())
        # print('label_num', label_num.size())
        label_num = torch.stack(label_num)
        self.median = torch.median(label_num)
        print('median', self.median)
    
    def forward(self, inputs, mode, gzsl):
        assert mode == 'eval'
        if mode == 'eval':
            return self.evaluate(inputs, gzsl)
    
    def evaluate(self, inputs, gzsl):
        # pooled_output = self.transformer_forward(**inputs)
        pooled_output = None
        outputs = self.model_loss(**inputs, gzsl=gzsl)
        return outputs
    
    def resize_token_embeddings(self, *args):
        pass
    
    def set_special_ids(self, *args):
        pass
    
    def model_loss(self, label_num, tgt_dimensions, gzsl, **kwargs):
        # sequence_output = outputs[0] #[b_size,sequence lengh,hz]   
        # texts [B,S]
        # dimensions [B]
        # pooled = self.roberta(texts) #[B,self.c_hz]
        assert label_num.size()[0] == tgt_dimensions.size()[0]
        b_size = label_num.size()[0]
        
        if gzsl == 'y1':
            num_si = self.n_y1
            si_logits = self.si_logits_y1
            pred_si = self.pred_si_y1
        elif gzsl == 'y1+y2':
            num_si = self.n_y1 + self.n_y2
            si_logits = self.si_logits_y1y2
            pred_si = self.pred_si_y1y2
        """ elif gzsl == 'y1+y3':
            num_si = self.n_y1 + self.n_y3
            si_logits = self.si_logits_y1y3
            pred_si = self.pred_si_y1y3 """

        si_logits = si_logits.repeat(b_size,1)
        pred_si = pred_si.repeat(b_size,1)
        pred_si = pred_si.squeeze(dim=1)
        # print('pred', pred_si.size())
        # print('label', tgt_dimensions.size())
        assert tgt_dimensions.size() == pred_si.size()

        pred_mu = self.median.repeat(b_size, 1)
        pred_mu = pred_mu.squeeze(dim=1)
        # print('pred_mu', pred_mu.size())
        # print('label_num', label_num.size())
        assert pred_mu.size() == label_num.size()
        
        outputs = {}
        outputs['classification'] = {}
        # print('tgt_dimensions', tgt_dimensions.min(), tgt_dimensions.max(), num_si)
        assert tgt_dimensions.min() >= 0 and tgt_dimensions.max() < num_si
        # print('si_logits', si_logits.size())
        outputs['classification']['majority'] = {'pred': pred_si, 'logits': si_logits, 'true': tgt_dimensions}
        # todo: add random classification.
        
        rand_logits = torch.randn(b_size, num_si)
        pred_rand_logits = torch.argmax(rand_logits, dim=1)
        outputs['classification']['random'] = {'pred': pred_rand_logits, 'logits': rand_logits, 'true': tgt_dimensions}
        
        outputs['regression'] = {}
        outputs['regression']['median'] = {'pred': pred_mu, 'true': label_num}

        outputs['total_loss'] = torch.tensor([0]).float() # Add so run_zeroshot.py doesn't complain
        return outputs
            
class DiscriminativeNumSIUnit(BaseUnitTransformer):
    # 1. p(D|C) # mask-num
    # 2. p(D|C,x) # do-not mask num!
    # +add unit table in discrimnantive way
    # use unit table in ZSL

    def __init__(self, config, args, class_buckets, dimID_2_derivedID):
        super().__init__(config, args, class_buckets, dimID_2_derivedID)
        print('config', config)
        # self.config.vocab_size
        # self.proj_c = torch.nn.Linear(self.c_hz, self.d_hz)
        # semantic_hz: is space for labels and zsl
        if self.do_weighted:
            weights = torch.tensor([1, 3.02,   88.4,  18.1,   79.4,  20.4,  24.3])
            self.dim_crossentropy = nn.CrossEntropyLoss(reduction='none', weight=weights)
        else:
            self.dim_crossentropy = nn.CrossEntropyLoss(reduction='none')
        self.crossentropy = nn.CrossEntropyLoss(reduction='none')

        #semantic projection layer
        self.proj_semantic = torch.nn.Linear(self.c_hz, self.sem_hz)
        
        if self.do_zsl:
            self.unit_embedder = UnitEmbedder(args, class_buckets, self.sem_hz)
        else:
            self.si_classifier = torch.nn.Linear(self.sem_hz, self.n_y1)

        if self.do_derived:
            self.derived_classifier = torch.nn.Linear(self.sem_hz, self.num_derived)


    def model_loss(self, pooled_output, gzsl, do_eval=False, label_dim=None, label_num=None, tgt_dimensions=None, tgt_derived=None, label_rec=None, **kwargs):
        # sequence_output = outputs[0] #[b_size,sequence lengh,hz]   
        # texts [B,S]
        # dimensions [B]
        # pooled = self.roberta(texts) #[B,self.c_hz]
        sem_context = self.proj_semantic(pooled_output)#[B,sem_hz]
        if self.do_zsl:
            si_logits = self.unit_embedder(sem_context, gzsl=gzsl)
            # todo: in ZSL setting for eval this should be different!
        else:
            si_logits = self.si_classifier(sem_context) #[B, num_si]
        
        si_loss = self.dim_crossentropy(si_logits, tgt_dimensions) #[B]
        # si_log_ll = -si_loss
        total_loss = si_loss
        
        outputs = {}
        if self.do_derived:
            derived_logits = self.derived_classifier(sem_context) #[B, num_derived]
            if self.supervised_derived:
                unit_mask = self.unit_mask(tgt_dimensions) #[B, num_derived]
                derived_logits = derived_logits - ((1 - unit_mask) * 1e10)
            derived_loss = self.crossentropy(derived_logits, tgt_derived)
            total_loss += derived_loss
            outputs.update({'U_loss': derived_loss})
            
        outputs.update({'total_loss': total_loss, 'si_loss': si_loss})

        if do_eval:

            # Argmax to get most likely dimension D
            pred_si = torch.argmax(si_logits, dim=1) #[B]

            if self.mask_out_number:
                outputs['classification'] = {'D|C': {'pred': pred_si, 'logits': si_logits, 'true':tgt_dimensions}}
            else:
                outputs['classification'] = {'D|XC': {'pred': pred_si, 'logits': si_logits, 'true':tgt_dimensions}}
            
            if self.do_derived:

                # Argmax to get most likely derived unit U
                pred_derived = torch.argmax(derived_logits, dim=1)

                if self.supervised_derived:
                    if self.mask_out_number:
                        outputs['classification']['U|DC'] = {'pred': pred_derived, 'logits': derived_logits, 'true':tgt_derived}
                    else:
                        outputs['classification']['U|XDC'] = {'pred': pred_derived, 'logits': derived_logits, 'true':tgt_derived}
                else:
                    if self.mask_out_number:
                        outputs['classification']['U|C'] = {'pred': pred_derived, 'logits': derived_logits, 'true':tgt_derived}
                    else:
                        outputs['classification']['U|XC'] = {'pred': pred_derived, 'logits': derived_logits, 'true':tgt_derived}

        return outputs
        
    def predict(self, texts):
        # argmax return most likely D
        outputs = self.forward(texts)
        si_logits = outputs['si_logits'] #[B,num_si]
        pred_si = torch.argmax(si_logits, dim=1) #[B]
        return pred_si
    
class ProbeDiscriminative(DiscriminativeNumSIUnit):
    def __init__(self, config, args, class_buckets, dimID_2_derivedID):
        super().__init__(config, args, class_buckets, dimID_2_derivedID)
        self.freeze_params()
    
    def freeze_params(self):
        for param in self.parameters():
            param.requires_grad = False
        
        for param in self.proj_semantic.parameters():
            param.requires_grad = True

        for param in self.si_classifier.parameters():
            param.requires_grad = True

        if self.do_derived:
            for param in self.derived_classifier.parameters():
                param.requires_grad = True
            # self.derived_classifier.requires_grad = True
        
        for k,v in self.named_parameters():
            if v.requires_grad:
                print(k,v.shape, v.requires_grad)


class NumBERT(BaseUnitTransformer):
    # - m1. number-is masket.
    # - 1st. unit is not masked. 2nd. unit is masked.
    # - only LMAE metric.
    def __init__(self, config, args, class_buckets, dimID_2_derivedID):
        super().__init__(config, args, class_buckets, dimID_2_derivedID)

        assert self.mask_out_number == True
        # assert self.supervised_derived == True# this is not needed i think
        # self.crossentropy = nn.CrossEntropyLoss(reduction='none')
        #semantic projection layer
        self.proj_semantic = torch.nn.Linear(self.c_hz, self.sem_hz)
        self.num_criterion = torch.nn.L1Loss(reduction='none')
        self.W_num = torch.nn.Linear(self.sem_hz, 1)
        
        self.do_dimension = args.do_dimension
        self.do_derived = args.do_derived

        if self.do_weighted:
            # weights = torch.tensor([1, 3.02,   88.4,  18.1,   79.4,  20.4,  24.3])
            #log weighted with epsilon = 1.05
            weights = torch.tensor([1, 2.25, 9.65, 6.489, 9.51, 6.80, 7.24])
            self.dim_crossentropy = nn.CrossEntropyLoss(reduction='none', weight=weights)
        else:
            self.dim_crossentropy = nn.CrossEntropyLoss(reduction='none')

        if self.do_dimension:
            self.dim_classifier = torch.nn.Linear(self.sem_hz, self.n_y1)
        
        if self.do_derived:
            self.derived_classifier = torch.nn.Linear(self.sem_hz, self.num_derived)
            self.crossentropy = nn.CrossEntropyLoss(reduction='none')

    def model_loss(self, pooled_output, gzsl, do_eval=False, label_dim=None, label_num=None, tgt_dimensions=None, tgt_derived=None, label_rec=None, **kwargs):

        sem_context = self.proj_semantic(pooled_output)#[B,sem_hz]
        

        pred_mu = self.W_num(sem_context) #[B,1]
        pred_mu = pred_mu.squeeze(dim=1) #[B]
        distance = self.num_criterion(torch.log(label_num), pred_mu)
        num_loss = distance
        total_loss = num_loss
        outputs = {'num_loss': num_loss}
        
        if self.do_dimension:
            dim_logits = self.dim_classifier(sem_context) #[B, num_si]
            dim_loss = self.dim_crossentropy(dim_logits, tgt_dimensions) #[B]
            # dim_log_ll = -dim_loss
            outputs['dim_loss'] = dim_loss
            total_loss += dim_loss

        
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
            outputs['derived_loss'] = derived_loss
            total_loss += derived_loss
        ########################################
        
        
        outputs['total_loss'] = total_loss
        
        if do_eval:
            outputs['regression'] = {}
            if self.do_dimension:
                pred_dim = torch.argmax(dim_logits, dim=1) #[B]
                outputs['classification'] = {'D|C': {'pred': pred_dim, 'logits': dim_logits, 'true':tgt_dimensions}}
                
            if self.do_derived:
                if self.supervised_derived:
                    # tgt_derived = tgt_derived.squeeze(dim=1) #[B]
                    pred_derived = torch.argmax(masked_derived_logits, dim=1) #[B]
                    outputs['classification']['U|DC'] = {'pred': pred_derived, 'logits': masked_derived_logits, 'true': tgt_derived}

                    logits_u_given_xdc = self.calc_u_given_xdc_genA(pred_mu, derived_logits, label_num, tgt_dimensions, tgt_derived)
                    pred_u_given_xdc = torch.argmax(logits_u_given_xdc, dim=1) #[B]
                    outputs['classification']['U|XDC'] = {'pred': pred_u_given_xdc, 'logits': logits_u_given_xdc, 'true': tgt_derived}

                logits_d_given_xc = self.calc_d_given_xc_genA(pred_mu, dim_logits, derived_logits, label_num, tgt_derived)
                pred_d_given_xc = torch.argmax(logits_d_given_xc, dim=1) #[B]
                outputs['classification']['D|XC'] = {'pred': pred_d_given_xc, 'logits': logits_d_given_xc, 'true': tgt_dimensions}

            
            if self.mask_out_unit and self.mask_out_dimension:
                outputs['regression']['X|C'] = {'pred': torch.exp(pred_mu), 'true': label_num}
            elif self.mask_out_unit:
                outputs['regression']['X|DC'] = {'pred': torch.exp(pred_mu), 'true': label_num}
            else:
                outputs['regression']['X|UC'] = {'pred': torch.exp(pred_mu), 'true': label_num}

        return outputs


    def calc_d_given_xc_genA(self, X_predict, D_logits, U_logits, X_target, U_target):
        """
        X = true X (non-SI)
        Calculate p(D|X,C) proportional to p(D|C) * p(X|D,C)
                                         = p(D|C) * sum_U p(U,X|D,C)
                                         = p(D|C) * sum_U p(U|D,C) p(X|U,D,C)
                                         = p(D|C) * sum_U p(U|D,C) p(X|U,C)

        Args:
            X_predict: log mu (mean) for p(X|U,C)   (shape: [B])
            D_logits:  logits for p(D|C)            (shape: [B, D])
            U_logits:  logits for p(U|D,C)          (shape: [B, U])
            X_target:  ground truth numbers in SI   (shape: [B])
            U_target:  ground truth units           (shape: [B])
        Returns:
            log_ll: p(D|X,C) (shape: [B, D])
        """
        B = D_logits.size()[0]
        nD = D_logits.size()[1]
        nU = U_logits.size()[1]

        # p(D|C)        
        D_log_softmax = torch.nn.LogSoftmax(dim=1)
        D_log_ll = D_log_softmax(D_logits) #[B,D]

        # p(U|D,C)        
        unit_mask = torch.arange(nD, device=X_target.device) # [D]
        unit_mask = self.unit_mask(unit_mask) # [D, U]
        assert(unit_mask.size() == (nD, nU))
        unit_mask = unit_mask.unsqueeze(0).repeat(B, 1, 1) # [B,D,U]

        U_logits = U_logits.unsqueeze(1).repeat(1, nD, 1) # [B,D,U]
        U_logits = U_logits - ((1 - unit_mask) * 1e10) # [B,D,U]
        U_log_softmax = torch.nn.LogSoftmax(dim=2)
        U_log_ll = U_log_softmax(U_logits) #[B,D,U]
        assert(U_log_ll.size() == (B, nD, nU))

        # p(X|U,C)        
        # Map from unit ID to SI conversion factor
        si_conv_factor = get_unit_conv_factors_as_tensor() #[U]
        si_conv_factor = si_conv_factor.to(device=U_logits.device)        
        si_conv_factor = si_conv_factor.unsqueeze(dim=0).repeat(B, 1) #[B,U]
        assert(si_conv_factor.size() == (B, nU))

        # Convert X_predict to non-SI
        X_predict = X_predict.unsqueeze(dim=1).repeat(1, nU) #[B,U]
        X_predict = torch.exp(X_predict) / si_conv_factor #[B, U]
        X_predict = torch.log(X_predict) #[B, U]

        # Use U_target to recover original X_target before it was converted to SI
        x_target_conv_factor = si_conv_factor.gather(1, U_target.view(-1, 1)).squeeze(1) #[B]
        X_target = X_target / x_target_conv_factor #[B]
        X_target = X_target.unsqueeze(dim=1).repeat(1, nU) #[B,U]
        
        distance = self.num_criterion(torch.log(X_target), X_predict) #[B,U]
        constant = torch.log(torch.abs(1/X_target)) #[B,U]
        X_log_ll = -distance + constant #[B,U]
        X_log_ll = X_log_ll.unsqueeze(dim=1).repeat(1, nD, 1) #[B,D,U]

        # sum_U p(U|D,C) p(X|U,C)
        XU_log_ll = U_log_ll + X_log_ll #[B,D,U]
        XU_log_ll = torch.logsumexp(XU_log_ll, dim=2) #[B,D]

        # p(D|X,C)
        log_ll = D_log_ll + XU_log_ll #[B,D]
        return log_ll


    def calc_u_given_xdc_genA(self, X_predict, U_logits, X_target, D_target, U_target):
        """
        X = true X (non-SI)
        Calculate p(U|X,D,C) proportional to p(U|D,C) * p(X|U,D,C)
                                           = p(U|D,C) * p(X|U,C)

        Args:
            X_predict: log mu (mean) for p(X|U,C)   (shape: [B])
            U_logits:  logits for p(U|D,C)          (shape: [B, U])
            X_target:  ground truth numbers in SI   (shape: [B])
            D_target:  ground truth dimensions      (shape: [B])
            U_target:  ground truth units           (shape: [B])
        Returns:
            log_ll:                                 (shape: [B, U])
        """
        B = U_logits.size()[0]
        nU = U_logits.size()[1]

        # p(U|D,C)
        unit_mask = self.unit_mask(D_target) #[B, U]
        U_logits = U_logits - ((1 - unit_mask) * 1e10) #[B, U]
        U_log_softmax = torch.nn.LogSoftmax(dim=1)
        U_log_ll = U_log_softmax(U_logits) #[B,U]


        # p(X|U,C)
        # Map from unit ID to SI conversion factor
        # si_conv_factor = torch.arange(nU, device=U_logits.device) # [U]
        # si_conv_factor = self.unit_conv_factor_2_SI(si_conv_factor) #[U] #TODO embedding layer [U] -> [1]
        si_conv_factor = get_unit_conv_factors_as_tensor() #[U]
        si_conv_factor = si_conv_factor.to(device=U_logits.device)        
        si_conv_factor = si_conv_factor.unsqueeze(dim=0).repeat(B, 1) #[B,U]
        assert(si_conv_factor.size() == (B, nU))

        # Convert X_predict to non-SI
        X_predict = X_predict.unsqueeze(dim=1).repeat(1, nU) #[B,U]
        X_predict = torch.exp(X_predict) / si_conv_factor #[B, U]
        X_predict = torch.log(X_predict) #[B, U]
        

        # Use U_target to recover original X_target before it was converted to SI
        x_target_conv_factor = si_conv_factor.gather(1, U_target.view(-1, 1)).squeeze(1) #[B]
        assert(x_target_conv_factor.size() == (B,))
        # x_target_conv_factor = self.unit_conv_factor_2_SI(U_target) #[B]
        X_target = X_target / x_target_conv_factor #[B]
        X_target = X_target.unsqueeze(dim=1).repeat(1, nU) #[B,U]
        
        distance = self.num_criterion(torch.log(X_target), X_predict) #[B,U]
        constant = torch.log(torch.abs(1/X_target)) #[B,U]
        X_log_ll = -distance + constant #[B,U]

        # p(U|X,D,C)
        log_ll = U_log_ll + X_log_ll #[B,U]
        return log_ll


class ProbeNumBERT(NumBERT):
    def __init__(self, config, args, class_buckets, dimID_2_derivedID):
        super().__init__(config, args, class_buckets, dimID_2_derivedID)
        self.freeze_params()
    
    def freeze_params(self):
        for param in self.parameters():
            param.requires_grad = False
        
        for param in self.proj_semantic.parameters():
            param.requires_grad = True
        
        for param in self.W_num.parameters():
            param.requires_grad = True
        
        if self.do_dimension:
            for param in self.dim_classifier.parameters():
                param.requires_grad = True
        
        if self.do_derived:
            for param in self.derived_classifier.parameters():
                param.requires_grad = True

        for k,v in self.named_parameters():
            if v.requires_grad:
                print(k,v.shape, v.requires_grad)

# class SharedGenNumSIUnit(BaseUnitTransformer):
#     def __init__(self, config, args, class_buckets, dimID_2_derivedID):
#         super().__init__(config, args, class_buckets, dimID_2_derivedID)
#         # self.proj_c = torch.nn.Linear(self.c_hz, self.d_hz)
#         assert self.do_zsl == False
#         if self.do_weighted:
#             weights = torch.tensor([1, 3.02,   88.4,  18.1,   79.4,  20.4,  24.3])
#             self.dim_crossentropy = nn.CrossEntropyLoss(reduction='none', weight=weights)
#         else:
#             self.dim_crossentropy = nn.CrossEntropyLoss(reduction='none')

#         self.crossentropy = nn.CrossEntropyLoss(reduction='none')
#         self.num_criterion = torch.nn.L1Loss(reduction='none')
#         self.proj_semantic = torch.nn.Linear(self.c_hz, self.sem_hz)
#         self.num_heads = args.num_heads
#         self.attn_1 = nn.MultiheadAttention(self.sem_hz, self.num_heads, batch_first=True)
#         self.attn_2 = nn.MultiheadAttention(self.sem_hz, self.num_heads, batch_first=True)

#         #todo: ivan this is broken:
#         # self.dim_embedder = UnitEmbedder(args, class_buckets, self.sem_hz)
#         self.dim_embedder = torch.nn.Embedding(self.n_y1, self.sem_hz)
#         self.unit_embedder = torch.nn.Embedding(self.num_derived, self.sem_hz)

#         self.W_D = torch.nn.Linear(self.sem_hz, self.n_y1)
#         self.W_y = torch.nn.Linear(2*self.sem_hz, 1)
        
#         if self.do_derived:
#             self.W_cat = torch.nn.Linear(2*self.sem_hz, self.sem_hz)
#             self.W_U = torch.nn.Linear(self.sem_hz, self.num_derived)

#     def model_loss(self, pooled_output, gzsl, do_eval=False, label_dim=None, label_num=None, tgt_dimensions=None, tgt_derived=None, **kwargs):
#         B_size = pooled_output.size()[0]
#         sem_context = self.proj_semantic(pooled_output) #[B,sem_hz]
        
#         h1 = sem_context
#         dim_ind = torch.arange(self.n_y1, device=h1.device)
#         dim_emb = self.dim_embedder(dim_ind) #[sem_hz,D]
#         # print('dim_emb', dim_emb.size())
#         batch_dim_emb = dim_emb.repeat(B_size, 1, 1) #[B,D,sem_hz]
#         # print('batch_dim_emb', batch_dim_emb.size())
#         Bh1D = torch.cat([h1.unsqueeze(dim=1), batch_dim_emb], dim=1) #[B,1+D, sem_hz]
#         # print('Bh1D', Bh1D.size())
#         Bh2, _ = self.attn_1(Bh1D,Bh1D,Bh1D) #[B,1+D, sem_hz]
#         # print('Bh2', Bh2.size())
#         h2 = Bh2[:,0,:] #[B,sem_hz]
#         # print('h2', h2.size())
        
#         D_logits = self.W_D(h2) #[B,D]
#         D_loss = self.dim_crossentropy(D_logits, tgt_dimensions) #[B]
#         D_log_ll = -D_loss
        
#         if self.do_derived:
#             unit_ind = torch.arange(self.num_derived, device=h1.device)
#             unit_emb = self.unit_embedder(unit_ind) #[sem_hz,U]
#             batch_unit_emb = unit_emb.repeat(B_size, 1, 1) #[B,U,sem_hz]
#             Bh2U = torch.cat([h2.unsqueeze(dim=1), batch_unit_emb], dim=1) #[B,1+U, sem_hz]
#             # print('Bh2U', Bh2U.size())
#             Bh3, _ = self.attn_2(Bh2U,Bh2U,Bh2U) #[B,1+U, sem_hz]
#             # print('Bh3', Bh3.size())
#             h3 = Bh3[:,0,:] #[B,sem_hz]
#             # print('h3', h3.size())
#             # !Condition on TRUE D
#             true_dim_emb = self.dim_embedder(tgt_dimensions) #[B,sem_hz]
#             h4 = self.W_cat(torch.cat([h3, true_dim_emb], dim=1))
#             # print('h4', h4.size())
#             U_logits = self.W_U(h4) #[B,U]

#             # !Condition on TRUE D
#             unit_mask = self.unit_mask(tgt_dimensions) #[B, num_derived]            
#             masked_U_logits = U_logits - ((1 - unit_mask) * 1e10) #[B, num_derived] 
#             U_loss = self.crossentropy(masked_U_logits, tgt_derived) #[B]
#             # Get log-likelihoods
#             U_log_ll = -U_loss #[B]

#             # !Condition on TRUE U
#             true_unit_emb = self.unit_embedder(tgt_derived) #[B,sem_hz]
#             # print('true_unit_emb', true_unit_emb.size())
#             pred_y = self.W_y(torch.cat([h4, true_unit_emb], dim=1))

#         else:            
#             # !Condition on TRUE D
#             true_dim_emb = self.dim_embedder(tgt_dimensions) #[B,sem_hz]
#             pred_y = self.W_y(torch.cat([h2, true_dim_emb], dim=1)) #[B,1]
        
#         pred_y = pred_y.squeeze(dim=1) #[B]
#         # print('pred_y', pred_y.size())
#         distance = self.num_criterion(torch.log(label_num), pred_y)
#         num_loss = distance
#         constant = torch.log(torch.abs(1/label_num))
#         num_log_ll = -distance + constant
        
#         outputs = {}
#         total_loss = D_loss + num_loss
#         if self.do_derived:
#             total_loss = total_loss + U_loss
#             outputs['U_loss'] = U_loss
#         outputs = {'total_loss': total_loss, 'num_loss': num_loss, 'si_loss': D_loss}
        
#         # print('total_loss', total_loss.size())
#         # print('U_loss', U_loss.size())
#         # print('num_loss', num_loss.size())
#         # print('D_loss', D_loss.size())
        
#         # inference mode
#         if do_eval:
#             #only 1 D_Pred
#             D_pred = torch.argmax(D_logits, dim=1) #[B]  
            
#             if self.do_derived:
#                 U_pred = torch.argmax(masked_U_logits, dim=1) #[B]

#                 # !Condition on PRED D
#                 pred_dim_emb = self.dim_embedder(D_pred) #[B,sem_hz]
#                 inf_h4 = self.W_cat(torch.cat([h3, pred_dim_emb], dim=1))
#                 # print('inf_h4', inf_h4.size())
#                 inf_U_logits = self.W_U(inf_h4) #[B,U]

#                 # !Condition on PRED D
#                 unit_mask = self.unit_mask(D_pred) #[B, num_derived]            
#                 masked_inf_U_logits = inf_U_logits - ((1 - unit_mask) * 1e10) #[B, num_derived] 
#                 inf_U_pred = torch.argmax(masked_inf_U_logits, dim=1) #[B]
        
#                 # !Condition on PRED U
#                 pred_unit_emb = self.unit_embedder(inf_U_pred) #[B,sem_hz]
#                 # print('pred_unit_emb', pred_unit_emb.size())
#                 inf_pred_y = self.W_y(torch.cat([inf_h4, pred_unit_emb], dim=1))
#             else:
#                 # !Condition on PRED D
#                 pred_dim_emb = self.dim_embedder(D_pred) #[B,sem_hz]
#                 inf_pred_y = self.W_y(torch.cat([h2, pred_dim_emb], dim=1)) #[B,1]
            
#             # ivan todo here
#             # derived_logits_u_given_xdc = self.calc_u_given_dxc(d_pred_mu, masked_derived_logits, label_num)
#             # D_pred == d_pred_mu
#             # masked_U_logits == masked_derived_logits


#             inf_pred_y = inf_pred_y.squeeze(dim=1)
#             outputs['classification'] = {}
#             outputs['regression'] = {}
#             # *: means argmax
#             outputs['classification']['D|C'] = {'pred': D_pred, 'logits': D_logits, 'true': tgt_dimensions}
            
#             if self.do_derived:
#                 outputs['classification']['U|DC'] = {'pred': U_pred, 'logits': masked_U_logits, 'true': tgt_derived}
#                 outputs['classification']['U|D*C'] = {'pred': inf_U_pred, 'logits': masked_inf_U_logits, 'true': tgt_derived}
#                 outputs['regression']['X|UDC'] = {'pred': torch.exp(pred_y), 'true': label_num}
#                 outputs['regression']['X|U*D*C'] = {'pred': torch.exp(inf_pred_y), 'true': label_num}
#             else:
#                 outputs['regression']['X|DC'] = {'pred': torch.exp(pred_y), 'true': label_num}
#                 outputs['regression']['X|D*C'] = {'pred': torch.exp(inf_pred_y), 'true': label_num}
        
#         return outputs






class GenNumSIUnit(BaseUnitTransformer):
    # This model is called Gen-B in the paper
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
                                
                ############ Begin D|XC for Gen-B ############
                si_logits_d_given_xc = self.calc_d_given_xc_genB(d_pred_mu, si_logits, derived_logits, label_num) #[B, num_derived]
                pred_d_given_xc = torch.argmax(si_logits_d_given_xc, dim=1) #[B]
                outputs['classification']['D|XC'] = {'pred': pred_d_given_xc, 'logits': si_logits_d_given_xc, 'true': tgt_dimensions}
                ############ End D|XC for Gen-B ############
                
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

    def calc_d_given_xc_genB(self, X_predict, D_logits, U_logits, X_target):
        """
        Calculate p(D|X,C) when Gen-B models U.
        p(D|X,C) proportional to p(D|C) * p(X|D,C)
                               = p(D|C) * sum_U p(X,U|D,C) 
                               = p(D|C) * sum_U p(U|D,C) * p(X|U,C)            

        Args:
            X_predict: log mu (mean) for p(X|U,C)   (shape: [B, U])
            D_logits:  logits for p(D|C)            (shape: [B, D])
            U_logits:  logits for p(U|D,C)          (shape: [B, U])
            X_target:  ground truth numbers         (shape: [B])
        Returns:
            log_ll: p(D|X,C) (shape: [B, |D|])
        """
        B = D_logits.size()[0]
        nD = D_logits.size()[1]
        nU = U_logits.size()[1]
        
        # p(D|C)        
        D_log_softmax = torch.nn.LogSoftmax(dim=1)
        D_log_ll = D_log_softmax(D_logits) #[B,D]
        
        # p(U|D,C)        
        unit_mask = torch.arange(nD, device=X_target.device) # [D]
        unit_mask = self.unit_mask(unit_mask) # [D, U]
        assert(unit_mask.size() == (nD, nU))
        unit_mask = unit_mask.unsqueeze(0).repeat(B, 1, 1) # [B,D,U]

        U_logits = U_logits.unsqueeze(1).repeat(1, nD, 1) # [B,D,U]
        U_logits = U_logits - ((1 - unit_mask) * 1e10) # [B,D,U]
        U_log_softmax = torch.nn.LogSoftmax(dim=2)
        U_log_ll = U_log_softmax(U_logits) #[B,D,U]
        assert(U_log_ll.size() == (B, nD, nU))

        # p(X=X_target|U,C)
        X_target = X_target.repeat(nU, 1).permute(1,0) #[B,U]
        distance = self.num_criterion(torch.log(X_target), X_predict) #[B,U]
        constant = torch.log(torch.abs(1/X_target)) #[B,U]
        X_log_ll = -distance + constant #[B,U]
        X_log_ll = X_log_ll.unsqueeze(1).repeat(1, nD, 1) #[B,D,U]
        assert(X_log_ll.size() == (B, nD, nU))

        # sum_U p(U|D,C) * p(X=X_target|U,C)
        XU_log_ll = U_log_ll + X_log_ll #[B,D,U]
        XU_log_ll = torch.logsumexp(XU_log_ll, dim=2) #[B,D]
        assert(XU_log_ll.size() == (B, nD))

        # p(D|X,C) = p(D|C) * sum_U p(U|D,C) * p(X|U,C)
        log_ll = D_log_ll + XU_log_ll #[B,D]
        
        return log_ll #[B,D]

    def calc_d_given_xc(self, d_pred_mu, si_logits, tgt_numbers):
        """
        Calculate p(D|X,C) when Gen-B does not model U.
        Proportional to p(D|C) * p(X|D,C).

        Args:
            d_pred_mu:   log mu (means) for p(X|D,C)  (shape: [B, D])
            si_logits:   logits for p(D|C)            (shape: [B, D])
            tgt_numbers: ground truth numbers         (shape: [B])
        Returns:
            log_ll: log likelihood for p(D|X,C)       (shape: [B, D])
        """

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


class DownGen(GenNumSIUnit):
    def __init__(self, config, args, class_buckets, dimID_2_derivedID):
        super().__init__(config, args, class_buckets, dimID_2_derivedID)
        # self.proj_c = torch.nn.Linear(self.c_hz, self.d_hz)
    
    def init_downstream(self):
        if self.args.downstream == 'NumerSense':
            self.num_classes = 12
        else:
            self.num_classes = self.args.num_classes
        self.downstream_classifier = torch.nn.Linear(self.sem_hz, self.num_classes)
        

    def model_loss(self, pooled_output, gzsl, do_eval=False, label=None, **kwargs):
        sem_context = self.proj_semantic(pooled_output)
        pred_logits = self.downstream_classifier(sem_context)
        total_loss = self.crossentropy(pred_logits, label) #[B]

        outputs = {'total_loss': total_loss}
        if do_eval:
            pred_label = torch.argmax(pred_logits, dim=1) #[B]
            outputs['classification'] = {}
            outputs['classification']['Y|C'] = {'pred': pred_label, 'logits': pred_logits, 'true': label}
    
        return outputs

class SharedGenNumSIUnit(BaseUnitTransformer):
    def __init__(self, config, args, class_buckets, dimID_2_derivedID):
        super().__init__(config, args, class_buckets, dimID_2_derivedID)
        # self.proj_c = torch.nn.Linear(self.c_hz, self.d_hz)
        assert self.do_zsl == False
        # assert self.supervised_derived == True
        if self.do_weighted:
            weights = torch.tensor([1, 3.02,   88.4,  18.1,   79.4,  20.4,  24.3])
            self.dim_crossentropy = nn.CrossEntropyLoss(reduction='none', weight=weights)
        else:
            self.dim_crossentropy = nn.CrossEntropyLoss(reduction='none')
        self.crossentropy = nn.CrossEntropyLoss(reduction='none')
        self.num_criterion = torch.nn.L1Loss(reduction='none')
        self.proj_semantic = torch.nn.Linear(self.c_hz, self.sem_hz)
        self.num_heads = args.num_heads
        self.attn_1 = nn.MultiheadAttention(self.sem_hz, self.num_heads, batch_first=True)
        self.attn_2 = nn.MultiheadAttention(self.sem_hz, self.num_heads, batch_first=True)

        #todo: ivan this is broken:
        # self.dim_embedder = UnitEmbedder(args, class_buckets, self.sem_hz)
        self.dim_embedder = torch.nn.Embedding(self.n_y1, self.sem_hz)
        self.unit_embedder = torch.nn.Embedding(self.num_derived, self.sem_hz)

        self.W_D = torch.nn.Linear(self.sem_hz, self.n_y1)
        self.W_y = torch.nn.Linear(2*self.sem_hz, 1)
        
        if self.do_derived:
            self.W_cat = torch.nn.Linear(2*self.sem_hz, self.sem_hz)
            self.W_U = torch.nn.Linear(self.sem_hz, self.num_derived)

    def model_loss(self, pooled_output, gzsl, do_eval=False, label_dim=None, label_num=None, tgt_dimensions=None, tgt_derived=None, **kwargs):
        B_size = pooled_output.size()[0]
        sem_context = self.proj_semantic(pooled_output) #[B,sem_hz]
        
        h1 = sem_context
        dim_ind = torch.arange(self.n_y1, device=h1.device)
        dim_emb = self.dim_embedder(dim_ind) #[sem_hz,D]
        # print('dim_emb', dim_emb.size())
        batch_dim_emb = dim_emb.repeat(B_size, 1, 1) #[B,D,sem_hz]
        # print('batch_dim_emb', batch_dim_emb.size())
        Bh1D = torch.cat([h1.unsqueeze(dim=1), batch_dim_emb], dim=1) #[B,1+D, sem_hz]
        # print('Bh1D', Bh1D.size())
        Bh2, _ = self.attn_1(Bh1D,Bh1D,Bh1D) #[B,1+D, sem_hz]
        # print('Bh2', Bh2.size())
        h2 = Bh2[:,0,:] #[B,sem_hz]
        # print('h2', h2.size())
        
        D_logits = self.W_D(h2) #[B,D]
        D_loss = self.dim_crossentropy(D_logits, tgt_dimensions) #[B]
        D_log_ll = -D_loss
        
        if self.do_derived:
            unit_ind = torch.arange(self.num_derived, device=h1.device)
            unit_emb = self.unit_embedder(unit_ind) #[sem_hz,U]
            batch_unit_emb = unit_emb.repeat(B_size, 1, 1) #[B,U,sem_hz]
            Bh2U = torch.cat([h2.unsqueeze(dim=1), batch_unit_emb], dim=1) #[B,1+U, sem_hz]
            # print('Bh2U', Bh2U.size())
            Bh3, _ = self.attn_2(Bh2U,Bh2U,Bh2U) #[B,1+U, sem_hz]
            # print('Bh3', Bh3.size())
            h3 = Bh3[:,0,:] #[B,sem_hz]
            # print('h3', h3.size())
            # !Condition on TRUE D
            true_dim_emb = self.dim_embedder(tgt_dimensions) #[B,sem_hz]
            h4 = self.W_cat(torch.cat([h3, true_dim_emb], dim=1))
            # print('h4', h4.size())
            U_logits = self.W_U(h4) #[B,U]

            # !Condition on TRUE D
            unit_mask = self.unit_mask(tgt_dimensions) #[B, num_derived]            
            masked_U_logits = U_logits - ((1 - unit_mask) * 1e10) #[B, num_derived] 
            U_loss = self.crossentropy(masked_U_logits, tgt_derived) #[B]
            # Get log-likelihoods
            U_log_ll = -U_loss #[B]

            # !Condition on TRUE U
            true_unit_emb = self.unit_embedder(tgt_derived) #[B,sem_hz]
            # print('true_unit_emb', true_unit_emb.size())
            pred_y = self.W_y(torch.cat([h4, true_unit_emb], dim=1))

        else:            
            # !Condition on TRUE D
            true_dim_emb = self.dim_embedder(tgt_dimensions) #[B,sem_hz]
            pred_y = self.W_y(torch.cat([h2, true_dim_emb], dim=1)) #[B,1]
        
        pred_y = pred_y.squeeze(dim=1) #[B]
        # print('pred_y', pred_y.size())
        distance = self.num_criterion(torch.log(label_num), pred_y)
        num_loss = distance
        constant = torch.log(torch.abs(1/label_num))
        num_log_ll = -distance + constant
        
        outputs = {}
        total_loss = D_loss + num_loss
        if self.do_derived:
            total_loss = total_loss + U_loss
            outputs['U_loss'] = U_loss
        outputs = {'total_loss': total_loss, 'num_loss': num_loss, 'si_loss': D_loss}
        
        # print('total_loss', total_loss.size())
        # print('U_loss', U_loss.size())
        # print('num_loss', num_loss.size())
        # print('D_loss', D_loss.size())
        
        # inference mode
        if do_eval:
            #only 1 D_Pred
            D_pred = torch.argmax(D_logits, dim=1) #[B]  
            
            if self.do_derived:
                U_pred = torch.argmax(masked_U_logits, dim=1) #[B]

                # !Condition on PRED D
                pred_dim_emb = self.dim_embedder(D_pred) #[B,sem_hz]
                inf_h4 = self.W_cat(torch.cat([h3, pred_dim_emb], dim=1))
                # print('inf_h4', inf_h4.size())
                inf_U_logits = self.W_U(inf_h4) #[B,U]

                # !Condition on PRED D
                unit_mask = self.unit_mask(D_pred) #[B, num_derived]            
                masked_inf_U_logits = inf_U_logits - ((1 - unit_mask) * 1e10) #[B, num_derived] 
                inf_U_pred = torch.argmax(masked_inf_U_logits, dim=1) #[B]
        
                # !Condition on PRED U
                pred_unit_emb = self.unit_embedder(inf_U_pred) #[B,sem_hz]
                # print('pred_unit_emb', pred_unit_emb.size())
                inf_pred_y = self.W_y(torch.cat([inf_h4, pred_unit_emb], dim=1))

                # masked_U_logits = P(U|DC), [B, num_derived] 
                # pred_y = P(X|UDC), [B]
                # label_num = true numbers [B]  
                logits_u_given_dxc = self.calc_u_given_dxc_logits(h4, masked_U_logits, label_num) # [B, num_derived]             
                pred_derived_given_xdc = torch.argmax(logits_u_given_dxc, dim=1) #[B]

            else:
                # !Condition on PRED D
                pred_dim_emb = self.dim_embedder(D_pred) #[B,sem_hz]
                inf_pred_y = self.W_y(torch.cat([h2, pred_dim_emb], dim=1)) #[B,1]
            
            inf_pred_y = inf_pred_y.squeeze(dim=1)
            outputs['classification'] = {}
            outputs['regression'] = {}
            # *: means argmax
            outputs['classification']['D|C'] = {'pred': D_pred, 'logits': D_logits, 'true': tgt_dimensions}
            if self.do_derived:
                outputs['classification']['U|D*C'] = {'pred': inf_U_pred, 'logits': masked_inf_U_logits, 'true': tgt_derived}
                outputs['classification']['U|DC'] = {'pred': U_pred, 'logits': masked_U_logits, 'true': tgt_derived}
                outputs['classification']['U|XDC'] = {'pred': pred_derived_given_xdc, 'logits': logits_u_given_dxc, 'true': tgt_derived}
                outputs['regression']['X|UDC'] = {'pred': torch.exp(pred_y), 'true': label_num}
                outputs['regression']['X|U*D*C'] = {'pred': torch.exp(inf_pred_y), 'true': label_num}
            else:
                outputs['regression']['X|DC'] = {'pred': torch.exp(pred_y), 'true': label_num}
                outputs['regression']['X|D*C'] = {'pred': torch.exp(inf_pred_y), 'true': label_num}
        
        return outputs

    def calc_u_given_dxc_logits(self, h4, masked_U_logits, label_num):
        # p(U|XDC) proportional to P(X|UDC) * P(U|DC) = P(X|UC) * P(U|DC)        
        # h4 [B,sem_hz]        
        # masked_U_logits = P(U|DC), [B, num_derived]         
        # label_num = true numbers [B]  
        
        num_derived = masked_U_logits.size()[1]        
        batch_size = masked_U_logits.size()[0]

        u_idx = torch.arange(num_derived, device=label_num.device) # [num_derived]
        u_idx = u_idx.repeat(batch_size, 1) # [B, num_derived]
        all_unit_emb = self.unit_embedder(u_idx) # [B, num_derived, sem_hz]        
        h4 = h4.unsqueeze(1).repeat(1, num_derived, 1) # [B, num_derived, sem_hz]
        h4_cat_unit_emb = torch.cat((h4, all_unit_emb), 2) # [B, num_derived, sem_hz*2]
        pred_y = self.W_y(h4_cat_unit_emb) #[B, num_derived, 1]
        pred_y = pred_y.squeeze(2) #[B, num_derived]
        
        label_num = label_num.repeat(num_derived, 1).permute(1,0) # [B,num_derived]
        distance = self.num_criterion(torch.log(label_num), pred_y) #[B,num_derived]
        constant = torch.log(torch.abs(1/label_num)) #[B,num_derived]
        num_log_ll = -distance + constant #[B,num_derived]

        log_softmax = torch.nn.LogSoftmax(dim=1)
        derived_log_ll = log_softmax(masked_U_logits) #[B,num_derived]
        
        log_ll = num_log_ll + derived_log_ll #[B, num_derived]
        return log_ll

class LatentGen(GenNumSIUnit):
    def __init__(self, config, args, class_buckets, dimID_2_derivedID):
        super().__init__(config, args, class_buckets, dimID_2_derivedID)
        
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)

    def model_loss(self, pooled_output, gzsl, do_eval=False, label_num=None, tgt_dimensions=None, **kwargs):
    # def forward(self, texts, tgt_dimensions, tgt_numbers=None):
        """
        Args:
            texts (Tensor): input tensor of tokens [Batch, Seq-Len]
            tgt_dimensions (Tensor): True index of dimension or SI-unit since its 1 to 1 correspondence
            tgt_numbers (Tensor, optional): Tensor of float values for each sentence. Defaults to None.

        Returns:
            Dict: 
        """
        # todo: rename [label_unit: tgt_dimensions]
        # todo: pass tgt_numbers
        # print('label_num', label_num.size())
        # texts [B,S]
        # tgt_dimensions [B]
        ################SI_Units################
        sem_context = self.proj_semantic(pooled_output)
        si_logits = self.si_classifier(sem_context) #[B, num_si]
        si_log_p = self.logsoftmax(si_logits)
        # si_loss = self.crossentropy(si_logits, tgt_dimensions) #[B]
        # si_log_ll = -si_loss
        
        ################number###################
        d_pred_mu = self.W_num(sem_context) #[B, num_si]
        # print('label_num', label_num.size())
        # print('d_pred_mu', d_pred_mu.size())
        label_num_repeat = label_num.repeat(self.n_y1, 1).permute(1,0) #[B, num_si]
        # print('label_num', label_num.size())

        distance = self.num_criterion(torch.log(label_num_repeat), d_pred_mu)
        num_loss = distance
        constant = torch.log(torch.abs(1/label_num_repeat))
        num_log_ll = -distance + constant
        ########################################
        # print('si_log_p', si_log_p.size())
        # print('num_log_ll', num_log_ll.size())
        log_ll = torch.logsumexp(num_log_ll + si_log_p, dim=1) #[B]
            
        total_loss = -log_ll #[B]
        outputs = {'total_loss': total_loss}
        # cluster evaluation
        # metrics.adjusted_mutual_info_score(labels_true, labels_pred)  
        # metrics.rand_score(true,pred)        
        if do_eval:
            pred_si = torch.argmax(si_logits, dim=1) #[B]
            # todo: add flag to change from (D|C) to (D|XC) based on the mask-num flag
            outputs['unsupervised'] = {}
            outputs['unsupervised']['D|C'] = {'pred': pred_si, 'logits': si_logits, 'true': tgt_dimensions}
        
            si_logits_d_given_xc = self.calc_d_given_xc(d_pred_mu, si_logits, label_num) #[B,num_si]
            if torch.any(si_logits_d_given_xc != si_logits_d_given_xc):
                print('si_logits_d_given_xc', si_logits_d_given_xc.size())
                print(si_logits_d_given_xc.min(), si_logits_d_given_xc.max())
                foohere
            
            pred_si_given_xc = torch.argmax(si_logits_d_given_xc, dim=1) #[B]
            outputs['unsupervised']['D|XC'] = {'pred': pred_si_given_xc, 'logits': si_logits_d_given_xc, 'true': tgt_dimensions}

            pred_mu = d_pred_mu.gather(1, pred_si.view(-1,1)).squeeze(1) #[B]
            outputs['regression'] = {}
            outputs['regression']['X|DC'] = {'pred': torch.exp(pred_mu), 'true': label_num}
        
        return outputs


def main():
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaForProbabilisticSequenceClassification.from_pretrained('roberta-base', return_dict=True)
    inputs = tokenizer("Hello, my dog is cute and friendly", return_tensors="pt")
    labels = torch.tensor([.7]).view(1,1)
    loss, probs = model(**inputs, labels=labels)
    print('loss', loss.size())
    print('probs', probs.size())

if __name__ == '__main__':
    main()