
import sys


            
import os
import json
import random
import logging
import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Tuple, Union
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm.auto import tqdm
import torchaudio
from torch.utils.data import DataLoader
from transformers import SchedulerType, get_scheduler
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import average_precision_score

from Qformer import BertConfig, BertLMHeadModel

def generate_length_mask(lens, max_length=None):
    lens = torch.as_tensor(lens)
    N = lens.size(0)
    if max_length is None:
        max_length = max(lens)
    idxs = torch.arange(max_length).repeat(N).view(N, max_length)
    idxs = idxs.to(lens.device)
    mask = (idxs < lens.view(-1, 1)).int()
    return mask

class QformerBridgeNet(torch.nn.Module):
    def __init__(self, Qformer_model_name: str = "bert-base-uncased", num_query_token: int = 32, 
                 hiddin_size: int = 1024, speech_width: int = 1024, nlabels: int = 527, freeze_QFormer: bool = False):
        super().__init__()
        
        self.Qformer_model_name = Qformer_model_name
        self.Qformer_model_ckpt = None
        self.audio_Qformer, self.audio_query_tokens, encoder_config = self.init_Qformer(num_query_token=num_query_token,  speech_width=speech_width)
        self.audio_Qformer.cls = None
        self.audio_Qformer.bert.embeddings.word_embeddings = None
        self.audio_Qformer.bert.embeddings.position_embeddings = None
        for layer in self.audio_Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        
        self.freeze_QFormer = freeze_QFormer
        
        if freeze_QFormer:
            for name, param in self.speech_Qformer.named_parameters():
                param.requires_grad = False
            self.speech_Qformer.eval()
            self.speech_query_tokens.requires_grad = False

        self.hiddin_projection = torch.nn.Linear(encoder_config.hidden_size, hiddin_size)
        torch.nn.init.xavier_uniform_(self.hiddin_projection.weight, gain=torch.nn.init.calculate_gain("relu"))
        # To train audio tagging
        self.nlabels = nlabels
        self.projection = torch.nn.Linear(hiddin_size, nlabels)
        torch.nn.init.xavier_uniform_(self.projection.weight, gain=torch.nn.init.calculate_gain("relu"))
        self.logit_loss: torch.nn.Module
        self.activation: torch.nn.Module = torch.nn.Sigmoid()
        self.logit_loss = torch.nn.BCEWithLogitsLoss()
        
    def init_Qformer(self, num_query_token, speech_width, num_hidden_layers=2, cross_attention_freq=2):
        encoder_config = BertConfig.from_pretrained(self.Qformer_model_name)
        encoder_config.num_hidden_layers = num_hidden_layers
        encoder_config.encoder_width = speech_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel(config=encoder_config)
        if self.Qformer_model_ckpt:
            Qformer.load_state_dict(torch.load(self.Qformer_model_ckpt))
            #Qformer.load_state_dict(torch.load(self.Qformer_model_ckpt), strict=False)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens, encoder_config
    
    def hidden(self, batch,):
        audio_feature, lens = batch['embed'], batch['embed_len']
        frame_atts = generate_length_mask(lens).to(audio_feature.device)
        audio_query_tokens=self.audio_query_tokens.expand(audio_feature.shape[0], -1, -1)
        #frame_atts = torch.ones(audio_feature.size()[:-1], dtype=torch.long).to(audio_feature.device)
        
        #print(audio_query_tokens.shape, audio_feature.shape, frame_atts.shape)
        audio_query_output=self.audio_Qformer.bert(
            query_embeds=audio_query_tokens, #[32,768]
            encoder_hidden_states=audio_feature,
            encoder_attention_mask=frame_atts,
            return_dict=True,
            )
        audio_hidden = audio_query_output.last_hidden_state
        audio_hidden = self.hiddin_projection(audio_hidden)
        return audio_hidden


    def forward(self, batch) -> torch.Tensor:
        x = self.hidden(batch)
        return x
    
    def training_step(self, batch):
        # training_step defined the train loop.
        # It is independent of forward
        x = self.hidden(batch)
        x = x.mean(dim=1)
        y_hat = self.projection(x)

        y =  batch['event_onehot']
        loss = self.logit_loss(y_hat, y)
        return loss

    def eval_step(self, batch):
        x = self.hidden(batch)
        x = x.mean(dim=1)
        y_hat = self.projection(x)

        y_pr = self.activation(y_hat)
        z = {
            "prediction": y_pr,
            "prediction_logit": y_hat,
            "target": batch['event_onehot'],
        }
        # https://stackoverflow.com/questions/38987/how-do-i-merge-two-dictionaries-in-a-single-expression-taking-union-of-dictiona
        return {**z}

class EmbeddingExtractor(torch.nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.nfeatures = 512 # CLAP
        if conf['embedding_type'] == 'caption':
            if conf['embedding_extractor'] == 'CLAP':
                import laion_clap
                from laion_clap.clap_module.factory import load_state_dict as clap_load_state_dict
                self.extractor = laion_clap.CLAP_Module(enable_fusion=False)
                ckpt_path = '~/.cache/laion_clap/630k-audioset-best.pt'
                ckpt = clap_load_state_dict(ckpt_path, skip_params=True)
                del_parameter_key = ["text_branch.embeddings.position_ids"]
                ckpt = {"model."+k:v for k, v in ckpt.items() if k not in del_parameter_key}
                self.extractor.load_state_dict(ckpt)

                self.get_embedding = self._get_clap_embedding
            if conf['embedding_extractor'] == 'T5':
                from transformers import AutoTokenizer, T5EncoderModel
                self.text_encoder_name = "~/.cache/huggingface/hub/models--google--flan-t5-large/snapshots/0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a/"
                self.tokenizer = AutoTokenizer.from_pretrained(self.text_encoder_name)
                self.extractor = T5EncoderModel.from_pretrained(self.text_encoder_name)
                self.get_embedding = self._get_t5_embedding
                self.nfeatures = 1024
        elif conf['embedding_type'] == 'caption_tts':
            if conf['embedding_extractor'] in ['hubert', 'wavlm', 'dac']:
                self.get_embedding = lambda caption, caption_tts, caption_tts_lens: (caption_tts.squeeze(1), caption_tts_lens)
                self.nfeatures = 1024
        else:
            raise ValueError(f"Unknown embedding_type {conf['embedding_type']}")    
    
    def _get_clap_embedding(self, caption, caption_tts, caption_tts_lens):
        clap_embed = self.extractor.get_text_embedding(caption, use_tensor=True).unsqueeze(1)
        #embed_len = torch.ones(len(clap_embed)).to(device)
        embed_len = np.ones(len(clap_embed), np.int32)
        return clap_embed, embed_len
    
    def _get_t5_embedding(self, caption, caption_tts, caption_tts_lens):
        batch = self.tokenizer(caption, max_length=self.tokenizer.model_max_length, padding=True, truncation=True, return_tensors="pt")
        input_ids, attention_mask = batch.input_ids.to(device), batch.attention_mask.to(device)
        with torch.no_grad():
            encoder_hidden_states = self.extractor(input_ids=input_ids, attention_mask=attention_mask)[0]
        boolean_encoder_mask = (attention_mask == 1).to(device)
        embed_len = attention_mask.sum(dim=1).to(device)
        return encoder_hidden_states, embed_len
    
    def forward(self, caption, caption_tts, caption_tts_lens):
        #return torch.zeros(len(caption), 512)
        return self.get_embedding(caption, caption_tts, caption_tts_lens)

class SpeechDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, conf):
        super().__init__()
        self.data = json.load(open(data_path, 'r'))['audios']
        self.data_path = data_path
        self.data_split = data_path.split('/')[-2]

        data_df = pd.read_csv(conf['sound_event_annotation'], skiprows=3, header=None, names=['YTID', 'start_seconds', 'end_seconds', 'positive_labels'], sep=' ')
        data_df['YTID_r'] = data_df['YTID'].apply(lambda x: x[:-1])
        self.filename_to_label = dict(zip(data_df['YTID_r'], data_df['positive_labels']))
        index_df = pd.read_csv(conf['sound_event_meta'])          
        self.label_to_index = dict(zip(index_df['mid'], index_df['index']))
        self.index_to_displayname = dict(zip(index_df['index'], index_df['display_name']))

        self.nlabels = len(self.label_to_index)
        self.conf = conf
        if self.conf['embedding_extractor'] == 'hubert':
            self.caption_tts_h5 = h5py.File(f"data/hdf5/hubert_{self.data_split}.h5", "r")
        elif self.conf['embedding_extractor'] == 'wavlm':
            self.caption_tts_h5 = h5py.File(f"data/hdf5/wavlm_{self.data_split}.h5", "r")
        elif self.conf['embedding_extractor'] == 'dac':
            self.caption_tts_h5 = h5py.File(f"data/hdf5/dac_{self.data_split}.h5", "r")

    def _prob_to_label(self, prob, th=0.5):
        prob_flag = prob > th
        output = []
        for idx in range(self.nlabels):
            if prob_flag[idx]:
                output.append(self.index_to_displayname[idx])
        return output

    def __getitem__(self, index):
        item = self.data[index]
        audio_id = item['audio_id']
        caption, caption_tts = item['captions'][0]['tokens'], f"data/vits_output/{self.data_split}/{audio_id}.wav"
        
        if self.conf['embedding_type'] == 'caption_tts':
            caption_tts = self.caption_tts_h5[audio_id][()]
            caption_tts = torch.tensor(caption_tts)
            caption_tts_len = caption_tts.shape[1]
        else:
            caption_tts_len = None

        tags = self.filename_to_label[audio_id]
        tag_idx = [self.label_to_index[tag] for tag in tags.split(",")]
        event_onehot = np.zeros(self.nlabels)   
        event_onehot[tag_idx] = 1
        
        return caption, caption_tts, caption_tts_len, event_onehot, audio_id
    
    def __len__(self):
        return len(self.data)

def padding_collate_fn(batch):
    caption, caption_tts, caption_tts_len, event_onehot, audio_id = zip(*batch)

    if caption_tts_len[0] == None:
        padding_caption_tts = caption_tts
    else:
        B, max_len, D = len(caption_tts), max(caption_tts_len), caption_tts[0].shape[-1]
        padding_caption_tts = torch.zeros((B, max_len, D), dtype=caption_tts[0].dtype)
        for idx, i_len in enumerate(caption_tts_len):
            padding_caption_tts[idx, :i_len, :] = caption_tts[idx]
    event_onehot = torch.tensor(np.array(event_onehot))

    return caption, padding_caption_tts, caption_tts_len, event_onehot, audio_id

class EarlyStopping:
    def __init__(self, patience=20, min_delta=0.0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def _check(self, score):
        if self.best_score is None:
            self.best_score = score
        elif (self.mode == 'min' and score >= self.best_score - self.min_delta) or \
             (self.mode == 'max' and score <= self.best_score + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        self.best_score_flag = self.counter == 0
        return self.early_stop, self.best_score_flag 

def setup_logger(log_file):
    logger = logging.getLogger('training_logger')
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def get_map(nlabels, y_true, y_pred):
    ap_scores = []
    for i in range(nlabels):
        if np.any(y_true[:, i] == 1):
            ap = average_precision_score(y_true[:, i], y_pred[:, i])
            ap_scores.append(ap)
    
    if ap_scores:
        mAP = np.mean(ap_scores)
    else:
        mAP = 0.0
    
    return mAP, ap_scores

hyperparams_candidate = {
    "max_epochs": [500],
    "lr": [3.2e-3],
    "batch_size": [512], # dac
    "patience": [20],

    "sound_event_meta": ["data/AudioSet/metadata/class_labels_indices.csv"],
    "sound_event_annotation": ["data/AudioSet/unbalanced_train/unbalanced_train_segments.csv"],

    "train_file_path": ["data/audiocaps/train/text.json"],
    "val_file_path": ["data/audiocaps/val/text.json"],
    "test_file_path": ["data/audiocaps/test/text.json"],
}

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a diffusion model for text to audio generation task.")
    parser.add_argument("--embedding_type", "-t", type=str, default='caption_tts', choices=['caption', 'caption_tts'])
    parser.add_argument("--embedding_extractor", "-e", type=str, default='dac', choices=['CLAP', 'T5', 'hubert', 'wavlm', 'dac'])
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--infer", action='store_true')
    parser.add_argument("--layer", "-l", type=int, default=1)
    parser.add_argument("--ckpt", "-c", type=str, default="")
    parser.add_argument("--device", "-d", type=str, default="cuda:0")
    args = parser.parse_args()
    return args

if __name__=="__main__":
    
    args = parse_args()
    if args.output_dir is None:
        args.output_dir = f"output/qformer_{args.embedding_type}_{args.embedding_extractor}" 
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger(f"{args.output_dir}/inference_{args.ckpt.split('/')[-1][:-3]}.log") if args.infer else setup_logger(f"{args.output_dir}/semantic.log")

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)    
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu") 
    embedding_extractor = EmbeddingExtractor(vars(args)).eval().to(device)
    for param in embedding_extractor.parameters():
        param.requires_grad = False

    if not args.infer: # Training
        exp_result = {}
        hyperparams_list = list(ParameterGrid(hyperparams_candidate))
        for hyperparams_i, hyperparams in enumerate(hyperparams_list):
            hyperparams.update(vars(args))
            logger.info(f"***** Initialization exp_id:{hyperparams_i} *****")
            train_dataset = SpeechDataset(hyperparams["train_file_path"], hyperparams)
            train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=hyperparams["batch_size"], collate_fn=padding_collate_fn)
            val_dataset = SpeechDataset(hyperparams["val_file_path"], hyperparams)
            val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=hyperparams["batch_size"], collate_fn=padding_collate_fn)
            test_dataset = SpeechDataset(hyperparams["test_file_path"], hyperparams)
            test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=hyperparams["batch_size"], collate_fn=padding_collate_fn)
            logger.info("Load dataset, done!")

            model = QformerBridgeNet(speech_width=embedding_extractor.nfeatures, nlabels=train_dataset.nlabels).to(device)
            
            
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=hyperparams["lr"],
                betas=(0.9, 0.999),
                weight_decay=1e-2,
                eps=1e-08,
            )
            lr_scheduler = get_scheduler(
                name='linear',
                optimizer=optimizer,
                num_warmup_steps=5 * len(train_dataloader),
                num_training_steps=hyperparams['max_epochs'] * len(train_dataloader),
            )
            logger.info("Build model and set up exp, done!")

            logger.info("***** Running training *****")
            logger.info(f"  Num examples = {len(train_dataset)}")
            logger.info(f"  Num Epochs = {hyperparams['max_epochs']}")
            logger.info(f"  Instantaneous batch size per device = {hyperparams['batch_size']}")    
            
            # best_loss, best_epoch = np.inf, 0
            early_stopping = EarlyStopping(patience=hyperparams['patience'], min_delta=0.00, mode='max')
            for epoch in range(hyperparams["max_epochs"]):
                model.train()
                total_loss = 0      
                for step, batch in enumerate(tqdm(train_dataloader)):
                    #caption, caption_tts, y, _ = batch
                    caption, caption_tts, caption_tts_len, event_onehot, audio_id = batch
                    with torch.no_grad():
                        embed, embed_len = embedding_extractor(caption, caption_tts, caption_tts_len)
                    loss = model.training_step({"embed": embed.to(device), "embed_len": embed_len, "event_onehot": event_onehot.to(device)})                  
                    loss.backward()
                    total_loss += loss.detach().float()
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                result = {}
                result["epoch"] = epoch,
                result["train_loss"] = round(total_loss.item()/len(train_dataloader), 4)
                result["lr"] = round(optimizer.param_groups[0]['lr'], 8)


                model.eval()
                y_true, y_pred = [], []
                with torch.no_grad():
                    for step, batch in enumerate(val_dataloader):
                        caption, caption_tts, caption_tts_len, event_onehot, audio_id = batch
                    with torch.no_grad():
                        embed, embed_len = embedding_extractor(caption, caption_tts, caption_tts_len)
                        output = model.eval_step({"embed": embed.to(device), "embed_len": embed_len, "event_onehot": event_onehot.to(device)})
                        y_true.append(event_onehot.numpy())
                        y_pred.append(output["prediction"].detach().cpu().numpy())
                    val_mAP, ap_scores = get_map(train_dataset.nlabels, y_true=np.concatenate(y_true, axis=0), y_pred=np.concatenate(y_pred, axis=0))
                # result["val_event"] = len(ap_scores) 155
                result["val_mAP"] = val_mAP
                logger.info(result)
                early_stop_flag, best_score_flag = early_stopping._check(val_mAP)
                if best_score_flag:
                    torch.save(model.state_dict(), f"{args.output_dir}/exp{hyperparams_i}_best.pt")
                if early_stop_flag:
                    torch.save(model.state_dict(), f"{args.output_dir}/exp{hyperparams_i}_earlystop_epoch{epoch}.pt")
                    break 

            logger.info("***** Running test *****")  
            y_true, y_pred = [], []         
            with torch.no_grad():
                for step, batch in enumerate(test_dataloader):
                    caption, caption_tts, caption_tts_len, event_onehot, audio_id = batch
                    embed, embed_len = embedding_extractor(caption, caption_tts, caption_tts_len)
                    output = model.eval_step({"embed": embed.to(device), "embed_len": embed_len, "event_onehot": event_onehot.to(device)})
                    y_true.append(event_onehot.numpy())
                    y_pred.append(output["prediction"].detach().cpu())
                test_mAP, ap_scores = get_map(train_dataset.nlabels, y_true=np.concatenate(y_true, axis=0), y_pred=np.concatenate(y_pred, axis=0))
            result.update({"Test mAP": test_mAP, "n_event": len(ap_scores)})
            logger.info(result)
            exp_result[f"exp{hyperparams_i}_result"] = result
        logger.info(exp_result)
    
    else: # Inference only
        hyperparams = list(ParameterGrid(hyperparams_candidate))[args.layer - 1]
        hyperparams.update(vars(args))
        test_dataset = SpeechDataset(hyperparams["test_file_path"], hyperparams)
        test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=hyperparams["batch_size"], collate_fn=padding_collate_fn)
        model = QformerBridgeNet(nlabels=test_dataset.nlabels).to(device)
        model.load_state_dict(torch.load(args.ckpt))
        model.eval()

        logger.info("***** Running test *****")       
        y_true, y_pred = [], []  
        result = {}   
        output_sample = []      
        with torch.no_grad():
            for step, batch in enumerate(test_dataloader):
                caption, caption_tts, caption_tts_len, event_onehot, audio_id = batch
                # x = embedding_extractor(caption, caption_tts)
                # output = model.eval_step(x.to(device), y.to(device))
                embed, embed_len = embedding_extractor(caption, caption_tts, caption_tts_len)
                output = model.eval_step({"embed": embed.to(device), "embed_len": embed_len, "event_onehot": event_onehot.to(device)})
                gt = event_onehot.detach().cpu().numpy()
                pred = output["prediction"].detach().cpu().numpy()
                y_true.append(gt)
                y_pred.append(pred)
                for step_idx in range(len(caption)):
                    output_sample.extend([
                        {
                            "audio_id": audio_id[step_idx],
                            "caption": caption[step_idx],
                        },
                        {
                            "audio_id": audio_id[step_idx],
                            "gt": test_dataset._prob_to_label(gt[step_idx]),
                            "pred": test_dataset._prob_to_label(pred[step_idx]),
                        }
                    ])
            test_mAP, ap_scores = get_map(test_dataset.nlabels, y_true=np.concatenate(y_true, axis=0), y_pred=np.concatenate(y_pred, axis=0))
            result.update({"Test mAP": test_mAP, "n_event": len(ap_scores)})
            logger.info(result)
            for i in range(20 * 2):
                logger.info(output_sample[i])
        
           
            