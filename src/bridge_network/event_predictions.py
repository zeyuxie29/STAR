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
from tqdm.auto import tqdm
import torchaudio
from torch.utils.data import DataLoader
from transformers import SchedulerType, get_scheduler
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import average_precision_score

class FullyConnectedPrediction(torch.nn.Module):
    def __init__(self, nfeatures: int, nlabels: int, conf: Dict):
        super().__init__()

        hidden_modules: List[torch.nn.Module] = []
        curdim = nfeatures

        last_activation = "linear"
        if conf["hidden_layers"]:
            for i in range(conf["hidden_layers"]):
                linear = torch.nn.Linear(curdim, conf["hidden_dim"])
                conf["initialization"](
                    linear.weight,
                    gain=torch.nn.init.calculate_gain(last_activation),
                )
                hidden_modules.append(linear)
                if not conf["norm_after_activation"]:
                    hidden_modules.append(conf["hidden_norm"](conf["hidden_dim"]))
                hidden_modules.append(torch.nn.Dropout(conf["dropout"]))
                hidden_modules.append(torch.nn.ReLU())
                if conf["norm_after_activation"]:
                    hidden_modules.append(conf["hidden_norm"](conf["hidden_dim"]))
                curdim = conf["hidden_dim"]
                last_activation = "relu"

            self.hidden = torch.nn.Sequential(*hidden_modules)
        else:
            self.hidden = torch.nn.Identity()  # type: ignore
        self.projection = torch.nn.Linear(curdim, nlabels)

        conf["initialization"](
            self.projection.weight, gain=torch.nn.init.calculate_gain(last_activation)
        )
        self.logit_loss: torch.nn.Module
        self.activation: torch.nn.Module = torch.nn.Sigmoid()
        self.logit_loss = torch.nn.BCEWithLogitsLoss()

    def forward_logit(self, x: torch.Tensor) -> torch.Tensor:
        x = self.hidden(x)
        x = self.projection(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_logit(x)
        x = self.activation(x)
        return x
    
    def training_step(self, x, y):
        # training_step defined the train loop.
        # It is independent of forward
        y_hat = self.forward_logit(x)
        loss = self.logit_loss(y_hat, y)
        return loss

    def eval_step(self, x, y):
        # -> Dict[str, Union[torch.Tensor, List(str)]]:
        y_hat = self.forward_logit(x)
        y_pr = self.forward(x)
        z = {
            "prediction": y_pr,
            "prediction_logit": y_hat,
            "target": y,
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
                self.get_embedding = lambda caption, caption_tts: caption_tts.squeeze(1)
                self.nfeatures = 1024
        else:
            raise ValueError(f"Unknown embedding_type {conf['embedding_type']}")    
    
    

    def _get_clap_embedding(self, caption, caption_tts):
        clap_embed = self.extractor.get_text_embedding(caption, use_tensor=True)
        return clap_embed
    
    def _get_t5_embedding(self, caption, caption_tts):
        batch = self.tokenizer(caption, max_length=self.tokenizer.model_max_length, padding=True, truncation=True, return_tensors="pt")
        input_ids, attention_mask = batch.input_ids.to(device), batch.attention_mask.to(device)
        with torch.no_grad():
            encoder_hidden_states = self.extractor(input_ids=input_ids, attention_mask=attention_mask)[0]
        #boolean_encoder_mask = (attention_mask == 1).to(device)
    
        return encoder_hidden_states.mean(dim=1) # mean pooling
    
    def forward(self, caption, caption_tts):
        #return torch.zeros(len(caption), 512)
        return self.get_embedding(caption, caption_tts)
        

class EmbeddingDataset(torch.utils.data.Dataset):
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
            caption_tts = torch.tensor(caption_tts).mean(dim=1)

        tags = self.filename_to_label[audio_id]
        tag_idx = [self.label_to_index[tag] for tag in tags.split(",")]
        event_onehot = np.zeros(self.nlabels)   
        event_onehot[tag_idx] = 1
        
        return caption, caption_tts, event_onehot, audio_id
    
    def __len__(self):
        return len(self.data)

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
    "hidden_layers": [1, 2],
    "hidden_dim": [1024],
    "dropout": [0.1],
    "hidden_norm": [torch.nn.BatchNorm1d],
    "initialization": [torch.nn.init.xavier_uniform_],
    "norm_after_activation": [False],

    "max_epochs": [500],
    "lr": [3.2e-3],
    "batch_size": [1024],
    "patience": [20],

    "sound_event_meta": ["data/AudioSet/metadata/class_labels_indices.csv"],
    "sound_event_annotation": ["data/AudioSet/unbalanced_train/unbalanced_train_segments.csv"],

    "train_file_path": ["data/audiocaps/train/text.json"],
    "val_file_path": ["data/audiocaps/val/text.json"],
    "test_file_path": ["data/audiocaps/test/_text.json"],
}

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a diffusion model for text to audio generation task.")
    parser.add_argument("--embedding_type", "-t", type=str, default='caption_tts', choices=['caption', 'caption_tts'])
    parser.add_argument("--embedding_extractor", "-e", type=str, default='dac', choices=['CLAP', 'T5', 'hubert', 'wavlm', 'dac'])
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--infer", action='store_true')
    parser.add_argument("--layer", "-l", type=int, default=1)
    parser.add_argument("--ckpt", "-c", type=str, default="output/caption_CLAP/exp0_earlystop_epoch64.pt")
    parser.add_argument("--device", "-d", type=str, default="cuda:0")
    args = parser.parse_args()
    return args

if __name__=="__main__":
    
    args = parse_args()
    if args.output_dir is None:
        args.output_dir = f"output/{args.embedding_type}_{args.embedding_extractor}" 
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
            train_dataset = EmbeddingDataset(hyperparams["train_file_path"], hyperparams)
            train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=hyperparams["batch_size"])
            val_dataset = EmbeddingDataset(hyperparams["val_file_path"], hyperparams)
            val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=hyperparams["batch_size"])
            test_dataset = EmbeddingDataset(hyperparams["test_file_path"], hyperparams)
            test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=hyperparams["batch_size"])
            logger.info("Load dataset, done!")

            model = FullyConnectedPrediction(nfeatures=embedding_extractor.nfeatures, nlabels=train_dataset.nlabels, 
                                            conf=hyperparams).to(device)
            
            
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
                    caption, caption_tts, y, _ = batch
                    with torch.no_grad():
                        x = embedding_extractor(caption, caption_tts)
                    loss = model.training_step(x.to(device), y.to(device))                  
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
                        caption, caption_tts, y, _ = batch
                        x = embedding_extractor(caption, caption_tts)
                        output = model.eval_step(x.to(device), y.to(device))
                        y_true.append(y.numpy())
                        y_pred.append(output["prediction"].detach().cpu().numpy())
                    val_mAP, ap_scores = get_map(train_dataset.nlabels, y_true=np.concatenate(y_true, axis=0), y_pred=np.concatenate(y_pred, axis=0))
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
                    caption, caption_tts, y, _ = batch
                    x = embedding_extractor(caption, caption_tts)
                    output = model.eval_step(x.to(device), y.to(device))
                    y_true.append(y.numpy())
                    y_pred.append(output["prediction"].detach().cpu())
                test_mAP, ap_scores = get_map(train_dataset.nlabels, y_true=np.concatenate(y_true, axis=0), y_pred=np.concatenate(y_pred, axis=0))
            result.update({"Test mAP": test_mAP, "n_event": len(ap_scores)})
            logger.info(result)
            exp_result[f"exp{hyperparams_i}_result"] = result
        logger.info(exp_result)
    
    else: # Inference only
        hyperparams = list(ParameterGrid(hyperparams_candidate))[args.layer - 1]
        hyperparams.update(vars(args))
        test_dataset = EmbeddingDataset(hyperparams["test_file_path"], hyperparams)
        test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=hyperparams["batch_size"])
        model = FullyConnectedPrediction(nfeatures=embedding_extractor.nfeatures, nlabels=test_dataset.nlabels, 
                                            conf=hyperparams).to(device)
        model.load_state_dict(torch.load(args.ckpt))
        model.eval()

        logger.info("***** Running test *****")       
        y_true, y_pred = [], []  
        result = {}   
        output_sample = []      
        with torch.no_grad():
            for step, batch in enumerate(test_dataloader):
                caption, caption_tts, y, audio_id = batch
                x = embedding_extractor(caption, caption_tts)
                output = model.eval_step(x.to(device), y.to(device))
                gt = y.detach().cpu().numpy()
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