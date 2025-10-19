import torch
import torch.nn as nn

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from Qformer import BertConfig, BertLMHeadModel


try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
    DEVICE_TYPE = "npu"
except ModuleNotFoundError:
    DEVICE_TYPE = "cuda"

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
                 hiddin_size: int = 1024, speech_width: int = 1024, freeze_QFormer: bool = True,
                 load_from_pretrained: str = None):
        super().__init__()
        
        self.Qformer_model_name = Qformer_model_name
        self.audio_Qformer, self.audio_query_tokens, encoder_config = self.init_Qformer(num_query_token=num_query_token,  speech_width=speech_width)
        self.audio_Qformer.cls = None
        self.audio_Qformer.bert.embeddings.word_embeddings = None
        self.audio_Qformer.bert.embeddings.position_embeddings = None
        for layer in self.audio_Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        
        self.freeze_QFormer = freeze_QFormer
        if freeze_QFormer:
            for name, param in self.audio_Qformer.named_parameters():
                param.requires_grad = False
            self.audio_Qformer.eval()
            self.audio_query_tokens.requires_grad = False

        self.hiddin_projection = torch.nn.Linear(encoder_config.hidden_size, hiddin_size)
        #torch.nn.init.xavier_uniform_(self.hiddin_projection.weight, gain=torch.nn.init.calculate_gain("relu"))

        if load_from_pretrained:
            state_dict = torch.load(load_from_pretrained)
            del_key = ["projection.weight", "projection.bias"]
            del_state_dict = {k:v for k, v in state_dict.items() if k not in del_key}
            self.load_state_dict(del_state_dict)
            print("Load adaptor_model_pt from", load_from_pretrained)     
        
        
    def init_Qformer(self, num_query_token, speech_width, num_hidden_layers=2, cross_attention_freq=2):
        encoder_config = BertConfig.from_pretrained(self.Qformer_model_name)
        encoder_config.num_hidden_layers = num_hidden_layers
        encoder_config.encoder_width = speech_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel(config=encoder_config)
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
        return audio_hidden

    def forward(self, batch) -> torch.Tensor:   
        with torch.no_grad(), torch.amp.autocast(
            device_type=DEVICE_TYPE, enabled=False
        ):
            x = self.hidden(batch)
        x = self.hiddin_projection(x)

        mask = torch.ones(x.shape[:2])
        mask = (mask == 1).to(x.device)
        return {"output": x, "mask": mask}
        
        
if __name__ == '__main__':
    text_encoder = T5TextEncoder()
    text = ["a man is speaking", "a woman is singing while a dog is barking"]
    text_encoder.eval()
    with torch.no_grad():
        output = text_encoder(text)
