import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import math
import pandas as pd

class resnetData:
    def __init__(self, data, pkl_file):
        self.data = data
        self.encoded_image_features = pd.read_pickle(pkl_file)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        caption_sequence = self.data.iloc[idx]['tokenized_captions']
        target_sequence = caption_sequence[1:] + [0]
        
        image_name = self.data.iloc[idx]['image']
        image_tensor = self.encoded_image_features[image_name]
        image_tensor = image_tensor.permute(0, 2, 3, 1) # Reshape to (batch_size, height, width, channels)
        image_tensor_view = image_tensor.view(image_tensor.size(0), -1, image_tensor.size(3)) # Reshape to (batch_size, height*width, channels)

        return torch.Tensor(caption_sequence), torch.Tensor(target_sequence), image_tensor_view
    
def get_resnet_dataloader(data, pkl_file, batch_size=32):
    dataset_resnet = resnetData(data, pkl_file)
    dataloader_resnet = DataLoader(dataset_resnet, batch_size = batch_size, shuffle = True)
    return dataloader_resnet

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_length, device, dropout = 0.1):
        super(PositionalEmbedding, self).__init__()
        self.dropout = nn.Dropout(p = dropout)
        self.device = device

        positional_embedding = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length, dtype = torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        positional_embedding[:, 0::2] = torch.sin(position*div_term)
        positional_embedding[:, 1::2] = torch.cos(position*div_term)
        positional_embedding = positional_embedding.unsqueeze(0)
        self.register_buffer("positional_embedding", positional_embedding)

    def forward(self, x):
        if self.positional_embedding.size(0) < x.size(0):
            self.positional_embedding = self.positional_embedding.repeat(x.size(0), 1, 1).to(self.device)
        self.positional_embedding = self.positional_embedding[:x.size(0), :, :]

        x = x + self.positional_embedding
        return self.dropout(x)
    
class decoderTransformer(nn.Module):
    def __init__(self, n_head, n_decoder_layers, vocab_size, embedding_size, max_length, device):
        super(decoderTransformer, self).__init__()
        self.pos_encoder = PositionalEmbedding(embedding_size, max_length, device, 0.1)
        self.TransformerDecoderLayer = nn.TransformerDecoderLayer(d_model =  embedding_size, nhead = n_head)
        self.TransformerDecoder = nn.TransformerDecoder(decoder_layer = self.TransformerDecoderLayer, num_layers = n_decoder_layers)
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(vocab_size , embedding_size)
        self.last_linear_layer = nn.Linear(embedding_size, vocab_size)
        self.encoder_layer = nn.Linear(2048, embedding_size)  # Assuming ResNet-50 features of size 2048
        self.device = device
        self.max_length = max_length
        self.pad_token = 0
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.last_linear_layer.bias.data.zero_()
        self.last_linear_layer.weight.data.uniform_(-initrange, initrange)

    def generate_Mask(self, size, decoder_inp):
        decoder_input_mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        decoder_input_mask = decoder_input_mask.float().masked_fill(decoder_input_mask == 0, float('-inf')).masked_fill(decoder_input_mask == 1, float(0.0))

        decoder_input_pad_mask = decoder_inp.float().masked_fill(decoder_inp == 0, float(0.0)).masked_fill(decoder_inp > 0, float(1.0))
        decoder_input_pad_mask_bool = decoder_inp == 0

        return decoder_input_mask, decoder_input_pad_mask, decoder_input_pad_mask_bool

    def forward(self, encoded_image, decoder_inp):
        encoded_image = encoded_image.permute(1,0,2)
        encoded_image = self.encoder_layer(encoded_image)

        decoder_inp_embed = self.embedding(decoder_inp)* math.sqrt(self.embedding_size)
        
        decoder_inp_embed = self.pos_encoder(decoder_inp_embed)
        decoder_inp_embed = decoder_inp_embed.permute(1,0,2)
        

        decoder_input_mask, decoder_input_pad_mask, decoder_input_pad_mask_bool = self.generate_Mask(decoder_inp.size(1), decoder_inp)
        decoder_input_mask = decoder_input_mask.to(self.device)
        decoder_input_pad_mask = decoder_input_pad_mask.to(self.device)
        decoder_input_pad_mask_bool = decoder_input_pad_mask_bool.to(self.device)
        

        decoder_output = self.TransformerDecoder(tgt = decoder_inp_embed, memory = encoded_image, tgt_mask = decoder_input_mask, tgt_key_padding_mask = decoder_input_pad_mask_bool)
        
        final_output = self.last_linear_layer(decoder_output)

        return final_output,  decoder_input_pad_mask