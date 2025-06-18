import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from PIL import Image
from imageapp import resnet_encoder
import random
import pandas as pd
from imageapp.utilis import vocab_info

def process_image(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    image = Image.open(image_path).convert("RGB")
    print(f"Image size: {image.size}")
    encoded = resnet_encoder.single_image_feature(image, device).to(device)
    encoded = encoded.permute(0, 2, 3, 1)
    encoded = encoded.view(encoded.size(0), -1, encoded.size(-1))

    model = torch.load("./imageapp/BestModel", map_location=device, weights_only=False)
    print("Model loaded and set to evaluation mode.")
    model.eval()
    model = model.to(device)
    input_sequence = [vocab_info.pad_token] * vocab_info.max_seq_length
    input_sequence[0] = vocab_info.start_token
    input_sequence = torch.tensor(input_sequence).unsqueeze(0).long().to(device)
    predicted_sequence = []

    print("Model is on device:", next(model.parameters()).device)
    print("Encoded image device:", encoded.device)
    print("Input sequence device:", input_sequence.device)

    with torch.no_grad():
        for eval_iter in range(vocab_info.max_seq_length):
            output, padding_mask = model(encoded, input_sequence)
            output = output[eval_iter, 0, :]

            indices = torch.topk(output, k=1).indices.tolist()
            next_word_idx = random.choice(indices)
            next_word = pd.read_pickle("imageapp/utilis/index_to_word.pkl")[next_word_idx]

            input_sequence[:, eval_iter+1] = next_word_idx

            if next_word == "<end>":
                break

            predicted_sequence.append(next_word)

    caption = " ".join(predicted_sequence)
    return caption


    

if __name__ == "__main__":
    image_path = "media/uploads/monkey.jpg"
    caption = process_image(image_path)
    print("Generated caption:", caption)
