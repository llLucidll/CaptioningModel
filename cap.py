from functools import partial

import torch    
from PIL import Image
from torch.utils.data import Dataset
import os
from nltk.translate.bleu_score import corpus_bleu
from transformers import Seq2SeqTrainer
from transformers import default_data_collator
from transformers import VisionEncoderDecoderModel
from transformers import Seq2SeqTrainingArguments
from transformers import ViTFeatureExtractor, GPT2Tokenizer



class Args:
    """Configuration.
    """
    # Encoder-Decoder for captioning
    encoder = "ViT-Base"
    decoder = "GPT-2"

    # Dataset path
    root_dir = "" #Replace with path

    YOUR_CCID = "asunil"
    name = f"cap-vlm-{YOUR_CCID}"

    # Hyperparameters
    batch_size = 16
    lr = 5e-5
    epochs = 3

    # Generation cfgs
    num_beams = 5
    max_length = 45    


    # Train ops
    logging_steps = 50

class FlickrDataset(Dataset):
    def __init__(
        self, 
        args, 
        processor, 
        tokenizer,
        mode: str = "train",
        ):
        assert mode in ["train", "val", "test"]
        self.args = args
        self.processor = processor
        self.tokenizer = tokenizer

        self.img_paths, self.captions = [], []
        data_file = os.path.join(args.root_dir, f'{mode}.txt')
        with open(data_file, 'r') as file:
            for line in file:
                parts = line.strip().split(';')
                img_path, caption = parts
                self.img_paths.append(os.path.join(args.root_dir, 'Images', img_path))
                self.captions.append(caption)
    

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        caption = self.captions[idx]

        caption = f"{self.tokenizer.bos_token} {caption} {self.tokenizer.eos_token}"

        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            print(f"Error: Image file not found: {img_path}. Skipping this image.")
            return self.__getitem__((idx + 1) % len(self))

        pixel_values = self.processor(images = image, return_tensors = "pt").pixel_values


        encoding = {
            "pixel_values": pixel_values.squeeze(0),  # Remove batch dimension
            "labels": self.tokenizer(
                caption,
                padding="max_length",
                truncation=True,
                max_length=self.args.max_length,
                return_tensors="pt"
            ).input_ids.squeeze(0),  # Remove batch dimension
            "path": img_path,
            "captions": caption,
        }

        return encoding

    
def train_cap_model(args):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    processor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")

    # Reference: https://huggingface.co/docs/transformers/en/model_doc/vision-encoder-decoder
    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained("google/vit-base-patch16-224", "gpt2")
    model = model.to("cuda")    # NOTE: Send your model to GPU

  
    # NOTE: The format of GPT2 inputs:
    # <|endoftext|> + article + " TL;DR: " + summary + <|endoftext|>
    # For captoning, we want:
    # <|beginoftext|> + caption + <|endoftext|>
    # followed by a number of paddings "<|pad|>"

    tokenizer.add_special_tokens({'bos_token': '<|beginoftext|>', 'pad_token': '<|pad|>'})
    model.decoder.resize_token_embeddings(len(tokenizer))

    # Load train/val dataset
    train_dataset = FlickrDataset(args, tokenizer=tokenizer, processor=processor, mode="train")
    val_dataset = FlickrDataset(args, tokenizer=tokenizer, processor=processor, mode="val")

   
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.generation_config.bos_token_id = tokenizer.bos_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id


    model.generation_config.max_length = args.max_length #None
    model.generation_config.num_beams = args.num_beams #None

    training_args = Seq2SeqTrainingArguments(
        output_dir = args.name,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        predict_with_generate=True,
        fp16=True,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=args.logging_steps,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        report_to="none"
    )

    # Instantiate seq2seq model trainer
    compute_metrics = partial(compute_bleu_score, tokenizer=tokenizer)
    trainer = Seq2SeqTrainer(
        tokenizer=tokenizer,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=default_data_collator,
    )

    
    trainer.train()
    trainer.save_model(args.name)
    

def load_trained_model(
    ckpt_dir: str,
    ):
   
    config = VisionEncoderDecoderModel.from_pretrained(ckpt_dir).config

    processor = ViTFeatureExtractor.from_pretrained(ckpt_dir)

    tokenizer = GPT2Tokenizer.from_pretrained(ckpt_dir)

    model = VisionEncoderDecoderModel.from_pretrained(ckpt_dir)
    if torch.cuda.is_available():
        model = model.cuda()

    return model, processor, tokenizer

def inference(
    img_path,
    model, 
    processor,
    tokenizer,
    ):
    
    image = Image.open(img_path).convert("RGB")
    img_tensor = processor(images=image, return_tensors="pt").pixel_values   # TODO: Preproces the image

  
    if torch.cuda.is_available():
        img_tensor = img_tensor.cuda()

    generated_ids = model.generate(img_tensor, max_length=tokenizer.model_max_length, num_beams=5)

    # Tokens -> Str
    generated_caption = tokenizer.decode(generated_ids[0], skip_special_tokens = True)

    return generated_caption

def compute_bleu_score(pred, tokenizer):
    """
    Compute BLEU score.
    https://www.geeksforgeeks.org/nlp-bleu-score-for-evaluating-neural-machine-translation-python/
    https://cloud.google.com/translate/automl/docs/evaluate#interpretation
    https://www.nltk.org/api/nltk.translate.bleu_score.html
    """

    pred_ids = pred.predictions
    labels_ids = pred.label_ids#.squeeze(1)

    # Decode predictions and labels while handling special tokens and padding
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == tokenizer.pad_token_id] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    # Prepare data for BLEU score calculation
    pred_bleu = [line.split() for line in pred_str]
    label_bleu = [[line.split()] for line in label_str]

    # Calculate BLEU score
    bleu_output = corpus_bleu(label_bleu, pred_bleu)
    bleu_score = round(bleu_output, 4)
    print("BLEU:", bleu_score)

    return {
        "bleu_score": bleu_score
    }
