from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from transformers import GPT2Tokenizer, ViTFeatureExtractor

from cap_submission import *

def inspect_data(
    args,
    n_imgs: int = 2,
    ):
    
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.pad_token = tokenizer.eos_token
    processor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")

    dataset = FlickrDataset(args, tokenizer=tokenizer, processor=processor)
    indices = np.random.randint(0, len(dataset), size=(n_imgs, ))
    # Visualize with matplotlib
    for i, idx in enumerate(indices):
        encoding = dataset[idx]
        print(encoding["labels"])
        print(encoding["captions"])
        img = np.array(Image.open(encoding["path"]))

        plt.subplot(1, n_imgs, i + 1)
        plt.imshow(img, cmap="gray")
        print()

    plt.show()
    del dataset



def main():
    args = Args()


  
    inspect_data(args)

    
    train_cap_model(args)

    
    model, processor, tokenizer = load_trained_model("./" + args.name)
    print(inference('./flickr8k/images/2631625732_75b714e685.jpg', model, processor, tokenizer))
    print(inference('./flickr8k/images/1215334959_b1970965f7.jpg', model, processor, tokenizer))
    print(inference('./flickr8k/images/3695064885_a6922f06b2.jpg', model, processor, tokenizer))


if __name__ == "__main__":
    main()