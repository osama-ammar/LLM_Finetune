
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline
from sklearn.model_selection import train_test_split


# Step 1: Data Preparation
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length, device):
        # tokenization and encoding using the tokenizer.
        self.encodings = tokenizer(texts,
                                   # Ensures that any text longer than max_length is truncated to fit within the maximum length Example: If max_length=8 and the text has 10 tokens, it keeps only the first 8 tokens.
                                   truncation=True,
                                   # Ensures that all encoded sequences have the same length (defined by max_length). Shorter sequences are padded with a special [PAD] token to reach max_length.
                                   padding="max_length",
                                   max_length=max_length,
                                   return_tensors="pt")  # Converts the output into PyTorch tensors for easy integration with PyTorch models.
        self.device = device

    def __len__(self):
        return len(self.encodings.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings.input_ids[idx].to(self.device),
            # The attention mask is a tensor that tells the model which parts of the input sequence are real data (tokens of the original text) and which parts are just padding.
            "attention_mask": self.encodings.attention_mask[idx].to(self.device)
        }


# we will fine tune the model over this text
texts = [
    "osama is working at atomica",
    "osama is a biophysicist.",
    "osama holds a master degree."
    "osama is working at atomica.ai",
    "osama is a physicist.",
    "osama holds a MSC degree."
]


def get_loaders(batch_size, tokenizer, device):

    # Split dataset into training and validation sets
    train_texts, val_texts = train_test_split(
        texts, test_size=0.3, random_state=42)

    # Create training and validation datasets and loaders
    train_dataset = TextDataset(
        train_texts, tokenizer, max_length=20, device=device)
    val_dataset = TextDataset(val_texts, tokenizer,
                              max_length=20, device=device)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
