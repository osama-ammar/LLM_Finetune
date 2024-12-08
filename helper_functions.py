import torch
import mmap
import random
import pickle
from torch.nn import Module

# from onnx import load as load_onnx
# from onnx.checker import check_model
# import onnxruntime as onnxrt


"""

Block size: Controls how many tokens per sequence (e.g., 8 tokens in this case).
Batch size: Controls how many sequences are processed together (e.g., 2 sequences at a time).
n_embd: Controls the dimensionality of the embedding space for each token (e.g., 768).

ex):
tokenizer(words):
        ---output--> ["I", "love", "learning", "about", "NLP", "and", "how", "models", "work", "!", "I", "also", "enjoy", "reading", "about", "transformers", "and", "embeddings", "."]

block size = 8  : 
        Block 1: ["I", "love", "learning", "about", "NLP", "and", "how", "models"]
        Block 2: ["work", "!", "I", "also", "enjoy", "reading", "about", "transformers"]
        Block 3: ["and", "embeddings", "."]
        
batch size =2 :
        Batch 1:
        Sequence 1: ["I", "love", "learning", "about", "NLP", "and", "how", "models"]
        Sequence 2: ["work", "!", "I", "also", "enjoy", "reading", "about", "transformers"]

        Batch 2:
        Sequence 3: ["and", "embeddings", "."]

embedding :
        Token: "I" → [0.12, 0.45, -0.33, ..., 0.98] (size: 768)
        Token: "love" → [-0.25, 0.77, 0.02, ..., -0.14] (size: 768)

Block 1 Embedding Matrix:
        [[ 0.12,  0.45, -0.33, ...,  0.98],  # "I"
        [-0.25,  0.77,  0.02, ..., -0.14],   # "love"
        [ 0.30, -0.21,  0.56, ...,  0.49],   # "learning"
        ...
        [ 0.05,  0.88, -0.31, ..., -0.02]]   # "models"
        Matrix size: 8 [tokens] × 768 [n_embd] dimensions


==================================================================================

mmap,  allows efficient access to large files without loading them entirely into memory. It:

    Randomly selects a position in the file.
    Reads a block of text of size block_size * batch_size - 1.
    Decodes the block, ignoring errors and replacing \r.
    Tokenizes the block and converts it into a PyTorch tensor.

"""


# ckaracter tokenizer
def char_tokenizer(input_text, training_chars, mode):
    # chars = "abcdefghijklmnopqrstuvwxyz " # we will use this beacuase any character will not come outside this
    if mode == "encoder":
        string_to_int = {
            ch: i for i, ch in enumerate(training_chars)
        }  # string_to_int = {'a': 0, 'b': 1, ..., 'z': 25, ' ': 26}
        string_to_int['<UNK>'] = len(string_to_int)  # Add an unknown token to handle unknown character
        encode = lambda input_text:[string_to_int.get(char, string_to_int['<UNK>']) for char in input_text
        ]  # converting input string into integers based on string_to_int dict map
        return encode(input_text)

    if mode == "decoder":
        # in this mode input_text is an encoded text ex([0, 7,2,5,9,4,3..])
        int_to_string = {i: ch for i, ch in enumerate(training_chars)}
        decode = lambda input_text: "".join([int_to_string[i] for i in input_text])
        return decode(input_text)


# word tokenizer
def word_tokenizer(input_text, training_text, mode):
    # convering text into list of words , get unique words only to avoid repetition
    training_text = set(training_text.split())
    if mode == "encoder":
        string_to_int = {word: i for i, word in enumerate(training_text)}
        encode = lambda input_text: [string_to_int[word] for word in input_text]
        return encode(input_text)

    if mode == "decoder":
        # in this mode input_text is an encoded text ex([0, 7,2,5,9,4,3..])
        int_to_string = {i: word for i, word in enumerate(training_text)}
        decode = lambda input_text: "".join([int_to_string[i] for i in input_text])
        return decode(input_text)


def get_random_chunk(chars, batch_size, block_size, split="train"):
    filename = "data/output_train.txt" if split == "train" else "data/output_val.txt"
    with open(filename, "rb") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            # Determine the file size and a random position to start reading
            file_size = len(mm)
            
            #============================================================
            text = f.read()
            chars = sorted(list(set(text)))  # Extract unique characters
            chars.append('<UNK>')  # Optionally add an unknown token
            #============================================================


            if block_size * batch_size > file_size:
                raise ValueError(
                    "The requested block size is larger than the file size."
                )

            start_pos = random.randint(0, (file_size) - block_size * batch_size)

            # Seek to the random position and read the block of text
            mm.seek(start_pos)
            block = mm.read(block_size * batch_size - 1)

            # Decode the block to a string, ignoring any invalid byte sequences
            decoded_block = block.decode("utf-8", errors="ignore").replace("\r", "")

            # Train and test splits
            data = torch.tensor(
                char_tokenizer(decoded_block, chars, mode="encoder"), dtype=torch.long
            )

    return data


####################
# Model operations
####################


def save_weights(
    model: Module,
    path: str,
    mode: str = "pth"
) -> None:

    if mode == "pkl":
        with open("model_pickled.pkl", "wb") as f:
            pickle.dump(model, f)
    else:
        state = {
        "state_dict": model.to("cpu").state_dict(),
            }
        torch.save(state, "model.pth")
        
    print("model saved")


def load_ckp(checkpoint_path, model, optimizer):
    """to load model and optimizer last states to resume training if needed"""

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return model, optimizer, checkpoint["epoch"]


# def onnx_export(model, dummy_input, save_path):
#     """export ....."""

#     # send the dummy input to cpu
#     dummy_input = dummy_input.to("cpu")

#     input_names = ["input"]
#     output_names = ["output"]
#     # to make the onnx model accept different input batch sizes
#     dynamic_axes = {"input": {0: "batch_size"}, "output": {0: "batch_size"}}

#     torch.onnx.export(
#         model,
#         dummy_input,
#         save_path,
#         input_names=input_names,
#         output_names=output_names,
#         dynamic_axes=dynamic_axes,
#         export_params=True,
#         keep_initializers_as_inputs=False,
#         do_constant_folding=True,  # Constant folding is a technique that can be used to optimize ONNX models. It involves statically computing parts of the graph that rely only on constant initializers. This eliminates the need to compute them during runtime, which can improve the performance of the model.
#         opset_version=16,  # onnx version to export to
#     )

#     # Checks
#     model_onnx = load_onnx(save_path)  # load onnx model
#     check_model(model_onnx)  # check onnx model
#     print("Model exported to ONNX format.")


# def use_onnx(onnx_model_path, input):
#     """usi ....."""

#     # Run the ONNX model with the dummy input tensor
#     session = onnxrt.InferenceSession(onnx_model_path)

#     input_names = ["input"]
#     output_names = ["output"]
#     result = session.run(output_names, {input_names: input})

#     # Print the output of the ONNX model
#     print(result)
