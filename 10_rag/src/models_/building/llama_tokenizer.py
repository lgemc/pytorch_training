from transformers import AutoTokenizer

def load_tokenizer():
    """
    Load a Llama model from Hugging Face Hub.
    :param model_name: The name of the model to load.
    :return: The loaded model.
    """
    # Load the model
    model = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    model.pad_token = model.eos_token

    return model
