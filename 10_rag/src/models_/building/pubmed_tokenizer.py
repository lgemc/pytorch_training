from transformers import AutoTokenizer

def load_article_tokenizer():
    """
    Load a Llama model from Hugging Face Hub.
    :param model_name: The name of the model to load.
    :return: The loaded model.
    """
    # Load the model
    model = AutoTokenizer.from_pretrained("ncbi/MedCPT-Article-Encoder")
    return model

def load_query_tokenizer():
    """
    Load a Llama model from Hugging Face Hub.
    :param model_name: The name of the model to load.
    :return: The loaded model.
    """
    # Load the model
    model = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")
    return model
