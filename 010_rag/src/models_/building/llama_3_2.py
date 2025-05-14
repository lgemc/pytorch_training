from transformers import LlamaForCausalLM

LLAMA_3_2 = "meta-llama/Llama-3-2"

def load(model_name: str):
    """
    Load a Llama model from Hugging Face Hub.
    :param model_name: The name of the model to load.
    :return: The loaded model.
    """
    # Load the model
    model = LlamaForCausalLM.from_pretrained(model_name)
    return model