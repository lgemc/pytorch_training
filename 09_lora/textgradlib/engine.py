from textgrad import EngineLM
DEFAULT_PROMPT="You are a question answering model. Given the context and question, please provide the answer. Think step by step."
class Engine(EngineLM):
    """
    This class is a wrapper for the EngineLM class from textgrad.
    It provides an interface for loading and using language models.
    """

    def __init__(self, model, tokenizer, device="cuda", max_length=50):
        """
        Initialize the Engine with a model and tokenizer.

        Args:
            model: The language model to be used.
            tokenizer: The tokenizer for the language model.
            device: The device to run the model on (default is "cuda").
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(self.device)
        self.max_length = max_length
        self.model.eval()

    def generate(self, input_text):
        """
        Generate text using the model.

        Args:
            input_text: The input text to generate from.
            max_length: The maximum length of the generated text.

        Returns:
            The generated text.
        """
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=self.max_length,
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)