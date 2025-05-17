def build_tokenizer_function(
        tokenizer
) -> callable:
    return lambda query: tokenizer(
            query,
            max_length=800,
            truncation=True,
            padding="max_length",
            add_special_tokens=False,
            return_tensors="np",
        )["input_ids"].astype("float32")