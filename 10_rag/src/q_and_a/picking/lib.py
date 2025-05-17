from typing import Callable, List, Any

PickerType = Callable[
    [Any],  # The response from the language model
    int  # Return option at the array
]