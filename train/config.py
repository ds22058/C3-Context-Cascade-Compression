IGNORE_INDEX = -100

DEFAULT_PAD_TOKEN = "<|endoftext|>"
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<imgpad>"
DEFAULT_IM_START_TOKEN = "<img>"
DEFAULT_IM_END_TOKEN = "</img>"

RECONSTRUCT_PROMPT = "Repeat the text: "

SYSTEM_MESSAGE = (
    "<|im_start|>system\n"
    "You should follow the instructions carefully and explain your answers in detail."
)

ROLE_USER = "<|im_start|>user\n"
ROLE_ASSISTANT = "<|im_start|>assistant\n"
SEP_TOKEN = "<|im_end|>"
