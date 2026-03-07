"""
Wandb setup
"""

import wandb
import os

def setup_wandb(api_key):
    if api_key:
        os.environ["WANDB_API_KEY"] = api_key
        wandb.login(key=api_key, relogin=True)
    else:
        os.environ["WANDB_MODE"] = "disabled"