# header.py - 修复后的版本

import torch
import datetime
import types
import deepspeed
# 注释掉过时的 transformers.deepspeed 导入
# from transformers.deepspeed import HfDeepSpeedConfig
import transformers
import numpy as np
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import autocast, GradScaler
from torch.nn import DataParallel
from torch.optim import lr_scheduler
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import os
import re
import math
import random
import json
import time
import logging
from copy import deepcopy
import ipdb
import argparse
# from model.ImageBind import data
from transformers import LlamaTokenizer, LlamaForCausalLM, LlamaConfig
from torch.nn.utils.rnn import pad_sequence
from peft import LoraConfig, TaskType, get_peft_model

# 条件导入 HfDeepSpeedConfig（如果可用）
try:
    from transformers.deepspeed import HfDeepSpeedConfig
except ImportError:
    # 对于较新的 transformers 版本，HfDeepSpeedConfig 已被移除
    # 可以在需要时直接使用 deepspeed 配置文件路径
    HfDeepSpeedConfig = None

logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'