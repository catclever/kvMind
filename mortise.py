import os
from transformers import AutoModelForCausalLM
import read_weights
import torch
from safetensors import safe_open
import pickle

def check_cp(checkpoint_dir):
    # 打开 safetensors 文件
    with safe_open(f"{checkpoint_dir}/model.safetensors", framework="pt") as f:
        # 列出所有张量的名称和形状
        for key in f.keys():
            tensor = f.get_tensor(key)
            print(f"Key: {key}, Shape: {tensor.shape}")

    # 查看键和数据结构
    # for key, value in checkpoint.items():
    #     print(f"Key: {key}, Type: {type(value)}")
    #     if isinstance(value, torch.Tensor):
    #         print(f"  Shape: {value.shape}")

def check_pth(pth_file):
    # 加载 .pth 文件
    checkpoint = torch.load(pth_file, map_location='mps')

    # 查看键和数据结构
    for key, value in checkpoint.items():
        print(f"Key: {key}, Type: {type(value)}")
        if isinstance(value, torch.Tensor):
            print(f"  Shape: {value.shape}")

def compare_pth(file_1, file_2):
    # 加载两个 pth 的权重
    pth_1 = torch.load(file_1)
    pth_2 = torch.load(file_2)

    # 初始化全局差异
    total_diff = 0.0
    num_params = 0

    # 计算每个参数的差异，并累加
    for key in pth_1.keys():
        if key in pth_2:
            diff = torch.norm(pth_1[key] - pth_2[key])
            total_diff += diff.item()
            num_params += 1
            print(f"Difference in {key}: {diff}")

    # 计算平均差异
    average_diff = total_diff / num_params if num_params > 0 else 0.0
    print(f"Total Difference: {total_diff}")
    print(f"Average Difference: {average_diff}")

def load_embedding_checkpoint(model, check_point:str=None, embedding_path:str=None, lm_head_path:str=None, framework='pt', device='mps'):
    model = AutoModelForCausalLM.from_pretrained(model).to(device)
    if embedding_path:
        # model = AutoModelForCausalLM.from_pretrained(model, load_in_8bit=False,)
        embedding_weights = read_weights.read_safetensor(embedding_path, device=device)
        model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.clone())
        model.load_state_dict(embedding_weights, strict=False)
        if lm_head_path:
            lm_head_weights = read_weights.read_safetensor(lm_head_path, device=device)
            model.load_state_dict(lm_head_weights, strict=False)
    elif check_point:
        if lm_head_path:
            if os.path.exists(f'{check_point}/model.safetensors'):
                weights = read_weights.read_safetensor(f'{check_point}/model.safetensors', framework=framework, device=device)
            else:
                weights = read_weights.merge_safetensors(check_point)
            model.load_state_dict(weights, strict=False)
            lm_head_weights = read_weights.read_safetensor(lm_head_path, device=device)
            model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.clone())
            model.load_state_dict(lm_head_weights, strict=False)
        else:
            model = AutoModelForCausalLM.from_pretrained(check_point)
    elif lm_head_path:
        
        lm_head_weights = read_weights.read_safetensor(lm_head_path, device=device)
        model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.clone())
        model.load_state_dict(lm_head_weights, strict=False)
    return model


class TokenGenerator:
    """
    通用的Token生成器类，可用于各种模型和场景，支持继承和自定义
    """
    def __init__(
        self,
        model,
        tokenizer,
        device=None,
    ):
        """
        初始化Token生成器
        
        参数:
            model: 语言模型
            tokenizer: 分词器
            device: 计算设备
            max_context_length: 最大上下文长度
        """
        # 确定设备
        if device is None:
            self.device = model.device if hasattr(model, 'device') else (
                torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
            )
        else:
            self.device = torch.device(device)
        
        self.model = model
        self.tokenizer = tokenizer
    
    def prepare_stop_tokens(self, stop_tokens=None):
        """
        准备停止生成的token列表
        
        参数:
            stop_tokens: 自定义停止token列表
            
        返回:
            完整的停止token列表
        """
        if stop_tokens is None:
            stop_tokens = []
        if self.tokenizer.eos_token_id is not None and self.tokenizer.eos_token_id not in stop_tokens:
            stop_tokens.append(self.tokenizer.eos_token_id)
        return stop_tokens
    
    def prepare_initial_input(self, initial_input=None, prompt=None, initial_cache=None):
        """
        准备初始输入和缓存
        
        参数:
            initial_input: 初始输入token IDs
            prompt: 文本提示
            initial_cache: 初始KV缓存
            
        返回:
            处理后的当前输入和缓存
        """
        current_cache = initial_cache
        
        if prompt is not None:
            # 编码提示文本
            inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(self.device)
            current_input = inputs.input_ids
            
            # 前向传播处理提示
            with torch.no_grad():
                outputs = self.model(
                    input_ids=current_input,
                    past_key_values=current_cache,
                    use_cache=True,
                )
            
            # 更新缓存
            current_cache = outputs.past_key_values
            
            # 获取最后一个token作为下一步生成的输入
            current_input = current_input[:, -1].unsqueeze(-1)
        elif initial_input is not None:
            current_input = initial_input.to(self.device) if hasattr(initial_input, 'to') else torch.tensor(
                [[initial_input]], dtype=torch.long, device=self.device
            )
        else:
            # 如果没有提供输入，使用BOS token
            current_input = torch.tensor(
                [[self.tokenizer.bos_token_id if hasattr(self.tokenizer, 'bos_token_id') and self.tokenizer.bos_token_id is not None else self.tokenizer.eos_token_id]],
                dtype=torch.long,
                device=self.device
            )
        
        return current_input, current_cache
    
    def sample_next_token(self, logits, temperature=0.7, top_p=0.9):
        """
        采样下一个token
        
        参数:
            logits: 模型输出的logits
            temperature: 采样温度
            top_p: 核采样参数
            
        返回:
            采样得到的下一个token
        """
        # 应用温度
        if temperature > 0:
            logits = logits / temperature
        
        # 应用top-p采样
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
            
            # 移除概率累积超过top_p的token
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            # 使用scatter操作实现mask
            indices_to_remove = torch.zeros_like(logits, dtype=torch.bool).scatter_(
                dim=-1, index=sorted_indices, src=sorted_indices_to_remove
            )
            logits[indices_to_remove] = -float('Inf')
        
        # 采样获取下一个token
        probs = torch.nn.functional.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        return next_token
    
    def generate(
        self,
        initial_input=None,
        initial_cache=None,
        prompt=None,
        max_tokens=100,
        temperature=0.7,
        top_p=0.9,
        stop_tokens=None,
    ):
        """
        生成token序列
        
        参数:
            initial_input: 初始输入token IDs (可选)
            initial_cache: 初始KV缓存 (可选)
            prompt: 文本提示 (如果提供，会覆盖initial_input)
            max_tokens: 最大生成token数
            temperature: 采样温度
            top_p: 核采样参数
            stop_tokens: 停止生成的token列表
            
        返回:
            生成的token列表和最终的KV缓存
        """
        # 准备停止token
        stop_tokens = self.prepare_stop_tokens(stop_tokens)
        
        # 准备初始输入和缓存
        current_input, current_cache = self.prepare_initial_input(
            initial_input, prompt, initial_cache
        )
        
        # 开始生成
        generated_tokens = []
        
        for i in range(max_tokens):
            
            # 前向传播
            with torch.no_grad():
                outputs = self.model(
                    input_ids=current_input,
                    past_key_values=current_cache,
                    use_cache=True
                )
                
                # 更新缓存
                current_cache = outputs.past_key_values
                
                # 获取下一个token的概率分布
                logits = outputs.logits[:, -1, :]
                
                # 采样下一个token
                next_token = self.sample_next_token(logits, temperature, top_p)
                
                # 添加到生成列表
                token_id = next_token.item()
                generated_tokens.append(token_id)
                
                # 更新当前token
                current_input = next_token
                
                # 检查是否生成了停止标记
                if token_id in stop_tokens:
                    break
        
        return generated_tokens, current_cache

def decode_tokens(tokenizer, tokens, skip_special_tokens=True):
    """
    解码生成的token列表为文本
    
    参数:
        tokenizer: 分词器
        tokens: token ID列表
        skip_special_tokens: 是否跳过特殊token
        
    返回:
        解码后的文本
    """
    if isinstance(tokens, torch.Tensor):
        tokens = tokens.tolist()
    return tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)


def save_kv_cache(kv_cache, file_path):
    """
    保存KV缓存到文件
    
    参数:
        kv_cache: 要保存的KV缓存对象
        file_path: 保存的文件路径
    
    返回:
        成功保存返回True，否则返回False
    """
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        # 使用pickle保存缓存对象
        with open(file_path, 'wb') as f:
            pickle.dump(kv_cache, f)
        
        print(f"KV缓存已成功保存到: {file_path}")
        return True
    except Exception as e:
        print(f"保存KV缓存时出错: {e}")
        return False

def load_kv_cache(file_path, device=None):
    """
    从文件加载KV缓存
    
    参数:
        file_path: KV缓存文件路径
        device: 加载到的设备，如果为None则使用原始设备
    
    返回:
        加载的KV缓存对象，加载失败则返回None
    """
    try:
        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            return None
        
        # 使用pickle加载缓存对象
        with open(file_path, 'rb') as f:
            kv_cache = pickle.load(f)
        
        # 如果指定了设备，将缓存移动到该设备
        if device is not None:
            kv_cache = move_kv_cache_to_device(kv_cache, device)
        
        print(f"KV缓存已成功从 {file_path} 加载")
        return kv_cache
    except Exception as e:
        print(f"加载KV缓存时出错: {e}")
        return None

def move_kv_cache_to_device(kv_cache, device):
    """
    将KV缓存移动到指定设备
    
    参数:
        kv_cache: KV缓存对象
        device: 目标设备
    
    返回:
        移动到新设备的KV缓存对象
    """
    device = torch.device(device)
    
    # 处理不同类型的KV缓存
    if hasattr(kv_cache, 'to') and callable(kv_cache.to):
        # 如果缓存对象有to方法，直接调用
        return kv_cache.to(device)
    elif isinstance(kv_cache, tuple) and all(isinstance(item, tuple) for item in kv_cache):
        # 处理嵌套元组结构 (常见于Transformer模型的KV缓存)
        return tuple(
            tuple(
                tensor.to(device) if isinstance(tensor, torch.Tensor) else tensor
                for tensor in layer
            )
            for layer in kv_cache
        )
    else:
        print(f"无法识别的KV缓存类型: {type(kv_cache)}")
        return kv_cache
