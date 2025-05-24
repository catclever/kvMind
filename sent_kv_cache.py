import torch
from transformers import DynamicCache, HybridCache
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, Gemma3ForConditionalGeneration
import re
import copy
from mortise import TokenGenerator
from gauge import print_hybrid_cache_layer_shapes
from model_manager import model_manager

# 加载模型和tokenizer
model_name = "google/gemma-3-4b-it"  # 你可以替换为其他Qwen模型
# model_name = "Qwen/Qwen2.5-1.5B-Instruct"  # 你可以替换为其他Qwen模型

# def load_model(model_name, device='cpu'):
#     """
#     加载模型和tokenizer
#     """
#     if "gemma-3" in model_name:
#         # 加载Gemma3模型
#         model = Gemma3ForConditionalGeneration.from_pretrained(
#             model_name,
#             device_map=device,  # 使用MPS或CPU
#             trust_remote_code=True,
#         )
#         processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
#         tokenizer = processor.tokenizer
#     else:
#         model = AutoModelForCausalLM.from_pretrained(
#             model_name,
#             device_map=device,  # 使用MPS或CPU
#             trust_remote_code=True,
#         )
#         tokenizer = AutoTokenizer.from_pretrained(
#             model_name,
#             trust_remote_code=True,
#         )
#     return model, tokenizer

def split_sentences_by_punctuation(text):
    """
    使用标点符号分割句子，但保持特殊标记的完整性
    特别处理：如果标点符号和<|im_end|>或<im_end>一起出现，将其视为同一个句子
    """
    # 处理常见的句末标点：。！？.!?
    pattern = r'([。！？\.!?])'
    
    # 预处理：找出所有特殊标记的位置，避免在这些位置分割
    special_tags = ["<|im_end|>", "<im_end>", "<IM_END>", "<eos>", "<end_of_turn>"]
    protected_regions = []
    
    for tag in special_tags:
        start = 0
        while True:
            pos = text.find(tag, start)
            if pos == -1:
                break
            # 保护标记前后的一小段区域，确保不会在标记内部分割
            protected_regions.append((max(0, pos-5), pos + len(tag) + 5))
            start = pos + len(tag)
    
    # 先按标点分割，然后重新组合标点和前面的文本
    parts = re.split(pattern, text)
    sentences = []
    
    # 组合文本和标点
    i = 0
    current_sentence = ""
    while i < len(parts):
        if i + 1 < len(parts) and re.match(pattern, parts[i+1]):
            # 当前部分加上后面的标点
            segment = parts[i] + parts[i+1]
            
            # 检查这个片段是否包含任何特殊标记或在保护区域内
            contains_special_tag = any(tag in segment for tag in special_tags)
            in_protected_region = False
            
            # 计算当前位置
            if current_sentence:
                current_pos = len(current_sentence)
            else:
                current_pos = 0
                
            # 检查是否在任何保护区域内
            for start, end in protected_regions:
                if current_pos >= start and current_pos <= end:
                    in_protected_region = True
                    break
                    
            # 如果包含特殊标记或在保护区域内，添加到当前句子
            if contains_special_tag or in_protected_region:
                current_sentence += segment
            else:
                # 否则完成当前句子并开始新句子
                sentences.append((current_sentence + segment).strip())
                current_sentence = ""
            
            i += 2
        else:
            # 处理没有标点的部分
            segment = parts[i]
            
            # 同样检查特殊标记和保护区域
            contains_special_tag = any(tag in segment for tag in special_tags)
            
            # 计算当前位置
            if current_sentence:
                current_pos = len(current_sentence)
            else:
                current_pos = 0
                
            # 检查是否在任何保护区域内
            in_protected_region = False
            for start, end in protected_regions:
                if current_pos >= start and current_pos <= end:
                    in_protected_region = True
                    break
            
            if contains_special_tag or in_protected_region:
                current_sentence += segment
            else:
                # 如果当前片段不为空，且我们有一个进行中的句子，添加到当前句子
                if segment.strip() and current_sentence:
                    current_sentence += segment
                # 否则，如果片段不为空，开始一个新句子
                elif segment.strip():
                    if current_sentence:
                        sentences.append(current_sentence.strip())
                    current_sentence = segment
            
            i += 1
    
    # 添加最后剩余的句子
    if current_sentence.strip():
        sentences.append(current_sentence.strip())
    
    # 最后清理：合并包含特殊标记的相邻句子
    i = 0
    while i < len(sentences) - 1:
        if any(tag in sentences[i] for tag in special_tags) or any(tag in sentences[i+1] for tag in special_tags):
            # 将下一个句子合并到当前句子
            sentences[i] = sentences[i] + " " + sentences[i+1]
            # 删除下一个句子
            sentences.pop(i+1)
        else:
            i += 1
    
    # 最终过滤空句子
    return [s.strip() for s in sentences if s.strip()]

def better_sequential_processing(sentences, model, tokenizer):
    """
    使用官方缓存类实现的串行处理方法，支持MPS
    """
    if not sentences:
        return None
    
    if "gemma" in model.config.model_type:
        # print(model.config)
        print(type(model.config.text_config))
        kv_cache = HybridCache(config=model.config.text_config, max_batch_size=1, max_cache_len=1024, device=model.device, dtype=model.dtype)
        # print(model.config)
        print(type(model.config.text_config))
        kv_cache = HybridCache(config=model.config.text_config, max_batch_size=1, max_cache_len=1024, device=model.device, dtype=model.dtype)
        
    else:
        kv_cache = DynamicCache()
   
    # 逐句处理
    for i, sentence in enumerate(sentences):
        print(f"处理第 {i+1}/{len(sentences)} 个句子: {sentence[:30]}{'...' if len(sentence) > 30 else ''}")
        
        # 编码当前句子
        inputs = tokenizer(sentence, return_tensors="pt").to(model.device)
        
        # 使用前向传播填充缓存
        with torch.no_grad():
            outputs = model(
                input_ids=inputs.input_ids,
                past_key_values=kv_cache,  # 传入当前缓存
                use_cache=True,
            )
            
            # 更新KV缓存 - DynamicCache会自动更新
            kv_cache = outputs.past_key_values
            if "gemma" in model.config.model_type:
                print_hybrid_cache_layer_shapes(kv_cache)
            
    return kv_cache

def process_prompt_with_cache(prompt, model, tokenizer, kv_cache):
    """
    处理提示并返回更新后的KV缓存和最后一个token ID
    """
    # 编码提示
    prompt_inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    
    # 检查提示输入是否为空
    if prompt_inputs.input_ids.shape[1] == 0:
        print("警告: 提示被编码为空序列，使用备用提示")
        prompt = "<|im_start|>assistant:"  # 简单的备用提示
        prompt_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    print(f"提示编码后的token数: {prompt_inputs.input_ids.shape[1]}")
    
    # 前向传播处理提示
    with torch.no_grad():
        outputs = model(
            input_ids=prompt_inputs.input_ids,
            past_key_values=copy.deepcopy(kv_cache),
            use_cache=True,
        )
    
    # 获取更新后的KV缓存
    updated_cache = outputs.past_key_values
    
    # 提取最后一个token ID用于开始生成
    last_token_id = prompt_inputs.input_ids[:, -1].unsqueeze(-1)
    
    return updated_cache, last_token_id, outputs


def gen_with_kv(kv_cache, model, tokenizer, prompt="根据以上信息，请回答：", eos_token="<|im_end|>", max_tokens=2000):
    """
    基于KV缓存的生成方法
    """
    # 处理提示并获取必要的输出
    print(f"处理提示: {prompt}")
    updated_cache, last_token_id, prompt_outputs = process_prompt_with_cache(prompt, model, tokenizer, kv_cache)
    
    # 获取eos token id用于结束生成
    im_end_tokens = tokenizer.encode(eos_token, add_special_tokens=False)
    eos_token_id = im_end_tokens[-1] if im_end_tokens else tokenizer.eos_token_id
    print(f"eos token id: {eos_token_id}")
    
    # 初始化TokenGenerator
    token_gen = TokenGenerator(model, tokenizer, device=model.device)
    
    # 使用TokenGenerator生成回答
    print(f"开始使用TokenGenerator生成回答...")
    
    # 准备停止生成的token列表
    stop_tokens = [eos_token_id]
    
    # 使用TokenGenerator生成
    generated_tokens, _ = token_gen.generate(
        initial_input=last_token_id,
        initial_cache=updated_cache,
        max_tokens=max_tokens,
        temperature=0.7,
        top_p=0.9,
        stop_tokens=stop_tokens
    )
    
    # 解码生成的tokens
    result = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return result

def sentence_based_reasoning(text, model, tokenizer, prompt="根据以上信息，请回答：", eos_token="<|im_end|>", max_tokens=2000):
    """
    基于句子的串行推理，使用TokenGenerator实现的生成方法
    """
    try:
        # 分割文本为句子
        sentences = split_sentences_by_punctuation(text)
        
        if not sentences:
            print("没有找到句子，无法进行处理")
            return "输入文本无法分割为句子"
        
        # 串行处理所有句子
        print(f"开始处理 {len(sentences)} 个句子...")
        kv_cache = better_sequential_processing(sentences, model, tokenizer)

        result = gen_with_kv(kv_cache, model, tokenizer, prompt, eos_token, max_tokens)
        return result

    except Exception as e:
        # 如果出现错误，记录错误
        print(f"处理出错: {e}")
        import traceback
        traceback.print_exc()
        return f"生成失败: {str(e)}"


# 示例使用
if __name__ == "__main__":
    # 检查MPS是否可用
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("使用MPS加速")
    else:
        device = torch.device("cpu")
        print("MPS不可用，使用CPU")

    model, tokenizer = model_manager.load_model(model_name, device)
    # 示例文本
    example_text = """<bos>system: 用户的名字叫kael。
    苹果公司是一家全球知名的科技公司。它成立于1976年。
    现在苹果公司主要生产iPhone、iPad和Mac电脑等产品。这些产品在全球范围内非常畅销。<eos>"""
    
    # txt_file = "/Users/kael/workbench/train_data/小王子/chapters.txt"
    # with open(txt_file, 'r', encoding='utf-8') as file:
    #     example_text = "<|bos|>system:\n"+file.read()+"<eos>"
    # 测试基于句子的串行推理
    print("\n使用MPS加速的KV缓存手动生成方法:")
    result = sentence_based_reasoning(example_text, model, tokenizer, 
                                        prompt="<start_of_turn>user:我叫什么名字？<end_of_turn><start_of_turn>assistant:", 
                                        eos_token="<end_of_turn>",
                                        max_tokens=100)
    print("\n最终结果:")
    print(result)