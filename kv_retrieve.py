import torch
from transformers import DynamicCache

def process_chunks_batch(chunks, model, tokenizer, padding_side='left'):
    """
    【批处理版本】将所有文本块进行批处理，生成一个单一的、经过填充的批处理KV缓存。
    不再将缓存拆分为独立的块，而是保持其批处理形式以便后续高效使用。
    
    Args:
        chunks: 文本块列表
        model: 模型
        tokenizer: 分词器
        padding_side: 填充方向，'left' 或 'right'，默认为 'left'
    """
    chunk_texts = chunks
    if not chunk_texts:
        return {
            "kv_cache": None,
            "attention_mask": None,
            "chunks_metadata": [],
            "padding_side": padding_side
        }

    # 临时设置分词器的填充方向
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = padding_side
    
    try:
        # 使用分词器的批处理功能，自动处理填充和注意力掩码
        inputs = tokenizer.batch_encode_plus(
            chunk_texts,
            return_tensors="pt",
            padding=True,  # 填充到批次中的最大长度
            truncation=True,
            max_length=model.config.max_position_embeddings
        ).to(model.device)

        # 一次模型前向传播，生成整个批次的KV缓存
        with torch.no_grad():
            outputs = model(**inputs, use_cache=True)
    finally:
        # 恢复原始的填充方向
        tokenizer.padding_side = original_padding_side

    # 返回一个包含所有信息的字典
    return {
        "kv_cache": outputs.past_key_values,
        "attention_mask": inputs.attention_mask,
        "chunks_metadata": chunks,  # 保留原始块信息
        "padding_side": padding_side
    }

def calculate_score_max(attention_slice):
    """
    使用最大值方法计算分数
    """
    return attention_slice.max()

def calculate_score_average(attention_slice, chunk_attention_mask=None):
    """
    计算注意力权重的平均分数（批量处理版本）
    """
    if chunk_attention_mask is not None:
        # 批量处理：[batch_size, heads, query_len, chunk_len]
        scores_per_chunk = attention_slice.sum(dim=[1, 2, 3])
        chunk_lengths = chunk_attention_mask.sum(dim=1)
        chunk_lengths = torch.max(chunk_lengths, torch.tensor(1.0, device=chunk_lengths.device))
        return scores_per_chunk / chunk_lengths
    else:
        # 如果没有提供mask，使用原始方法
        return attention_slice.mean()

def calculate_score_top_tokens(attention_slice, top_k=2, chunk_attention_mask=None):
    """
    计算注意力权重中top-k个token的平均分数（批量处理版本）
    """
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    elif top_k == 1:
        return calculate_score_max(attention_slice)
    
    # 批量处理逻辑
    token_scores = attention_slice.sum(dim=[1, 2])  # 对heads和query_len维度求和
    if chunk_attention_mask is not None:
        token_scores = token_scores * chunk_attention_mask  # 使用chunk_attention_mask屏蔽填充token
    
    batch_size, chunk_len = token_scores.shape
    normalized_scores = torch.zeros(batch_size, device=token_scores.device)
    
    for batch_idx in range(batch_size):
        if chunk_attention_mask is not None:
            valid_mask = chunk_attention_mask[batch_idx] > 0
            valid_scores = token_scores[batch_idx][valid_mask]
        else:
            valid_scores = token_scores[batch_idx]
        
        if len(valid_scores) > 0:
            actual_k = min(top_k, len(valid_scores))
            top_scores, _ = torch.topk(valid_scores, actual_k)
            normalized_scores[batch_idx] = top_scores.mean()
        else:
            normalized_scores[batch_idx] = 0.0
    
    return normalized_scores

def calculate_attention_scores_batch(last_layer_attentions, chunk_attention_mask, scoring_method="average", top_num=2):
    """
    从注意力图中计算批处理的分数
    
    Args:
        last_layer_attentions: 最后一层的注意力权重
        chunk_attention_mask: 文本块的注意力掩码
        scoring_method: "average", "top_tokens"
        top_num: top_tokens方法中的top-k参数
    
    Returns:
        normalized_scores: 归一化后的分数
    """
    # 提取query对chunk的注意力
    attention_to_chunks = last_layer_attentions[:, :, -1:, :]  # 取最后一个token（query）对所有token的注意力
    
    if scoring_method == "top_tokens":
        return calculate_score_top_tokens(attention_to_chunks, top_k=top_num, chunk_attention_mask=chunk_attention_mask)
    elif scoring_method == "average":
        return calculate_score_average(attention_to_chunks, chunk_attention_mask=chunk_attention_mask)

def select_top_k_chunks(scores, chunks, top_k):
    """
    根据分数选择Top-K个文本块
    
    Args:
        scores: 分数张量
        chunks: 原始文本块列表
        top_k: 选择的数量
    
    Returns:
        top_k_chunks: Top-K文本块
        top_k_scores: Top-K分数
        top_k_indices: Top-K索引
    """
    top_k_scores, top_k_indices = torch.topk(scores, k=min(top_k, len(chunks)))
    top_k_chunks = [chunks[i] for i in top_k_indices]
    
    print(f"检索到的Top-{len(top_k_chunks)}个块的分数: {top_k_scores.tolist()}")
    return top_k_chunks, top_k_scores, top_k_indices

def retrieve_and_score_chunks_batch(query, processed_data, model, tokenizer, top_k=3, scoring_method="average"):
    """
    【批处理检索版本】利用预先计算并填充好的批处理KV缓存，高效地为所有文本块评分。
    
    Args:
        scoring_method: "average" 或 "top_tokens"，决定评分方式
    """
    batched_kv_cache = processed_data["kv_cache"]
    chunk_attention_mask = processed_data["attention_mask"]
    original_chunks = processed_data["chunks_metadata"]
    padding_side = processed_data.get("padding_side", "left")

    if not original_chunks:
        return [], [], [], [], None

    # 临时设置分词器的填充方向以保持一致性
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = padding_side
    
    try:
        # 1. 准备 query 输入
        query_ids = tokenizer(query, return_tensors="pt").input_ids.to(model.device)
        num_chunks = batched_kv_cache[0][0].shape[0]
        
        # 将 query 复制批次大小次，以便与每个块的KV缓存进行交互
        expanded_query_ids = query_ids.expand(num_chunks, -1)

        # 2. 准备组合的 attention_mask
        query_attention_mask = torch.ones_like(expanded_query_ids)
        combined_attention_mask = torch.cat([chunk_attention_mask, query_attention_mask], dim=1)
        
        # 3. 模型前向传播
        with torch.no_grad():
            outputs = model(
                input_ids=expanded_query_ids,
                past_key_values=batched_kv_cache,
                attention_mask=combined_attention_mask,
                output_attentions=True,
                use_cache=True
            )
    finally:
        # 恢复原始的填充方向
        tokenizer.padding_side = original_padding_side
    
    # 4. 使用辅助函数计算分数
    last_layer_attentions = outputs.attentions[-1]
    normalized_scores = calculate_attention_scores_batch(
        last_layer_attentions, chunk_attention_mask, scoring_method
    )

    # 5. 使用辅助函数选择Top-K
    top_k_chunks, top_k_scores, top_k_indices = select_top_k_chunks(
        normalized_scores, original_chunks, top_k
    )
    print(f"Top-K 块的索引: {top_k_indices.tolist()}")
    
    # 6. 提取 Top-K 对应的、更新后的KV缓存和注意力掩码
    updated_full_kv_cache = outputs.past_key_values
    top_k_attention_masks = combined_attention_mask[top_k_indices]

    return top_k_chunks, updated_full_kv_cache, top_k_scores, top_k_attention_masks, query_ids, top_k_indices

def extract_top_k_caches_as_batch(updated_full_kv_cache, top_k_indices):
    """
    从完整的KV缓存中提取Top-K对应的缓存，返回批处理格式的DynamicCache
    """
    top_k_updated_caches = DynamicCache()
    # 逐层筛选
    for layer_idx in range(len(updated_full_kv_cache)):
        key_states, value_states = updated_full_kv_cache[layer_idx]
        # 按 batch 维度切片
        filtered_key = key_states[top_k_indices]
        filtered_value = value_states[top_k_indices]
        # 存入新缓存
        top_k_updated_caches.update(filtered_key, filtered_value, layer_idx=layer_idx)
    
    return top_k_updated_caches

def extract_top_k_caches_as_list(updated_full_kv_cache, top_k_indices):
    """
    从完整的KV缓存中提取Top-K对应的缓存，返回独立DynamicCache的列表
    """
    top_k_updated_caches = []
    for i in top_k_indices:
        individual_cache = DynamicCache()
        for layer_idx in range(len(updated_full_kv_cache)):
            key_tensor = updated_full_kv_cache[layer_idx][0][i:i+1]
            value_tensor = updated_full_kv_cache[layer_idx][1][i:i+1]
            individual_cache.update(key_tensor, value_tensor, layer_idx=layer_idx)
        top_k_updated_caches.append(individual_cache)
    
    return top_k_updated_caches

def stack_kv_caches(kv_caches):
    """
    将一个KV缓存对象（DynamicCache）的列表堆叠成一个批处理的KV缓存对象。
    """
    if not kv_caches:
        return None

    num_layers = len(kv_caches[0])
    
    max_seq_len = 0
    for cache in kv_caches:
        if cache.get_seq_length() > max_seq_len:
            max_seq_len = cache.get_seq_length()

    new_cache = DynamicCache()
    for layer_idx in range(num_layers):
        padded_keys = []
        padded_values = []
        for cache in kv_caches:
            key, value = cache[layer_idx]
            seq_len = key.shape[2]
            padding_needed = max_seq_len - seq_len
            
            if padding_needed > 0:
                padding_tuple = (0, 0, 0, padding_needed)
                padded_key = torch.nn.functional.pad(key, padding_tuple, "constant", 0)
                padded_value = torch.nn.functional.pad(value, padding_tuple, "constant", 0)
            else:
                padded_key = key
                padded_value = value
            
            padded_keys.append(padded_key)
            padded_values.append(padded_value)

        keys = torch.cat(padded_keys, dim=0)
        values = torch.cat(padded_values, dim=0)
        new_cache.update(keys, values, layer_idx)
        
    return new_cache