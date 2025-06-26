import torch
from copy import deepcopy
from kv_retrieve import (
    process_chunks_batch,
    retrieve_and_score_chunks_batch
)
from golemrytool.rigging_art import split_sentences_by_punctuation
from golemrytool.mortise import TokenGenerator

class RagKvPipeline:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.processed_chunks = []
        self.generator = TokenGenerator(model, tokenizer)

    def index_text(self, text: str=None, text_list: list=None, chunk_size=256):
        """
        索引文本（仅支持批处理模式）
        
        Args:
            text: 要索引的文本
            chunk_size: 块大小
        """
        if text:
            chunks = split_sentences_by_punctuation(text)
        elif text_list:
            chunks = text_list
        
        self.processed_chunks = process_chunks_batch(chunks, self.model, self.tokenizer)
        print(f"文本已索引为 {len(self.processed_chunks['chunks_metadata'])} 个块（批处理模式）。")

    def query(self, query, top_k=3, max_tokens=100, use_batch_generation=True, scoring_method="average"):
        """
        查询相关内容并生成答案
        
        Args:
            query: 查询文本
            top_k: 返回前K个相关块
            max_tokens: 最大生成token数
            use_batch_generation: 是否使用批处理生成
            scoring_method: 评分方法 - 'average', 'top_tokens'
        """
        if not self.processed_chunks:
            return "错误：请先使用 .index_text() 方法索引文本。"
        
        # 使用批处理检索
        top_k_chunks, updated_full_kv_cache, top_k_scores, top_k_attention_masks, query_ids, top_k_indices = retrieve_and_score_chunks_batch(
            query, deepcopy(self.processed_chunks), self.model, self.tokenizer, top_k=top_k, scoring_method=scoring_method
        )
        
        if not top_k_chunks:
            return "未能检索到相关上下文。"

        # 提取Top-K缓存为独立缓存列表
        from kv_retrieve_batch import extract_top_k_caches_as_list
        top_k_updated_caches = extract_top_k_caches_as_list(updated_full_kv_cache, top_k_indices)
        
        # 生成候选答案
        candidate_answers = []
        print(f"\n正在{'批量' if use_batch_generation else '串行'}生成候选答案...")
        
        if use_batch_generation and top_k_attention_masks is not None:
            # 批量生成模式
            try:
                # 构建批处理的 past_key_values
                num_layers = len(top_k_updated_caches[0])
                batch_past_key_values = []
                for layer_idx in range(num_layers):
                    keys = torch.cat([cache[layer_idx][0] for cache in top_k_updated_caches], dim=0)
                    values = torch.cat([cache[layer_idx][1] for cache in top_k_updated_caches], dim=0)
                    batch_past_key_values.append((keys, values))
                final_past_key_values = tuple(batch_past_key_values)
                
                with torch.no_grad():
                    generated_outputs = self.model.generate(
                        past_key_values=final_past_key_values,
                        attention_mask=top_k_attention_masks,
                        max_new_tokens=max_tokens,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 0,
                        use_cache=True
                    )
                
                # 处理生成结果
                for i, generated_ids in enumerate(generated_outputs):
                    answer_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                    candidate_answers.append(answer_text)
                    
                    chunk_text = (top_k_chunks[i] if isinstance(top_k_chunks[i], str) else top_k_chunks[i]['text']).replace('\n', ' ').strip()
                    print(f"  - 候选答案 {i+1} (来自块: \"{chunk_text[:50]}...\", 分数: {top_k_scores[i]:.2f}): {answer_text.strip()}")
                    
            except Exception as e:
                print(f"批量生成失败，回退到串行生成: {e}")
                use_batch_generation = False
        
        if not use_batch_generation:
            # 串行生成模式
            for i, cache in enumerate(top_k_updated_caches):
                # 准备attention_mask
                attention_mask = None
                if top_k_attention_masks is not None:
                    attention_mask = top_k_attention_masks[i].unsqueeze(0)
                
                generated_tokens, _ = self.generator.generate(
                    initial_cache=deepcopy(cache),
                    max_tokens=max_tokens,
                    attention_mask=attention_mask,
                    temperature=0.7,
                    top_p=0.9,
                    stop_tokens=[self.tokenizer.encode("<|im_end|>", add_special_tokens=False)[-1]]
                )
                
                decoded_answer = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                candidate_answers.append(decoded_answer)
                
                chunk_text = (top_k_chunks[i] if isinstance(top_k_chunks[i], str) else top_k_chunks[i]['text']).replace('\n', ' ').strip()
                print(f"  - 候选答案 {i+1} (来自块: \"{chunk_text[:50]}...\", 分数: {top_k_scores[i]:.2f}): {decoded_answer.strip()}")

        # 选择最佳答案
        if candidate_answers:
            best_answer = candidate_answers[0]
            return best_answer
        else:
            return "生成失败：未能生成有效的候选答案。"
        
if __name__ == '__main__':
    # 测试代码
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    print(f"正在从 {model_name} 加载模型...")
    from golemrytool.model_manager_v2 import model_manager
    model, tokenizer = model_manager.load_model(model_name, )
    
    long_text = """用户的名字叫kael。用户是猫咪公司的CEO。
    苹果公司是一家全球知名的科技公司。它成立于1976年。
    现在苹果公司主要生产iPhone、iPad和Mac电脑等产品。这些产品在全球范围内非常畅销。
    人工智能（AI）正在迅速改变世界。它是一门研究、开发用于模拟、延伸和扩展人的智能的理论、方法、技术及应用系统的新技术科学。
    AI的主要领域包括机器学习、自然语言处理、计算机视觉和专家系统。机器学习是AI的核心，它使计算机能够从数据中学习而无需显式编程。
    自然语言处理（NLP）则让计算机能够理解和生成人类语言。这在聊天机器人、机器翻译和情感分析等应用中至关重要。
    计算机视觉涉及解释和理解来自世界的视觉信息，例如图像和视频。应用范围从自动驾驶汽车到医学图像分析。
    尽管AI潜力巨大，但它也带来了挑战，包括数据隐私、算法偏见和对就业的影响。负责任地开发和部署AI是当前的重要议题。"""

    file = "/Users/kael/workbench/train_data/小王子/para_result.txt"
    with open(file, 'r') as f:
        long_text = f.readlines()

    prompt="<|im_start|>user:我应该怎么办？<|im_end|>"
    prompt_2="<|im_start|>user:这个世界会好么？<|im_end|>"

    pipeline = RagKvPipeline(model, tokenizer)
    
    # 测试批处理模式
    print("\n=== 测试批处理索引 + 批量生成 ===")
    pipeline.index_text(text_list=long_text)
    
    user_query = prompt
    print(f"\n用户问题: {user_query}")
    answer = pipeline.query(user_query, top_k=3, use_batch_generation=True, scoring_method="average")
    print(f"\n模型最终回答: {answer}")

    user_query = prompt_2
    print(f"\n用户问题: {user_query}")
    answer = pipeline.query(user_query, top_k=3, use_batch_generation=True, scoring_method="average")
    print(f"\n模型最终回答: {answer}")
    
    # 测试串行生成
    print("\n=== 测试批处理索引 + 串行生成 ===")
    user_query = prompt
    print(f"\n用户问题: {user_query}")
    answer = pipeline.query(user_query, top_k=3, use_batch_generation=False, scoring_method="average")
    print(f"\n模型最终回答: {answer}")