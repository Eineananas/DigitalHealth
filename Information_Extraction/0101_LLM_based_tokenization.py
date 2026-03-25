from openai import OpenAI
import pandas as pd
import re
from collections import Counter
import os
from tqdm import tqdm
import time

class ClinicalTextProcessor:
    
    def __init__(self, api_base="http://11.165.116.1:18000/v1", model="/Qwen3-8B"):
        self.client = OpenAI(
            api_key="EMPTY",
            base_url=api_base,
        )
        self.model = model
        
    def get_response(self, query_text):
        """调用LLM获取处理结果"""
        try:
            chat_response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a text processing specialist."},
                    {"role": "user", "content": f"Now please process the given text. Input: {query_text}"},
                ],
                timeout=60
            )
            return chat_response.choices[0].message.content
        except Exception as e:
            print(f"API调用失败: {e}")
            return ""
    
    def extract_tokens(self, result_text):
        """精确提取LLM输出的token列表"""
        # 定位到第一个[和最后一个]
        start_idx = result_text.find('[')
        end_idx = result_text.rfind(']') 
        
        if start_idx == -1 or end_idx == -1 or start_idx >= end_idx:
            return []
        
        # 提取[]内内容并按逗号分割
        list_content = result_text[start_idx+1:end_idx]
        items = list_content.split(',')
        
        tokens = []
        for item in items:
            # 清理空格和引号
            cleaned = item.strip().strip("'").strip('"')
            if cleaned and len(cleaned) > 0:
                tokens.append(cleaned)
        
        return tokens
    
    def process_excel(self, excel_path, output_dir="./results"):
        os.makedirs(output_dir, exist_ok=True)
        
        print("Reading Excel...")
        df = pd.read_excel(excel_path)
        total_rows = len(df)
        print(f"Find {total_rows} notes...")
        
        results = []
        all_tokens = []
        
        # 带进度条的批量处理
        for idx, query_text in tqdm(enumerate(df["clinical_note"]), 
                                   total=total_rows, desc="in progress..."):
            
            result = self.get_response(query_text)
            tokens = self.extract_tokens(result)
            all_tokens.extend(tokens)
            
            results.append({
                'row_index': idx + 1,
                'query_text': str(query_text)[:200] + '...' if len(str(query_text)) > 200 else str(query_text),
                'full_result': result,
                'extracted_tokens': ', '.join(tokens) if tokens else '',
                'token_count': len(tokens)
            })
            
            # 避免API过载
            time.sleep(0.1)
        
        # 保存详细结果
        result_df = pd.DataFrame(results)
        result_path = os.path.join(output_dir, 'processing_results.xlsx')
        result_df.to_excel(result_path, index=False)
        
        # 生成词频统计
        word_freq = Counter(all_tokens)
        freq_df = pd.DataFrame(word_freq.most_common(1000), 
                              columns=['token', 'frequency'])
        freq_path = os.path.join(output_dir, 'word_frequency.xlsx')
        freq_df.to_excel(freq_path, index=False)
        
        # 生成摘要报告
        summary = {
            '总处理行数': total_rows,
            '总token数': len(all_tokens),
            '独特token数': len(word_freq),
            '平均每行token数': round(len(all_tokens)/total_rows, 2) if total_rows > 0 else 0
        }
        
        print("\n" + "="*60)
        print(f"  📄 处理行数: {summary['总处理行数']}")
        print(f"  🔤 总token数: {summary['总token数']:,}")
        print(f"  🆔 独特token: {summary['独特token数']:,}")
        print(f"  📈 平均token/行: {summary['平均每行token数']}")
        print("\n🔝 Top 20 高频医学词汇:")
        for token, count in word_freq.most_common(20):
            print(f"    {token:<15} {count:>4}")
        print("="*60)
        
        return result_df, freq_df, summary


if __name__ == "__main__":
    excel_path = "PATH"
    processor = ClinicalTextProcessor()
    result_df, freq_df, summary = processor.process_excel(excel_path)
