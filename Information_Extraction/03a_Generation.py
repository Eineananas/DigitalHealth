# \b = 单词字符(\w) 和 非单词字符(\W) 之间的边界位置
# 单词字符 \w：a-z A-Z 0-9 _ （字母、数字、下划线）
# 非单词字符 \W：其他所有字符（空格、标点、特殊符号等）

import pandas as pd
import re

def extract_tokens(text):
    excel_path = r"C:/Users/TangWei/Downloads/token_dict.xlsx"
    
    try:
        df = pd.read_excel(excel_path)
        if 'token' not in df.columns or 'explanation' not in df.columns:
            return ["Error: Missing 'token' or 'explanation' in Excel File"]
        
        token_dict = dict(zip(df['token'], df['explanation']))
    except FileNotFoundError:
        return ["Error: Fail to get Excel file"]
    except Exception as e:
        return [f"Error: Fail to read Excel File - {str(e)}"]
    
    result = []
    
    # 遍历每个token，使用正则表达式单词边界匹配
    for token, explanation in token_dict.items():
        token_str = str(token).strip()
        # \b 是单词边界，确保精确匹配完整单词
        pattern = r'\b' + re.escape(token_str) + r'\b'
        # Ignore Capital or Lower Case
        if re.search(pattern, text, re.IGNORECASE):
            result.append(f"{token} : {explanation}")
    
    return result

# Testing Example
test_text = """F/67
comes alone, NSND, retired
CA pancreas, Whipple 3 years ago
completed chemotherapy afterwards

well clinically, except mild steatorrhoea after operation 
no symptoms, no fever, no chills
tight diet control (avoid oily food)

CEA: 4.2, CBP, LFT, RFT, amylase:N 

"""
#Monica scalett ceasefire deceased

matches = extract_tokens(test_text)
for match in matches:
    print(match)
