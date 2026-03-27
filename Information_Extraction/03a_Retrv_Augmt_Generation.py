# \b = 单词字符(\w) 和 非单词字符(\W) 之间的边界位置
# 单词字符 \w：a-z A-Z 0-9 _ （字母、数字、下划线）
# 非单词字符 \W：其他所有字符（空格、标点、特殊符号等）

from openai import OpenAI
import pandas as pd
import re
import json
import os
from tqdm import tqdm
import time


class ClinicalTextProcessor:
    """临床文本处理器 - 提取术语并调用LLM分析"""
    
    def __init__(self, 
                 token_dict_file=r"C:/Users/TangWei/Downloads/token_dict.xlsx", 
                 api_base="http://11.165.116.1:18000/v1", 
                 model="/Qwen3-8B"):
        """---------Initialization---------"""
        self.client = OpenAI(
            api_key="EMPTY",
            base_url=api_base,
        )
        self.model = model
        
        # 加载token字典
        try:
            df = pd.read_excel(token_dict_file)
            if 'token' not in df.columns or 'explanation' not in df.columns:
                raise ValueError("---------Cannot find 'token' or 'explanation' column in Excel File ---------")
            self.token_dict = dict(zip(df['token'], df['explanation']))
            print(f"---------✅ Successfully Loaded {len(self.token_dict)} tokens ------------------")
        except FileNotFoundError:
            print("❌ Error: Fail to find token_dict Excel")
            self.token_dict = {}
        except Exception as e:
            print(f"❌ Error: Fail to read Excel - {str(e)}")
            self.token_dict = {}

    def extract_tokens(self, query_text):
        """使用单词边界精确匹配提取临床术语"""
        result = []
        for token, explanation in self.token_dict.items():
            token_str = str(token).strip()
            pattern = r'\b' + re.escape(token_str) + r'\b'
            if re.search(pattern, query_text, re.IGNORECASE):
                result.append(f"{token} : {explanation}")
        return result

    def get_response(self, query_text):
        """调用LLM分析临床文本"""
        matches = self.extract_tokens(query_text) 
        len_match = len(matches)
        
        try:
            # 构建术语参考：匹配的token + 固定示例
            additional_terms = "F/50 refers to Gender: Female, Age: 50; M/35 refers to Gender: Male, Age: 35"
            
            if matches:
                terminology_ref = f"Here is a Detailed Explanation of the Jargons, Abbreviations, Terminologies that Appears in the Clinical Notes for your reference:\n {chr(10).join(matches)}\n{additional_terms}"
            else:
                terminology_ref = additional_terms
            
            chat_response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system", 
                        "content": """You are a medical text analyst. Extract structured information from clinical notes. 
                        Key extraction fields:
                        - patient_demographics (age, gender, occupation, etc.)
                        - diagnosis  
                        - procedures
                        - medications
                        - lab_results
                        - symptoms
                        - clinical_status
                        Output Format:
                        [<Variable_1>: Value | <Variable_2>: Value | <Variable_3>: Value | ... <Variable_n>: Value]
                        Example1:
                        Input: F/[61to70] <CR><LF>lives with husband<CR><LF>younger brother has newly diagnosed with metastatic malignancy <CR><LF><CR><LF>HT, HBsAg and HCV-ve <CR><LF><CR><LF> pT3N0 ca pancreatic head with Whipple operation done on 8/2017 <CR><LF>pre-op CEA 2.6<CR><LF> adj Gemzar x 6 completed in 4/2018<CR><LF> USG in [DATE]: no evidence of tumor recurrence <CR><LF> CT scan in 4USG/2019 <CR><LF><CR><LF>Private PET-CT [DATE]: <CR><LF> -wound nodule no uptake<CR><LF> -mesenteric LNs 0.9cm SUVmax 1.9 <CR><LF> -no recurrence <CR><LF><CR><LF>============<CR><LF> FU today <CR><LF><CR><LF>No cervical LNs palpable <CR><LF> Abdomen soft, non-tender <CR><LF><CR><LF> CBC normal <CR><LF> LRFT, bone profile normal <CR><LF>CEA 3.2 <CR><LF><CR><LF> CT DD/4/19
                        Output: <Gender>: F | <Age>: 61to70 | ...
                        Example2:
                        Input: Referred by GP for mx of radiological ca pancreas <CR><LF> <CR><LF> Hx of NIDDM on meds from GP <CR><LF>p/w epigastric pain <CR><LF> OGD found gastritis <CR><LF> CT found pancreatic mass in uncinate mass with liver, bone and L rib <CR><LF><CR><LF> PET-CT 30/4: <CR><LF> hypermet uncinate process lesion ~2.5cm<CR><LF>peripancreatic nodule ?regional LN<CR><LF> hypermet nodules in liver, likely mets <CR><LF> bones mets at 6th rib and R ischium <CR><LF><CR><LF> came alone <CR><LF> wt loss and epigas pain <CR><LF> 1-2tabs of Panadol prn <CR><LF> L rib pain <CR><LF> LOW>10lbs <CR><LF> not icteric <CR><LF><CR><LF> explained PETCT report of radiological metastatic ca pancreas <CR><LF>need for histological confirmation of malignancy (told to be extremely difficult by GP) <CR><LF><CR><LF>understand poor prognosis if pancreatic malignancy confirmed <CR><LF> wanted to at least see her dau graduate from Uni in July
                        Output: <Gender>: F | <Age>: 61to70 | ...
                          """
                    },
                    {
                        "role": "user", 
                        "content": f"""Clinical Note: \n{query_text} \n{terminology_ref}"""
                    }
                ],
                timeout=60
            )
            return chat_response.choices[0].message.content, len_match
        except Exception as e:
            print(f"---------❌ Cannot reach the LLM: {e}------------------")
            return f"Error: {str(e)}"

    def process_excel(self, excel_path, output_dir="./results"):
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        print("---------📖 Reading Excel File... ------------------")
        df = pd.read_excel(excel_path)
        
        if "clinical_note" not in df.columns:
            raise ValueError("---------Cannot find column 'clinical_note' in Exile File---------")
            
        total_rows = len(df)
        print(f"---------📊 Find {total_rows} Clinical Notes in total!!!------------------")
        
        results = []
        for idx, query_text in tqdm(enumerate(df["clinical_note"]), 
                                    total=total_rows, 
                                    desc="处理进度"):
            # 处理空文本
            if pd.isna(query_text) or str(query_text).strip() == "":
                result = "Empty note"
                len_match = 0
            else:
                result, len_match = self.get_response(str(query_text))
            
            results.append({
                'row_index': idx + 1,
                'original_text': str(query_text),
                'processed_result': result,
                'matched_tokens': len_match
            })
            
            # API限流
            time.sleep(0.2)
        
        result_df = pd.DataFrame(results)
        
        # 1. 保存CSV
        #csv_path = os.path.join(output_dir, "clinical_analysis_results.csv")
        #result_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        #print(f"💾 CSV结果已保存: {csv_path}")
        
        # 2. 保存详细JSON
        json_path = os.path.join(output_dir, "clinical_analysis_detailed.json")
        detailed_results = []
        for _, row in result_df.iterrows():
            detailed_results.append({
                'row_index': row['row_index'],
                'original_text': row['original_text'],
                'llm_analysis': row['processed_result'],
                'matched_token_count': row['matched_tokens']
            })
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, ensure_ascii=False, indent=2)
        print(f"💾 JSON结果已保存: {json_path}")
        
        return result_df


if __name__ == "__main__":
    excel_path = r"PATH.xlsx"  
    processor = ClinicalTextProcessor()
    result_df = processor.process_excel(excel_path)  
    print("🎉 ")
