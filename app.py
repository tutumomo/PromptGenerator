import streamlit as st
import requests
from typing import List
from dotenv import load_dotenv
import json
from datetime import datetime
import pathlib
import os

load_dotenv()  # 載入 .env 文件中的環境變數

# 修改 system prompt 常量
PROMPT_IMPROVEMENT_TEMPLATE = '''
# 角色 
你是一位AI prompt 專家，熟悉各種Prompt Optimizer框架(APE、CARE、CHAT、COAST、CREAT、CRISPE、RASCEF、RTF為主)，可以將使用者輸入的提示詞選定合適的Prompt框架後，編撰和優化AI prompts。
## 重要:
- 無論提問使用何種語言，一律以繁體中文進行回答(禁用簡體中文，且須符合台灣用語習慣)。 

請分析並改進以下提示詞:
{original_prompt}

只返回優化後的提示詞，不要有其他說明文字、標題或解釋(返回結果不要有"# 優化後提示詞："的字眼).
'''

# Ollama API 調用函數
def call_ollama(model_name, prompt, temperature, top_p, top_k, repeat_penalty, max_tokens):
    """
    調用本地 Ollama API 來改進提示詞
    """
    try:
        prompt_template = PROMPT_IMPROVEMENT_TEMPLATE.format(original_prompt=prompt)
        
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                "model": model_name,
                "prompt": prompt_template,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "repeat_penalty": repeat_penalty,
                "num_predict": max_tokens,  # Ollama 使用 num_predict 作為 max_tokens
                "stream": False
            }
        )
        return response.json().get('response', '生成失敗：無回應內容')
            
    except Exception as e:
        return f"調用 Ollama 失敗: {str(e)}"

# 新增函數來獲取 Ollama 模型列表
def get_ollama_models() -> List[str]:
    """
    獲取本地 Ollama 已安裝的模型列表
    """
    try:
        response = requests.get('http://localhost:11434/api/tags')
        if response.status_code == 200:
            models = [model['name'] for model in response.json()['models']]
            return sorted(models) if models else ["aya-expanse"]
        return ["aya-expanse"]
    except Exception as e:
        st.error(f"獲取 Ollama 模型列表失敗: {str(e)}")
        return ["aya-expanse"]  # 返回默認模型

# 儲存相關函數
def save_prompt_history(original_prompt: str, improved_prompt: str, generated_content: str = None) -> bool:
    """
    儲存提示詞歷史記錄
    """
    try:
        save_dir = pathlib.Path("prompts_history")
        save_dir.mkdir(exist_ok=True)
        
        save_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "original_prompt": original_prompt,
            "improved_prompt": improved_prompt,
            "generated_content": generated_content
        }
        
        filename = f"prompt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        file_path = save_dir / filename
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
            
        return True
    except Exception as e:
        st.error(f"儲存失敗: {str(e)}")
        return False

# 執行提示詞函數
def execute_prompt(model: str, prompt: str, temperature: float, top_p: float, top_k: int, repeat_penalty: float, max_tokens: int) -> str:
    """
    執行提示詞並回傳生成的內容
    """
    try:
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                "model": model,
                "prompt": prompt,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "repeat_penalty": repeat_penalty,
                "num_predict": max_tokens,  # Ollama 使用 num_predict 作為 max_tokens
                "stream": False
            }
        )
        return response.json().get('response', '生成失敗：無回應內容')
    except Exception as e:
        return f"生成失敗: {str(e)}"

def main():
    st.title("AI 提示詞優化器")
    
    with st.sidebar:
        st.header("設定")
        
        # 模型選擇
        model_options = get_ollama_models()
        selected_model = st.selectbox(
            "選擇本地模型:", 
            model_options, 
            index=model_options.index("aya-expanse") if "aya-expanse" in model_options else 0
        )

        # 參數設定
        st.header("參數設定")
        with st.expander("進階參數設定", expanded=True):
            temperature = st.slider(
                "Temperature (溫度):", 
                min_value=0.0, 
                max_value=2.0, 
                value=0.7, 
                step=0.1,
                help="控制生成文本的創意程度。較高的值會產生更多樣化的輸出，較低的值會產生更保守的輸出。"
            )
            
            max_tokens = st.slider(
                "Max Tokens (最大生成長度):", 
                min_value=50, 
                max_value=5000, 
                value=2000, 
                step=50,
                help="控制生成文本的最大長度。較高的值允許生成更長的回應。"
            )
            
            top_p = st.slider(
                "Top P (機率閾值):", 
                min_value=0.0, 
                max_value=1.0, 
                value=0.9, 
                step=0.05,
                help="控制生成文本的多樣性。較低的值會使輸出更加集中和確定。"
            )
            
            top_k = st.slider(
                "Top K (候選數量):", 
                min_value=1, 
                max_value=100, 
                value=40, 
                step=1,
                help="限制每次選擇的候選詞數量。較低的值會使輸出更加保守和可預測。"
            )
            
            repeat_penalty = st.slider(
                "Repeat Penalty (重複懲罰):", 
                min_value=1.0, 
                max_value=2.0, 
                value=1.1, 
                step=0.1,
                help="控制文本重複的程度。較高的值會降低重複內容的出現機率。"
            )

    # 主要內容區域
    st.header("輸入原始提示詞")
    original_prompt = st.text_area(
        "請輸入你想要優化的提示詞:",
        help="輸入你的原始提示詞，AI 將幫助你改進使其更加有效。",
        height=150
    )

    # 存儲優化後的提示詞的 session state
    if 'improved_prompt' not in st.session_state:
        st.session_state.improved_prompt = ""

    if st.button("優化提示詞"):
        if not original_prompt:
            st.error("請輸入原始提示。")
            return

        try:
            with st.spinner('優化中...'):
                response = call_ollama(
                    selected_model, 
                    original_prompt, 
                    temperature,
                    top_p,
                    top_k,
                    repeat_penalty,
                    max_tokens
                )
                st.session_state.improved_prompt = response
                
                # 儲存優化結果
                if save_prompt_history(original_prompt, response):
                    st.success("✅ 已儲存優化結果")
                
        except Exception as e:
            st.error(f"優化過程中發生錯誤: {str(e)}")
            return

    # 顯示優化後的提示詞和生成按鈕
    if st.session_state.improved_prompt:
        st.subheader("優化後的提示詞")
        improved_prompt = st.text_area(
            "優化結果:",
            value=st.session_state.improved_prompt,
            height=150
        )
        
        # 添加可選的用戶輸入框
        user_input = st.text_input(
            "請輸入你的需求: (可選)",
            help="如果需要，可以在此輸入具體需求。例如：想要一道香辣夠味的宮保雞丁食譜"
        )
        
        # 生成內容按鈕
        if st.button("執行提示詞"):
            with st.spinner('生成中...'):
                final_prompt = (
                    f"{improved_prompt}\n\n用戶輸入：{user_input}"
                    if user_input
                    else improved_prompt
                )
                
                generated_content = execute_prompt(
                    selected_model, 
                    final_prompt, 
                    temperature,
                    top_p,
                    top_k,
                    repeat_penalty,
                    max_tokens
                )
                
                # 儲存生成結果
                if save_prompt_history(original_prompt, improved_prompt, generated_content):
                    st.success("✅ 已儲存生成結果")
                
                st.subheader("生成的內容")
                st.markdown(generated_content)
                with st.expander("顯示原始文本（方便複製）"):
                    st.text_area("原始文本:", value=generated_content, height=300)

if __name__ == "__main__":
    main()
