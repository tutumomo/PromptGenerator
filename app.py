import streamlit as st
from openai import OpenAI
import os
import sys
import subprocess
import google.generativeai as genai
import requests
from typing import List
from google.api_core import retry
import json
from dotenv import load_dotenv

load_dotenv()  # 載入 .env 文件中的環境變數

# 修改 system prompt 常量
PROMPT_IMPROVEMENT_TEMPLATE = '''
# 角色 
你是一位AI prompt 專家，熟悉各種Prompt Optimizer框架(APE、CARE、CHAT、COAST、CREAT、CRISPE、RASCEF、RTF為主)，可以將使用者輸入的提示詞選定合適的Prompt框架後，編撰和優化AI prompts。
## 重要:
- 無論提問使用何種語言，一律以繁體中文進行回答(禁用簡體中文，且須符合台灣用語習慣)。 

請分析並改進以下提示詞:
{original_prompt}

請直接返回優化後的提示詞，不要包含任何其他說明文字、標題或解釋(返回結果不要有"# 優化後提示詞："的字眼).
'''

# OpenAI API 調用函數
def call_openai(api_key, model, prompt, temperature, top_p, max_tokens, presence_penalty, frequency_penalty):
    """
    調用 OpenAI API 來改進提示詞
    """
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "你是一位專業的提示詞工程師，專門幫助改進和優化提示詞。"},
            {"role": "user", "content": PROMPT_IMPROVEMENT_TEMPLATE.format(original_prompt=prompt)}
        ],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty
    )
    return response.choices[0].message.content

# Gemini API 調用函數
def call_gemini(api_key: str, model_name: str, prompt: str, temperature: float, top_p: float, max_tokens: int) -> str:
    """
    調用 Gemini API 來改進提示詞
    """
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        
        prompt_template = PROMPT_IMPROVEMENT_TEMPLATE.format(original_prompt=prompt)
        
        response = model.generate_content(
            prompt_template,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                top_p=top_p,
                max_output_tokens=max_tokens,
                candidate_count=1
            )
        )
        
        return response.text
    except Exception as e:
        return f"Gemini API 錯誤: {str(e)}"

# xAI API 調用函數
def call_xai(api_key, prompt, temperature, top_p, max_tokens):
    """
    調用 xAI API
    注意：目前 xAI/Grok API 尚未公開
    """
    # 返回提示信息
    return "xAI (Grok) API 目前尚未公開，但已知有 grok-beta 及 grok-vision-beta 兩個模型。請等待官方發布 API 存取方式。"

# Ollama API 調用函數
def call_ollama(model_name, prompt, temperature):
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
                "stream": False
            },
            stream=False
        )
        
        if response.status_code == 200:
            try:
                return response.json().get('response', '生成失敗：無回應內容')
            except json.JSONDecodeError as e:
                full_response = []
                for line in response.text.strip().split('\n'):
                    if line:
                        try:
                            data = json.loads(line)
                            if 'response' in data:
                                full_response.append(data['response'])
                        except json.JSONDecodeError:
                            continue
                return ''.join(full_response) if full_response else '生成失敗：回應格式錯誤'
        else:
            return f'生成失敗：HTTP 錯誤 {response.status_code}'
            
    except Exception as e:
        return f"調用 Ollama 失敗: {str(e)}"

# 新增函數來獲取 OpenAI 模型列表
def get_openai_models(api_key: str) -> List[str]:
    """
    獲取 OpenAI 可用的模型列表
    """
    try:
        client = OpenAI(api_key=api_key)
        models = client.models.list()
        # 篩選出所有可用於聊天的模型
        chat_models = []
        for model in models:
            if ('gpt' in model.id.lower() and 
                not any(x in model.id.lower() for x in ['instruct', 'similarity', 'edit', 'audio'])):
                chat_models.append(model.id)
        return sorted(chat_models) if chat_models else ["gpt-3.5-turbo", "gpt-4"]
    except Exception as e:
        st.error(f"獲取 OpenAI 模型列表失敗: {str(e)}")
        return ["gpt-3.5-turbo", "gpt-4"]

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

# 新增 Gemini 模型列表獲取函數
def get_gemini_models(api_key: str) -> List[str]:
    """
    獲取 Gemini 可用的模型列表
    """
    try:
        genai.configure(api_key=api_key)
        # 目前已知的 Gemini 模型
        default_models = ["gemini-pro", "gemini-pro-vision"]
        
        # 嘗試獲取模型列表
        try:
            models = genai.list_models()
            gemini_models = []
            for model in models:
                if 'gemini' in model.name.lower():
                    gemini_models.append(model.name)
            return sorted(gemini_models) if gemini_models else default_models
        except:
            return default_models
            
    except Exception as e:
        st.error(f"獲取 Gemini 模型列表失敗: {str(e)}")
        return ["gemini-pro"]  # 返回最基本的模型

# 新增 xAI 模型列表獲取函數
def get_xai_models(api_key: str) -> List[str]:
    """
    獲取 xAI 可用的模型列表
    注意：目前 xAI/Grok API 尚未公開，返回已知模型列表
    """
    # 目前已知的 Grok 模型
    default_models = ["grok-beta", "grok-vision-beta"]
    
    try:
        # 暫時不進行 API 調用，直接返回已知模型
        return default_models
    except Exception as e:
        st.error(f"獲取 xAI 模型列表失敗: {str(e)}")
        return default_models

# 新增函數來檢查 API 金鑰
def get_api_key(api_type: str) -> tuple[str, str]:
    """
    檢查並獲取 API 金鑰
    返回: (api_key, message)
    """
    env_var_map = {
        "OpenAI": "OPENAI_API_KEY",
        "Gemini": "GEMINI_API_KEY",
        "xAI": "XAI_API_KEY"
    }
    
    env_var = env_var_map.get(api_type)
    if not env_var:
        return None, ""
        
    api_key = os.getenv(env_var)
    if api_key:
        return api_key, f"✅ 已從環境變數 {env_var} 讀取 API 金鑰"
    return None, f"💡 可以設置環境變數 {env_var} 來儲存 API 金鑰"

# 新增執行提示詞的函數
def execute_prompt(api_type: str, api_key: str, model: str, prompt: str, 
                  temperature: float, top_p: float, max_tokens: int,
                  presence_penalty: float = 0, frequency_penalty: float = 0) -> str:
    """
    執行提示詞並回生成的內容
    """
    try:
        if api_type == "OpenAI":
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty
            )
            return response.choices[0].message.content
        
        elif api_type == "Gemini":
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(model)
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    top_p=top_p,
                    max_output_tokens=max_tokens
                )
            )
            return response.text
        
        elif api_type == "Ollama":
            response = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    "model": model,
                    "prompt": prompt,
                    "temperature": temperature,
                    "stream": False
                }
            )
            return response.json().get('response', '生成失敗：無回應內容')
        
        else:  # xAI
            return "xAI (Grok) API 目前尚未公開"
            
    except Exception as e:
        return f"生成失敗: {str(e)}"

def main():
    st.title("AI 提示詞優化器")
    
    with st.sidebar:
        st.header("設定")
        api_options = ["Ollama", "OpenAI", "Gemini", "xAI"]  # 修改順序，將 Ollama 放在第一位
        selected_api = st.selectbox("選擇 API:", api_options, index=0)  # index=0 會選擇 Ollama 作為預設

        # API 金鑰處理
        if selected_api != "Ollama":
            api_key, env_message = get_api_key(selected_api)
            if env_message:
                st.info(env_message)
            
            # 如果環境變數中沒有 API 金鑰，則顯示輸入框
            if not api_key:
                api_key = st.text_input(f"輸入 {selected_api} API 金鑰:", type="password")
        else:
            api_key = None

        # 模型選擇
        if selected_api == "OpenAI":
            if api_key:
                model_options = get_openai_models(api_key)
            else:
                model_options = ["gpt-3.5-turbo", "gpt-4"]
            selected_model = st.selectbox("選擇模型:", model_options)
        elif selected_api == "Gemini":
            if api_key:
                model_options = get_gemini_models(api_key)
            else:
                model_options = ["gemini-pro"]
            selected_model = st.selectbox("選擇模型:", model_options)
        elif selected_api == "xAI":
            if api_key:
                model_options = get_xai_models(api_key)
            else:
                model_options = ["grok-beta", "grok-vision-beta"]
            selected_model = st.selectbox("選擇模型:", model_options)
            st.warning("⚠️ xAI (Grok) API 目前尚未公開，但已知有 grok-beta 及 grok-vision-beta 兩個模型。請等待官方發布 API 存取方式。")
        else:  # Ollama
            model_options = get_ollama_models()
            selected_model = st.selectbox("選擇本地模型:", model_options, 
                                        index=model_options.index("aya-expanse") if "aya-expanse" in model_options else 0)

        # 參數設定
        st.header("參數設定")
        temperature = st.slider("Temperature:", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
        top_p = st.slider("Top P:", min_value=0.0, max_value=1.0, value=0.9, step=0.1)
        max_tokens = st.slider("Max Tokens:", min_value=50, max_value=6000, value=2000, step=50)
        
        if selected_api == "OpenAI":
            presence_penalty = st.slider("Presence Penalty:", min_value=-2.0, max_value=2.0, value=0.0, step=0.1)
            frequency_penalty = st.slider("Frequency Penalty:", min_value=-2.0, max_value=2.0, value=0.0, step=0.1)
        else:
            presence_penalty = 0
            frequency_penalty = 0

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

    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("優化提示詞"):
            if not original_prompt:
                st.error("請輸入原始提示。")
                return

            if not api_key and selected_api != "Ollama":
                st.error(f"請輸入有效的 {selected_api} API Key。")
                return

            try:
                with st.spinner('優化中...'):
                    if selected_api == "OpenAI":
                        response = call_openai(api_key, selected_model, original_prompt, 
                                             temperature, top_p, max_tokens, 
                                             presence_penalty, frequency_penalty)
                    elif selected_api == "Gemini":
                        response = call_gemini(api_key, selected_model, original_prompt, 
                                             temperature, top_p, max_tokens)
                    elif selected_api == "xAI":
                        response = call_xai(api_key, original_prompt, temperature, 
                                          top_p, max_tokens)
                    else:  # Ollama
                        response = call_ollama(selected_model, original_prompt, temperature)

                    st.session_state.improved_prompt = response
            except Exception as e:
                st.error(f"優化過程中發生錯誤: {str(e)}")
                return

    # 修改顯示優化後的提示詞和生成按鈕的部分
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
                # 根據是否有用戶輸入來組合最終的提示詞
                final_prompt = (
                    f"{improved_prompt}\n\n用戶輸入：{user_input}"
                    if user_input
                    else improved_prompt
                )
                
                generated_content = execute_prompt(
                    selected_api, api_key, selected_model, final_prompt,
                    temperature, top_p, max_tokens, presence_penalty, frequency_penalty
                )
                st.subheader("生成的內容")
                # 使用 markdown 顯示生成的內容
                st.markdown(generated_content)
                # 保留原始文本顯示，方便複製
                with st.expander("顯示原始文本（方便複製）"):
                    st.text_area("原始文本:", value=generated_content, height=300)

if __name__ == "__main__":
    main()
