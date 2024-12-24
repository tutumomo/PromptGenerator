# 修改 import 區塊
import streamlit as st
import requests
from typing import List, Tuple
from dotenv import load_dotenv
import json
from datetime import datetime
import pathlib
import os
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

load_dotenv()  # 載入 .env 文件中的環境變數

# 環境變數映射
ENV_VAR_MAP = {
    "X.AI": "XAI_API_KEY",
    "Mistral": "MISTRAL_API_KEY"
}

XAI_MODELS = [
    "grok-2",           # 最新的 grok-2 模型
    "grok-2-vision-1212", # 支援視覺功能的 grok-2
    "grok-beta",        # 原始的 grok 模型
    "grok-vision-beta"  # 原始的視覺模型
]

def get_api_key(api_type: str) -> Tuple[str, str]:
    """
    檢查並獲取 API 金鑰
    返回: (api_key, message)
    """
    env_var = ENV_VAR_MAP.get(api_type)
    if not env_var:
        return None, ""
        
    api_key = os.getenv(env_var)
    if api_key:
        return api_key, f"✅ 已從環境變數 {env_var} 讀取 API 金鑰"
    return None, f"💡 可以設置環境變數 {env_var} 來儲存 API 金鑰"

# 修改 system prompt 常量
PROMPT_IMPROVEMENT_TEMPLATE = '''
# 角色 
你是一位AI prompt 專家，熟悉各種Prompt Optimizer框架(APE、CARE、CHAT、COAST、CREAT、CRISPE、RASCEF、RTF為主)，可以將使用者輸入的提示詞選定合適的Prompt框架後，編撰和優AI prompts。
## 重要:
- 無論提問使用何種語言，一律以繁體中文進行回答(禁用簡體中文，且須符合台灣用語習慣)。 

請分析並改進下提示詞:
{original_prompt}

只返回優化後的提示詞，不要有其他說明文字、標題或解釋(返回結果不要有"# 優化後提示詞："的字眼).
'''

# 修改 execute_prompt 函數以支援不同的 API
def execute_prompt(api_type: str, model: str, prompt: str, temperature: float, top_p: float, top_k: int, repeat_penalty: float, max_tokens: int, api_key: str = None) -> str:
    """
    執行提示詞並回傳生成的內容
    """
    try:
        if api_type == "Ollama":
            response = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    "model": model,
                    "prompt": prompt,
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "repeat_penalty": repeat_penalty,
                    "num_predict": max_tokens,
                    "stream": True
                },
                stream=True
            )
            
            # 建立一個佔位元素來顯示生成的文本
            placeholder = st.empty()
            generated_text = ""
            
            for line in response.iter_lines():
                if line:
                    json_response = json.loads(line)
                    chunk = json_response.get('response', '')
                    generated_text += chunk
                    # 更新顯示的文本
                    placeholder.markdown(generated_text + "▌")
            
            # 最後移除游標並返回完整文本
            placeholder.markdown(generated_text)
            return generated_text
            
        elif api_type == "X.AI":
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.post(
                "https://api.x.ai/v1/chat/completions",
                headers=headers,
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stream": True
                },
                stream=True
            )
            
            placeholder = st.empty()
            generated_text = ""
            
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        json_str = line[6:]
                        if json_str.strip() == '[DONE]':
                            break
                        try:
                            chunk_data = json.loads(json_str)
                            if 'choices' in chunk_data and len(chunk_data['choices']) > 0:
                                choice = chunk_data['choices'][0]
                                if 'delta' in choice and 'content' in choice['delta']:
                                    chunk = choice['delta']['content']
                                    generated_text += chunk
                                    placeholder.markdown(generated_text + "▌")
                        except json.JSONDecodeError:
                            continue
            
            placeholder.markdown(generated_text)
            return generated_text if generated_text else "未能生成回應"
            
        else:  # Mistral
            client = MistralClient(api_key=api_key)
            
            placeholder = st.empty()
            generated_text = ""
            
            try:
                messages = [{"role": "user", "content": prompt}]
                
                # 使用串流模式
                for chunk in client.chat_stream(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                ):
                    if chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        generated_text += content
                        placeholder.markdown(generated_text + "▌")
                
                placeholder.markdown(generated_text)
                return generated_text
                    
            except Exception as e:
                error_msg = f"Mistral API 錯誤: {str(e)}"
                st.error(error_msg)
                return error_msg
            
    except Exception as e:
        return f"生成失敗: {str(e)}"

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

# 在 import 區塊添加新的函數
def get_mistral_models(api_key: str) -> List[str]:
    """
    從 Mistral API 獲取可用的模型列表
    """
    default_models = [
        "mistral-medium",
        "mistral-small",
        "mistral-tiny"
    ]
    
    if not api_key:
        return default_models
        
    try:
        client = MistralClient(api_key=api_key)
        models = client.list_models()
        # 檢查回應格式並適當處理
        if isinstance(models, (list, tuple)):
            return [str(model) for model in models] if models else default_models
        elif hasattr(models, 'data'):
            return [model.id for model in models.data] if models.data else default_models
        else:
            st.warning("無法解析模型列表，使用預設值")
            return default_models
    except Exception as e:
        st.error(f"獲取 Mistral 模型列表失敗: {str(e)}")
        return default_models

def main():
    st.title("AI 提示詞優化器")
    
    with st.sidebar:
        st.header("設定")
        
        # API 選擇
        api_options = ["Ollama", "X.AI", "Mistral"]
        selected_api = st.selectbox("選擇 API:", api_options, index=0)
        
        # API 金鑰輸入（如果需要）
        if selected_api in ["X.AI", "Mistral"]:
            env_api_key, env_message = get_api_key(selected_api)
            st.info(env_message)
            
            api_key = st.text_input(
                f"{selected_api} API Key:",
                value=env_api_key if env_api_key else "",
                type="password",
                help=f"請輸入您的 {selected_api} API 金鑰，或在環境變數中設置 {ENV_VAR_MAP[selected_api]}"
            )
        else:
            api_key = None

        # 模型選擇
        if selected_api == "Ollama":
            model_options = get_ollama_models()
            selected_model = st.selectbox(
                "選擇本地模型:", 
                model_options, 
                index=model_options.index("aya-expanse") if "aya-expanse" in model_options else 0
            )
        elif selected_api == "X.AI":
            selected_model = st.selectbox(
                "選擇模型:",
                XAI_MODELS,
                help="X.AI 提供的大型語言模型。grok-1 是最新版本，支援更多功能。"
            )
        else:  # Mistral
            if api_key:
                model_options = get_mistral_models(api_key)
            else:
                model_options = [
                    "mistral-tiny-2402",
                    "mistral-small-2402",
                    "mistral-medium-2402",
                    "mistral-large-2402"
                ]
            selected_model = st.selectbox(
                "選擇模型:",
                model_options,
                help="Mistral AI 提供的大型語言模型"
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
        help="輸入你的原始提示詞，AI 將幫助你進使其更加有效。",
        height=150
    )

    # 存儲優化後的提示詞的 session state
    if 'improved_prompt' not in st.session_state:
        st.session_state.improved_prompt = ""

    if st.button("優化提示詞"):
        if not original_prompt:
            st.error("請輸入原始提示。")
            return
            
        if selected_api in ["X.AI", "Mistral"] and not api_key:
            st.error(f"請輸入 {selected_api} API 金鑰。")
            return

        try:
            with st.spinner('優化中...'):
                response = execute_prompt(
                    selected_api,
                    selected_model,
                    PROMPT_IMPROVEMENT_TEMPLATE.format(original_prompt=original_prompt),
                    temperature,
                    top_p,
                    top_k,
                    repeat_penalty,
                    max_tokens,
                    api_key
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
                    selected_api,
                    selected_model, 
                    final_prompt, 
                    temperature,
                    top_p,
                    top_k,
                    repeat_penalty,
                    max_tokens,
                    api_key
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