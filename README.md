# AI 提示詞優化器

一個基於多種大型語言模型的提示詞優化工具，支援 Ollama、X.AI 和 Mistral AI。

## 功能特點

- 🤖 多模型支援：
  - Ollama（本地模型，如 aya-expanse）
  - X.AI（grok-2、grok-2-vision-1212、grok-beta、grok-vision-beta）
  - Mistral AI（mistral-tiny、mistral-small、mistral-medium）
- 🔄 提示詞優化：使用 AI 模型優化您的提示詞
- 🎯 提示詞執行：直接執行優化後的提示詞
- 💾 歷史記錄：自動儲存所有優化和生成的結果
- 🔧 參數調整：支援多種參數調整（Temperature、Max Tokens、Top P、Top K、Repeat Penalty）
- 🤖 即時串流：生成結果即時顯示
- 🇹🇼 繁體中文：完整的繁體中文介面

## 系統需求

- Python 3.8 或更高版本
- [Ollama](https://ollama.ai/)（如果要使用本地模型）
- X.AI API 金鑰（選用）
- Mistral AI API 金鑰（選用）
- 相關 Python 套件（見 requirements.txt）

## 安裝步驟

1. 克隆專案：
    ```bash
    git clone https://github.com/tutumomo/PromptGenerator.git
    cd PromptGenerator
    ```

2. 安裝依賴：
    ```bash
    pip install -r requirements.txt
    ```

3. 設定環境變數（如果要使用 X.AI 或 Mistral AI）：
    建立 `.env` 檔案並加入：
    ```env
    XAI_API_KEY=你的_X.AI_API_金鑰
    MISTRAL_API_KEY=你的_MISTRAL_API_金鑰
    ```

## 使用方法

1. 啟動應用程式：
    ```bash
    streamlit run app.py
    ```

2. 在瀏覽器中開啟顯示的網址（預設為 http://localhost:8501）

3. 選擇想要使用的 AI 模型和相關參數：
   - Ollama：選擇已安裝的本地模型
   - X.AI：需要 API 金鑰
   - Mistral AI：需要 API 金鑰

4. 輸入原始提示詞，點擊「優化提示詞」

5. 查看優化結果，可以選擇直接執行優化後的提示詞

## 注意事項

- 使用 Ollama 時需要確保服務正在運行
- 使用 X.AI 或 Mistral AI 時需要有效的 API 金鑰
- 建議使用虛擬環境來安裝依賴套件
- Mistral AI 需要 mistralai 1.2.3 或更高版本

## 授權

MIT License
