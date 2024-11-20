# AI Prompt Generator

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.29.0-FF4B4B.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> 一個強大的 AI 提示詞優化工具，支援多種 AI 模型 API，幫助您生成更好的提示詞。
![專案截圖](https://github.com/user-attachments/assets/abb1fc15-e4f3-46ab-8b5a-02f20b0e7391)

## 支援的 API
- OpenAI (GPT-3.5/GPT-4)
- Google Gemini Pro
- xAI Grok
- Ollama（本地大型語言模型）

## 功能特點
- 多 API 支援
- 可調整的生成參數（溫度、top_p 等）
- 本地模型整合
- 直觀的網頁界面
- 提示詞歷史記錄
- 快取機制提升效能

## 系統需求
- Python 3.8 或更高版本
- pip（Python 包管理器）
- 網際網路連接
- （可選）Ollama 本地安裝

## 安裝步驟

### 1. 下載專案
```bash
git clone https://github.com/tutumomo/ai-prompt-generator.git
cd ai-prompt-generator
```

### 2. 建立虛擬環境（推薦）
Windows:
```bash
python -m venv venv
.\venv\Scripts\activate
```

macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. 安裝依賴
```bash
pip install -r requirements.txt
```

### 4. 設定 API 金鑰
有兩種方式可以設定 API 金鑰：

1. 環境變數（推薦）：
   - OpenAI API：設置 `OPENAI_API_KEY`
   - Gemini API：設置 `GEMINI_API_KEY`
   - xAI API：設置 `XAI_API_KEY`

   可以創建 `.env` 文件：
   ```env
   OPENAI_API_KEY=your-openai-key
   GEMINI_API_KEY=your-gemini-key
   XAI_API_KEY=your-xai-key
   ```

2. 直接在界面輸入：
   - 如果沒有設置環境變數，可以直接在側邊欄輸入 API 金鑰
   - 注意：這種方式需要每次重新啟動應用時重新輸入

### 5. Ollama 安裝（可選）
如果要使用本地模型，需要安裝 Ollama：
1. 訪問 [Ollama 官網](https://ollama.ai) 下載對應系統的安裝包
2. 安裝完成後，運行以下命令下載模型：
   ```bash
   # 下載 Llama 2 模型
   ollama pull llama2
   # 下載其他支援的模型
   ollama pull mistral
   ollama pull codellama
   ollama pull vicuna
   ```

## 執行應用

1. 確保虛擬環境已啟動

Windows:
```bash
.\venv\Scripts\activate
```

macOS/Linux:
```bash
source venv/bin/activate
```

2. 運行應用
```bash
streamlit run app.py
```

3. 瀏覽器會自動打開應用（預設地址：http://localhost:8501）

## 使用指南

### 1. 選擇 API
在側邊欄選擇想要使用的 API：
- OpenAI
- Gemini
- xAI
- Ollama（本地模型）

### 2. 輸入 API 金鑰
- 除了 Ollama 外，其他 API 都需要輸入對應的 API 金鑰
- API 金鑰會在會話期間保存，重新整理頁面後需要重新輸入

### 3. 選擇模型
- OpenAI：可選 GPT-3.5-turbo 或 GPT-4
- Gemini：使用 Gemini Pro
- xAI：使用 Grok-1
- Ollama：可選 llama2、mistral、codellama、vicuna 等

### 4. 調整參數
- Temperature（溫度）：控制輸出的創造性（0-1）
- Top P：控制輸出的多樣性
- Max Tokens：控制回應的最大長度
- Presence Penalty：降低重複主題的可能性
- Frequency Penalty：降低重複用詞的可能性

### 5. 輸入提示詞
在文本框中輸入你的提示詞，點擊「生成提示」按鈕開始生成。

## 常見問題

1. **API 金鑰無效**
   - 確認金鑰是否正確複製
   - 檢查金鑰是否仍然有效
   - 確認是否有足夠的額度

2. **Ollama 連接失敗**
   - 確認 Ollama 服務是否正在運行
   - 檢查是否已下載所選模型
   - 確認防火牆設置

3. **生成速度慢**
   - 檢查網路連接
   - 考慮減少 max_tokens 參數
   - 本地模型可能受限於硬體性能

## 貢獻指南
歡迎提交 Pull Request 或開設 Issue！

## 授權
本專案採用 MIT 授權條款 - 查看 [LICENSE](LICENSE) 文件了解更多細節

## 聯絡方式
- 作者：TUTUMOMO
- Email：tutumomo@gmail.com
- GitHub：[https://github.com/tutumomo](https://github.com/tutumomo)

---
Made with ❤️ by TUTUMOMO
