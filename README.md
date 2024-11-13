# -AI-CUP-2024-

## AI CUP 2024 玉山人工智慧公開挑戰賽 - RAG 與 LLM 在金融問答的應用 - 初賽

這個專案是針對 AI CUP 2024 初賽而設計，目標是利用 RAG (Retrieval-Augmented Generation) 和大型語言模型 (LLM) 技術，在金融問答情境中進行應用開發。

### 專案功能概述

- **檔案檢索與問答系統**: 提供了一個基於自然語言處理技術的文件檢索與問答解決方案。
- **RAG 應用**: 利用檢索技術與生成式 AI 模型相結合，提升回答準確度。
- **中文文本處理**: 使用 `jieba` 進行中文文本的分詞與處理。
- **深度學習支援**: 使用 HuggingFace 的模型與工具來進行嵌入生成與回答生成。

### 目錄結構

- `README.md`: 提供專案的簡介與使用指南。
- `retrieval.py`: 實作文件檢索與問答功能，包括文本載入、檢索模型以及 HuggingFace 問答生成。
- `data_preprocess.py`: 包含 PDF 文檔的 OCR 處理與文本分割功能，以便後續的檢索與問答。

### 安裝需求

在使用此專案之前，請確保已安裝以下的 Python 依賴包，這些依賴包已在 `requirements.txt` 中列出，可以通過以下命令安裝：

```bash
pip install -r requirements.txt
```

### 使用方式

1. **載入文檔**
   
   使用 `load_documents` 函數載入文件。您可以根據需求指定不同的類別（例如 "faq" 或 "pdf"），並提供相應的文件來源清單。

2. **檢索與問答**
   
   載入文檔後，可以使用程式進行文本檢索，然後使用 HuggingFace 模型生成回應。

3. **OCR 文本處理**
   
   使用 `data_preprocess.py` 中的工具對 PDF 文檔進行 OCR 處理，以提取文本層並準備後續的檢索。

### 注意事項

- 建議在具有 GPU 支援的環境下運行此專案，以加速深度學習模型的推理過程。
- 如果使用加密的 PDF 文件，請確保文件已解密，否則可能無法載入內容。


