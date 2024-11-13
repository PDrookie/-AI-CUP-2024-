# 檔案檢索與問答腳本

此專案提供了一個基於自然語言處理技術的檔案檢索與問答系統，利用多種開源工具來處理中文文本，並實現高效的檔案檢索和問答功能。

## 功能概述

- 程式利用向量嵌入和Chroma向量資料庫進行語義搜索，從文檔中檢索與查詢相關的文本。
- 使用 jieba 進行中文分詞，以便更好地處理中文文本。
- 支援從 PDF 文件中載入內容，並進行文本分割，使得長文本能夠更有效地進行檢索。
- 提供基於 LLMChain 和 HuggingFace 模型的問答功能。

## 環境需求

在使用此腳本之前，請先確保已安裝以下的 Python 依賴包：

- os
- json
- torch
- jieba
- rank_bm25
- langchain
- transformers
- tqdm
- chromadb

可以通過以下命令來安裝所需依賴：

```bash
pip install torch jieba rank-bm25 langchain transformers tqdm chromadb
```

## 使用方法

### 載入文檔

使用 `load_documents(category, source_list)` 函數來載入文件。您可以根據需求指定不同的類別（如 "faq" 或 "pdf"），並提供相應的文件來源清單。

### 文本檢索與問答

載入文件後，利用 HuggingFaceEmbeddings 和 Chroma 向量存儲來生成向量嵌入，支持語義上的查詢與匹配。

### 提示模板

該腳本利用 `PromptTemplate` 來生成提示，用於優化回應的生成過程。

## 文件結構

- `load_documents(category, source_list)`:
  - 此函數用於載入指定類別的文件，並返回一系列的 Document 對象。

## 注意事項

- 該腳本需要 GPU 支援以便更快地運行深度學習模型，建議在有 GPU 的環境下使用。
- 如果使用 PDF 文件作為來源，請確保這些文件沒有加密，否則可能無法正確載入內容。
