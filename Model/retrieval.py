import os
import json
import torch
import jieba  # 用於中文文本分詞
from rank_bm25 import BM25Okapi  # 使用 BM25 模型進行文檔檢索
from langchain.document_loaders import PyMuPDFLoader  # 加載 PDF 文件
from langchain.text_splitter import RecursiveCharacterTextSplitter  # 文本分割器
from langchain.vectorstores import Chroma  # 向量存儲器
from langchain.embeddings import HuggingFaceEmbeddings  # 嵌入式模型
from langchain.chains import RetrievalQA, LLMChain  # 撥回文答鏈
from langchain.prompts import PromptTemplate  # 提示模板
from langchain.docstore.document import Document  # 文件類
from tqdm import tqdm  # 進度條
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline  # 設定模型和 tokenizer
from langchain.llms import HuggingFacePipeline  # 採用 LLM 管道
import chromadb  # Chroma 向量數據庫
import re


def load_documents(category, source_list):
    """
    載入相關的文件，可供 FAQ 或 PDF 的資料使用。

    參數：
        category (字符串) : 文件類別，如「faq」或「pdf」
        source_list (清單) : 文件來源清單
    
    返回：
        documents (清單）: 一系列的 Document 對象
    """
    documents = []
    for source in source_list:
        if category == 'faq':
            # 載入 FAQ 資料
            file_path = "./reference/faq/pid_map_content.json"
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            qa_list = data.get(str(source), [])
            for qa in qa_list:
                question = qa.get('question', '')
                answers = qa.get('answers', [])
                content = f"question:{question}\nanswer:{' '.join(answers)}"
                doc = Document(
                    page_content=content,
                    metadata={'source_id': source}
                )
                documents.append(doc)
        else:
            # 載入 PDF 文件
            file_path = os.path.join(
                "./reference", category, f"{source}.pdf"
            )
            loader = PyMuPDFLoader(file_path)
            loaded_docs = loader.load()
            for doc in loaded_docs:
                doc.metadata['source_id'] = source
            documents.extend(loaded_docs)
    return documents


def load_questions(json_file):
    """
    載入問題，從 JSON 文件中載入

    參數：
        json_file (字符串）: 問題 JSON 文件的路徑
    
    返回：
        questions (清單）: 問題的清單
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get("questions", [])


def save_predictions(predictions, output_file="pred_retrieve.json"):
    """
    將預測的結果儲存到 JSON 文件。

    參數：
        predictions (清單): 預測的結果清單
        output_file (字符串）: 存放預測的路徑和文件名
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(
            {"answers": predictions},
            f, ensure_ascii=False, indent=4
        )


class BM25Retriever:
    """
    BM25 檢索器，使用 jieba 分詞對許問該案的資料進行檢索。

    專用使用 jieba 分詞對許資料輸入分辨及 BM25 模型的下讀與查詢。
    """
    def __init__(self, documents):
        """
        設置 BM25Retriever 的初始化設定
        
        參數：
            documents (清單）: 一系列的 Document 對象
        """
        self.documents = documents
        self.corpus = [doc.page_content for doc in self.documents]
        # 使用 jieba 進行分詞
        self.tokenized_corpus = [
            list(jieba.cut_for_search(doc)) for doc in self.corpus
        ]
        # BM25 模型
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def get_relevant_documents(self, query, k=3):
        """
        根據許問輸入和多少個要查詢的最佳文件對象，找最相關的前 k 個文件。

        參數：
            query (字符串）: 許問輸入內容
            k (整數）: 要查詢的最佳文件數
        
        返回：
            top_n_docs (清單）: 前 k 個最相關的 Document 對象
        """
        tokenized_query = list(jieba.cut_for_search(query))
        top_n_docs = self.bm25.get_top_n(
            tokenized_query, self.documents, n=k
        )
        return top_n_docs


os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 設置適用的 CUDA 設備

device = 0 if torch.cuda.is_available() else -1  # 選擇是否使用 GPU

# 設置 Chroma 數據庫的保存目錄
persist_directory = '/kaggle/working/db1'
if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)
    print(f"創建保存目錄 {persist_directory}。")
else:
    print(f"保存目錄 {persist_directory} 已存在。")

# 設定嵌入模型
model_name = "BAAI/bge-m3"
embedding = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs={'device': 'cuda'}
)

# 載入 Tokenizer 和語言模型
hf_model_name = "ckip-joint/bloom-3b-zh"
tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
model = AutoModelForCausalLM.from_pretrained(
    hf_model_name,
    is_decoder=True,
    torch_dtype=torch.float16
).to("cuda")

# 創建文本生成管道
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=20,
    clean_up_tokenization_spaces=False,
    device=0
)
llm = HuggingFacePipeline(pipeline=pipe)

# 設定 LLM 的提示模板
DEFAULT_SEARCH_PROMPT = PromptTemplate(
    input_variables=["question", "context"],
    template=(
        "你是一個助手，幫助根據問題從上下文中找到最相關的文件的 source ID。\n"
        "根據以下問題和上下文，請識別並提供最相關文件的 source ID。\n"
        "問題: {question}\n"
        "上下文: {context}\n"
        "請僅提供 source ID 的數字。"
    ),
)

# 初始化 LLM 鏈
prompt = DEFAULT_SEARCH_PROMPT
llm_chain = LLMChain(prompt=prompt, llm=llm)

# 載入問題
questions = load_questions("./questions_preliminary.json")

# 初始化預測的結果清單
predictions = []

# 創建 Chroma 客戶端
client = chromadb.Client()

# 處理每個問題
for idx, question in enumerate(tqdm(questions)):
    # 根據問題的類別和來源載入相關文件
    PDF_data = load_documents(
        question["category"], question["source"]
    )
    # 文件分割成小塊
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=400
    )
    all_splits = text_splitter.split_documents(PDF_data)
    torch.cuda.empty_cache()

    # 為每個問題創建唯一的集合名稱
    collection_name = f"collection_{idx}"
    # 從文件塊組建向量數據庫
    vectordb = Chroma.from_documents(
        documents=all_splits,
        embedding=embedding,
        persist_directory=persist_directory,
        collection_name=collection_name
    )
    # 從向量數據庫創建撥回器
    retriever = vectordb.as_retriever(search_kwargs={'k': 20})
    torch.cuda.empty_cache()

    # 獲取查詢和問題 ID
    query = question["query"]
    qid = question["qid"]

    # 使用撥回器獲取相關文件
    docs = retriever.get_relevant_documents(query)
    torch.cuda.empty_cache()

    if docs:
        # 不提供重排序器，直接選擇前二個文件
        docs = docs[:2]
        # 從選擇的文件創建上下文
        context = "\n\n".join([
            f"Source ID: {doc.metadata['source_id']}\n"
            f"Content: {doc.page_content}"
            for doc in docs
        ])
        # 使用 LLM 鏈獲取 source ID
        response = llm_chain.run(
            question=query,
            context=context,
            max_new_tokens=5
        )
        torch.cuda.empty_cache()
        # 從答案中提取 source ID
        match = re.search(r'\b\d+\b', response)
        if match:
            source_id = int(match.group(0))
        else:
            source_id = None
    else:
        source_id = None

    # 將預測結果新增到清單
    predictions.append({"qid": qid, "retrieve": source_id})
    print(source_id)
    torch.cuda.empty_cache()

# 輸出并儲存預測的結果
print(predictions)
save_predictions(predictions)

# 與真實值進行比較，評估預測的結果
ground_truths_file = "./ground_truths_example.json"
predictions_file = "./pred_retrieve.json"

with open(ground_truths_file, 'r', encoding='utf-8') as f:
    ground_truths_data = json.load(f)
    ground_truths = ground_truths_data.get("ground_truths", [])

with open(predictions_file, 'r', encoding='utf-8') as f:
    predictions_data = json.load(f)
    predictions = predictions_data.get("answers", [])

correct = 0
incorrect = 0
results = []
category_stats = {}

# 比較預測結果和真實值
for gt in ground_truths:
    qid = gt.get("qid")
    true_retrieve = gt.get("retrieve")
    category = gt.get("category", "Unknown")

    if category not in category_stats:
        category_stats[category] = {"correct": 0, "total": 0}
    category_stats[category]["total"] += 1

    pred = next(
        (p for p in predictions if p.get("qid") == qid), None
    )

    if pred is not None:
        predicted_retrieve = pred.get("retrieve")
        if predicted_retrieve == true_retrieve:
            correct += 1
            category_stats[category]["correct"] += 1
            results.append({
                "qid": qid,
                "correct": True,
                "category": category
            })
        else:
            incorrect += 1
            results.append({
                "qid": qid,
                "correct": False,
                "predicted": predicted_retrieve,
                "expected": true_retrieve,
                "category": category
            })
    else:
        incorrect += 1
        results.append({
            "qid": qid,
            "correct": False,
            "predicted": None,
            "expected": true_retrieve,
            "category": category
        })

# 輸出評估結果
print(f"總共問題數: {len(ground_truths)}")
print(f"正確的預測數: {correct}")
print(f"錯誤的預測數: {incorrect}")
accuracy = correct / len(ground_truths) * 100 if len(ground_truths) > 0 else 0
print(f"整體準確率: {accuracy:.2f}%\n")

print("按類別的準確率:")
print("-----------------")
for category, stats in category_stats.items():
    category_accuracy = (
        stats["correct"] / stats["total"] * 100
        if stats["total"] > 0 else 0
    )
    print(f"類別: {category}")
    print(f"  正確的預測數: {stats['correct']}")
    print(f"  總預測數: {stats['total']}")
    print(f"  準確率: {category_accuracy:.2f}%\n")
