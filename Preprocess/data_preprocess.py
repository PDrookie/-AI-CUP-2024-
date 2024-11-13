import os
import sys
import shutil
import ocrmypdf
from PyPDF2 import PdfReader

def has_text_layer(pdf_path):
    """
    檢查 PDF 文件是否包含可提取的文字層。
    如果至少有一頁包含文字，則認為該 PDF 有文字層。

    參數：
        pdf_path (str): PDF 文件的路徑。

    返回：
        bool: 如果 PDF 文件中至少有一頁包含文字，返回 True，否則返回 False。
    """
    try:
        reader = PdfReader(pdf_path)
        # 僅檢查前 5 頁以提高速度
        for page_num in range(min(len(reader.pages), 5)):
            page = reader.pages[page_num]
            text = page.extract_text()
            if text and text.strip():
                # 如果發現文字，返回 True
                return True
        # 未發現文字，返回 False
        return False
    except Exception as e:
        print(f"❌ 無法讀取 {pdf_path}: {e}")
        return False

def process_pdf(input_pdf, output_pdf, languages='chi_tra+eng'):
    """
    使用 ocrmypdf 對 PDF 進行 OCR 處理，並將結果保存到指定位置。

    參數：
        input_pdf (str): 輸入 PDF 文件的路徑。
        output_pdf (str): 經 OCR 處理後保存的 PDF 文件路徑。
        languages (str): 設置 OCR 語言，默認為繁體中文（chi_tra）和英語（eng）。
    """
    try:
        ocrmypdf.ocr(
            input_pdf,
            output_pdf,
            deskew=True,            # 自動糾正傾斜
            language=languages,     # 設定語言
            force_ocr=False         # 僅對需要 OCR 的 PDF 進行處理
        )
        print(f"✅ OCR 處理完成: {input_pdf} -> {output_pdf}")
    except ocrmypdf.exceptions.PDFInfoNotInstalledError:
        print("❌ PDF 信息工具未安裝。請安裝 poppler 工具。")
    except ocrmypdf.exceptions.MissingDependencyError:
        print("❌ 缺少 ocrmypdf 依賴項。請確認 ocrmypdf 已正確安裝。")
    except Exception as e:
        print(f"❌ 無法處理 {input_pdf}: {e}")

def main(input_folder, output_folder):
    """
    主函數：批量處理指定文件夾中的 PDF 文件，進行 OCR 處理並保存結果。

    參數：
        input_folder (str): 包含待處理 PDF 文件的輸入文件夾。
        output_folder (str): 經過處理後保存 PDF 文件的輸出文件夾。
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        # 只處理 PDF 文件
        if filename.lower().endswith(".pdf"):
            input_pdf = os.path.join(input_folder, filename)
            output_pdf = os.path.join(output_folder, filename)

            if has_text_layer(input_pdf):
                # 如果已包含文字層，直接複製到輸出文件夾
                print(f"🔍 已包含文字層，跳過 OCR：{filename}")
                shutil.copy2(input_pdf, output_pdf)
            else:
                # 否則，進行 OCR 處理
                print(f"📝 進行 OCR 處理：{filename}")
                process_pdf(input_pdf, output_pdf)

if __name__ == "__main__":
    """
    主程序入口，負責解析命令行參數，並調用主函數進行 PDF OCR 處理。
    """
    # 檢查命令行參數
    if len(sys.argv) < 3:
        print("用法: python ocr_script.py <input_folder> <output_folder>")
        sys.exit(1)

    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    main(input_folder, output_folder)
