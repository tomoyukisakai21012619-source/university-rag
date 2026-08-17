# =====================================================================
# 大学認証評価報告書 インデックス作成スクリプト
# Google Colab で実行してください
#
# 【事前準備】
# 1. Google Colab を開き、このファイルの内容を貼り付けて実行
# 2. Google Drive をマウント（自動的に求められます）
# 3. 下記の設定値を書き換えてから実行
# =====================================================================

# ---- 設定 ここを書き換えてください ----
VOYAGE_API_KEY   = "pa-xxxxxxxx"          # Voyage AI APIキー
PINECONE_API_KEY = "xxxxxxxx"             # Pinecone APIキー
PINECONE_INDEX   = "university-rag"       # Pineconeインデックス名

# Google Drive上のPDFフォルダパス（マウント後のパス）
PDF_FOLDER = "/content/drive/MyDrive/大学調査/認証評価報告書等/自己点検評価報告書のみ"

# メタデータExcelファイルのパス
EXCEL_PATH = "/content/drive/MyDrive/大学調査/認証評価報告書等/大学一覧（高企部ターゲット整理）｜2025整理元シート.xlsx"

# ---- 設定ここまで ----

import os
import re
import time
import pandas as pd
import pdfplumber
import voyageai
from pinecone import Pinecone, ServerlessSpec
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---- Google Drive マウント（Colab上で実行時）----
try:
    from google.colab import drive
    drive.mount("/content/drive")
    print("Google Drive マウント完了")
except ImportError:
    print("Colab以外の環境では Google Drive マウントをスキップ")

# ---- パッケージインストール（Colab上で実行時）----
os.system("pip install -q pdfplumber voyageai pinecone pandas openpyxl")


def load_metadata(excel_path: str) -> dict:
    """Excelから大学メタデータを読み込んでdict化"""
    df = pd.read_excel(excel_path, sheet_name=0, header=0)
    meta_dict = {}
    for _, row in df.iterrows():
        name = str(row.iloc[0]).strip()
        if not name or name == "nan":
            continue
        meta_dict[name] = {
            "学校区分": str(row.iloc[1]).strip() if pd.notna(row.iloc[1]) else "",
            "地域":     str(row.iloc[2]).strip() if pd.notna(row.iloc[2]) else "",
            "都道府県": str(row.iloc[3]).strip() if pd.notna(row.iloc[3]) else "",
            "全体在学者数": float(row.iloc[4]) if pd.notna(row.iloc[4]) else 0,
            "偏差値上限":   float(row.iloc[5]) if pd.notna(row.iloc[5]) else 0,
            "偏差値下限":   float(row.iloc[6]) if pd.notna(row.iloc[6]) else 0,
        }
    print(f"メタデータ読み込み: {len(meta_dict)}大学")
    return meta_dict


def extract_university_name(filename: str) -> str:
    """ファイル名から大学名を抽出"""
    name = os.path.splitext(filename)[0]
    name = re.sub(r"_(自己点検評価報告書|認証評価結果報告書|自己点検|認証評価).*$", "", name)
    return name.strip()


def detect_report_type(filename: str) -> str:
    if "認証評価" in filename:
        return "認証評価結果"
    return "自己点検"


def extract_chunks(pdf_path: str, chunk_size: int = 500, overlap: int = 50) -> list[dict]:
    """PDFからテキストチャンクを抽出"""
    chunks = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            full_text_pages = []
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text and len(text.strip()) > 50:
                    full_text_pages.append((i + 1, text.strip()))

            for page_num, text in full_text_pages:
                words = text.split()
                for start in range(0, len(words), chunk_size - overlap):
                    chunk_words = words[start:start + chunk_size]
                    if len(chunk_words) < 30:
                        continue
                    chunks.append({
                        "text": " ".join(chunk_words),
                        "page": page_num,
                    })
    except Exception as e:
        print(f"  PDF読み込みエラー: {e}")
    return chunks


def process_single_file(args):
    pdf_path, meta_dict, voyage_client = args
    filename = os.path.basename(pdf_path)
    univ_name = extract_university_name(filename)
    report_type = detect_report_type(filename)

    meta = meta_dict.get(univ_name, {})
    if not meta:
        for key in meta_dict:
            if univ_name in key or key in univ_name:
                meta = meta_dict[key]
                break

    chunks = extract_chunks(pdf_path)
    if not chunks:
        return [], filename, "スキップ（テキストなし）"

    texts = [c["text"] for c in chunks]
    try:
        embeddings = voyage_client.embed(texts, model="voyage-3", input_type="document").embeddings
    except Exception as e:
        return [], filename, f"埋め込みエラー: {e}"

    vectors = []
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        vector_id = f"{univ_name}_{report_type}_{chunk['page']}_{i}"
        vectors.append({
            "id": vector_id,
            "values": emb,
            "metadata": {
                "text": chunk["text"][:1000],
                "source": filename,
                "page": chunk["page"],
                "大学名": univ_name,
                "報告書種別": report_type,
                "学校区分": meta.get("学校区分", ""),
                "都道府県": meta.get("都道府県", ""),
                "地域": meta.get("地域", ""),
                "全体在学者数": meta.get("全体在学者数", 0),
                "偏差値上限": meta.get("偏差値上限", 0),
                "偏差値下限": meta.get("偏差値下限", 0),
            }
        })

    return vectors, filename, f"{len(vectors)}チャンク"


def upsert_to_pinecone(index, vectors: list, batch_size: int = 100):
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        index.upsert(vectors=batch)
    return len(vectors)


def main():
    print("=" * 60)
    print("大学認証評価報告書 インデックス作成")
    print("=" * 60)

    # メタデータ読み込み
    meta_dict = load_metadata(EXCEL_PATH)

    # PDFファイル一覧取得
    pdf_files = [
        os.path.join(PDF_FOLDER, f)
        for f in os.listdir(PDF_FOLDER)
        if f.endswith(".pdf")
    ]
    print(f"PDFファイル数: {len(pdf_files)}")

    # Voyage AI クライアント
    voyage_client = voyageai.Client(api_key=VOYAGE_API_KEY)

    # Pinecone 初期化
    pc = Pinecone(api_key=PINECONE_API_KEY)

    if PINECONE_INDEX not in [idx.name for idx in pc.list_indexes()]:
        print(f"Pineconeインデックス '{PINECONE_INDEX}' を作成中...")
        pc.create_index(
            name=PINECONE_INDEX,
            dimension=1024,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        time.sleep(10)
        print("インデックス作成完了")
    else:
        print(f"既存インデックス '{PINECONE_INDEX}' を使用")

    index = pc.Index(PINECONE_INDEX)

    # 並列処理でインデックス作成
    total_chunks = 0
    args_list = [(path, meta_dict, voyage_client) for path in pdf_files]

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(process_single_file, args): args[0] for args in args_list}
        for i, future in enumerate(as_completed(futures), 1):
            vectors, filename, status = future.result()
            print(f"[{i}/{len(pdf_files)}] {filename}: {status}")
            if vectors:
                upsert_to_pinecone(index, vectors)
                total_chunks += len(vectors)

    stats = index.describe_index_stats()
    print(f"\n完了！合計チャンク数: {total_chunks}")
    print(f"Pinecone DB合計: {stats['total_vector_count']}ベクトル")


if __name__ == "__main__":
    main()
