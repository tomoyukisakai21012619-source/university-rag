import os
import json
import anthropic
import voyageai
from pinecone import Pinecone

REGION_MAP = {
    "北海道": ["北海道"],
    "東北": ["青森", "岩手", "宮城", "秋田", "山形", "福島"],
    "関東": ["東京", "神奈川", "千葉", "埼玉", "茨城", "栃木", "群馬"],
    "首都圏": ["東京", "神奈川", "千葉", "埼玉"],
    "東海": ["静岡", "愛知", "岐阜", "三重"],
    "北陸": ["新潟", "富山", "石川", "福井"],
    "甲信越": ["山梨", "長野"],
    "近畿": ["大阪", "兵庫", "京都", "滋賀", "奈良", "和歌山"],
    "関西": ["大阪", "兵庫", "京都", "滋賀", "奈良", "和歌山"],
    "中国": ["鳥取", "島根", "岡山", "広島", "山口"],
    "四国": ["徳島", "香川", "愛媛", "高知"],
    "九州": ["福岡", "佐賀", "長崎", "熊本", "大分", "宮崎", "鹿児島"],
    "沖縄": ["沖縄"],
}

FILTER_EXTRACTION_PROMPT = """
ユーザーの質問から、以下の検索条件を抽出してください。
該当する条件がない場合はnullにしてください。

質問: {question}

抽出するJSON形式:
{{
  "大学名": null または "大学名（完全一致）",
  "学校区分": null または "私立" または "国立" または "公立",
  "都道府県": null または ["都道府県名", ...],
  "地域名": null または "関東" など（都道府県に変換します）,
  "全体在学者数_以上": null または 数値,
  "全体在学者数_以下": null または 数値,
  "偏差値_以上": null または 数値,
  "偏差値_以下": null または 数値,
  "報告書種別": null または "自己点検" または "認証評価結果"
}}

JSONのみ返してください。
"""


def extract_filters(question: str, client: anthropic.Anthropic) -> dict:
    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=500,
            messages=[{"role": "user", "content": FILTER_EXTRACTION_PROMPT.format(question=question)}],
        )
        raw = response.content[0].text.strip()
        filters = json.loads(raw)

        if filters.get("地域名") and not filters.get("都道府県"):
            region = filters["地域名"]
            for key, prefs in REGION_MAP.items():
                if key in region or region in key:
                    filters["都道府県"] = prefs
                    break

        return filters
    except Exception:
        return {}


def build_pinecone_filter(filters: dict) -> dict | None:
    conditions = []

    if filters.get("大学名"):
        conditions.append({"大学名": {"$eq": filters["大学名"]}})
    if filters.get("学校区分"):
        conditions.append({"学校区分": {"$eq": filters["学校区分"]}})
    if filters.get("都道府県"):
        conditions.append({"都道府県": {"$in": filters["都道府県"]}})
    if filters.get("全体在学者数_以上"):
        conditions.append({"全体在学者数": {"$gte": filters["全体在学者数_以上"]}})
    if filters.get("全体在学者数_以下"):
        conditions.append({"全体在学者数": {"$lte": filters["全体在学者数_以下"]}})
    if filters.get("偏差値_以上"):
        conditions.append({"偏差値上限": {"$gte": filters["偏差値_以上"]}})
    if filters.get("偏差値_以下"):
        conditions.append({"偏差値下限": {"$lte": filters["偏差値_以下"]}})
    if filters.get("報告書種別"):
        conditions.append({"報告書種別": {"$eq": filters["報告書種別"]}})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def search(question: str, voyage_client, pinecone_index, top_k: int = 20) -> list[dict]:
    embedding = voyage_client.embed([question], model="voyage-3", input_type="query").embeddings[0]
    results = pinecone_index.query(vector=embedding, top_k=top_k, include_metadata=True)
    return results["matches"]


def answer(question: str, chunks: list[dict], claude_client: anthropic.Anthropic) -> str:
    context_parts = []
    for match in chunks:
        meta = match["metadata"]
        text = meta.get("text", "")
        source = meta.get("source", "不明")
        page = meta.get("page", "?")
        context_parts.append(f"【{source} p.{page}】\n{text}")

    context = "\n\n".join(context_parts)

    prompt = f"""以下の大学認証評価・自己点検評価報告書の内容をもとに、質問に答えてください。
回答は日本語で、出典（大学名・報告書種別・ページ番号）を明記してください。
情報が不足している場合はその旨を伝えてください。

【参考資料】
{context}

【質問】
{question}
"""
    response = claude_client.messages.create(
        model="claude-sonnet-5",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    )
    text_block = next((b for b in response.content if hasattr(b, "text")), None)
    return text_block.text if text_block else "回答を生成できませんでした。"


def query_rag(
    question: str,
    anthropic_api_key: str,
    voyage_api_key: str,
    pinecone_api_key: str,
    pinecone_index_name: str = "university-rag",
    top_k: int = 20,
) -> tuple[str, list[dict], dict]:
    claude_client = anthropic.Anthropic(api_key=anthropic_api_key)
    voyage_client = voyageai.Client(api_key=voyage_api_key)
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(pinecone_index_name)

    filters = extract_filters(question, claude_client)
    pinecone_filter = build_pinecone_filter(filters)

    embedding = voyage_client.embed([question], model="voyage-3", input_type="query").embeddings[0]

    if pinecone_filter:
        results = index.query(vector=embedding, top_k=top_k, include_metadata=True, filter=pinecone_filter)
        matches = results["matches"]
        if len(matches) < 3:
            results = index.query(vector=embedding, top_k=top_k, include_metadata=True)
            matches = results["matches"]
            filters["フォールバック"] = True
    else:
        results = index.query(vector=embedding, top_k=top_k, include_metadata=True)
        matches = results["matches"]

    response_text = answer(question, matches, claude_client)
    return response_text, matches, filters
