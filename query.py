import io
import os
import json
import anthropic
import voyageai
import pandas as pd
import requests
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

EXCEL_FILE_ID = "1ZBhQpO4cvu58uaD96rDVVFiypsM3r1SS"

_df_cache = None

STATS_DETECT_PROMPT = """以下の質問は「大学データの集計・カウント・一覧・ランキング」を求めていますか？
YESまたはNOのみ答えてください。

YESの例（Excelデータから集計・絞り込みが必要）：
- 「学生数が10000人以上の大学は何校？」
- 「偏差値60以上の私立大学の一覧を出して」
- 「関東の国立大学をリストアップして」
- 「在学者数の多い順にランキングして」

NOの例（PDFの内容を読んで答えるべき質問）：
- 「この16校に共通する認証評価の特徴は？」
- 「〇〇大学の改善指摘事項は？」
- 「地域連携が評価された事例を教えて」
- 「前の回答の大学について詳しく教えて」
- 「これらの大学の共通点は？」

※「この〇校」「これらの大学」「前の結果」などの指示語が含まれる場合は必ずNO。

質問: {question}

答え（YES/NO）:"""

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

STATS_FILTER_PROMPT = """
ユーザーの質問から、大学データを絞り込む条件を抽出してください。
該当しない場合はnullにしてください。

【フィールドの説明】
- 学校区分: 「私立」「国立」「公立」のいずれか
- 都道府県: 都道府県名のリスト（例: ["東京", "大阪"]）
- 地域名: 「関東」「関西」「東北」などの地方名
- 全体在学者数_以上: 在学者数（学生数）の下限（例: 質問に「10000人以上」とあれば10000）
- 全体在学者数_以下: 在学者数の上限
- 偏差値上限_以上: 大学の最高偏差値（偏差値上限）の下限（例: 「最高偏差値が60以上」なら60）
- 偏差値上限_以下: 大学の最高偏差値の上限
- 偏差値下限_以上: 大学の最低偏差値（偏差値下限）の下限（例: 「最低偏差値が55以上」「学部最低の偏差値が55以上」なら55）
- 偏差値下限_以下: 大学の最低偏差値の上限

質問: {question}

抽出するJSON形式（数値はint/floatで返す）:
{{
  "学校区分": null,
  "都道府県": null,
  "地域名": null,
  "全体在学者数_以上": null,
  "全体在学者数_以下": null,
  "偏差値上限_以上": null,
  "偏差値上限_以下": null,
  "偏差値下限_以上": null,
  "偏差値下限_以下": null
}}

JSONのみ返してください。
"""


def load_excel_df() -> pd.DataFrame:
    global _df_cache
    if _df_cache is not None:
        return _df_cache

    url = f"https://docs.google.com/spreadsheets/d/{EXCEL_FILE_ID}/export?format=xlsx"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    df = pd.read_excel(io.BytesIO(resp.content), sheet_name=0, header=0)
    df.columns = ["index", "大学名", "学校区分", "地域", "都道府県",
                  "全体在学者数", "偏差値上限", "偏差値下限"] + list(df.columns[8:])

    def safe_float(v):
        try:
            s = str(v).strip()
            if s in ["-", "BF", "ボーダーフリー", "", "nan", "None"]:
                return 35.0
            return float(s)
        except Exception:
            return 0.0

    df["全体在学者数"] = df["全体在学者数"].apply(safe_float)
    df["偏差値上限"] = df["偏差値上限"].apply(safe_float)
    df["偏差値下限"] = df["偏差値下限"].apply(safe_float)
    df = df[df["大学名"].notna() & (df["大学名"].astype(str).str.strip() != "") & (df["大学名"].astype(str) != "nan")]

    _df_cache = df
    return df


def is_stats_question(question: str, claude_client: anthropic.Anthropic) -> bool:
    try:
        resp = claude_client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=10,
            messages=[{"role": "user", "content": STATS_DETECT_PROMPT.format(question=question)}],
        )
        text = next((b.text for b in resp.content if hasattr(b, "text")), "NO")
        return "YES" in text.upper()
    except Exception:
        return False


def extract_number(text: str, keywords: list[str]) -> float | None:
    """質問文からキーワードに関連する数値を抽出する"""
    import re
    for kw in keywords:
        pattern = rf"{kw}[^0-9]*([0-9][0-9,，]*(?:\.[0-9]+)?)"
        m = re.search(pattern, text)
        if m:
            return float(m.group(1).replace(",", "").replace("，", ""))
    # キーワードの後ろだけでなく前も探す
    for kw in keywords:
        pattern = rf"([0-9][0-9,，]*(?:\.[0-9]+)?)[^0-9]*{kw}"
        m = re.search(pattern, text)
        if m:
            return float(m.group(1).replace(",", "").replace("，", ""))
    return None


def answer_from_excel(question: str, claude_client: anthropic.Anthropic) -> str:
    try:
        df = load_excel_df()
    except Exception as e:
        return f"大学データの読み込みに失敗しました: {e}"

    filtered = df.copy()
    applied = []

    # 学校区分
    for k in ["私立", "国立", "公立"]:
        if k in question:
            filtered = filtered[filtered["学校区分"] == k]
            applied.append(f"学校区分={k}")
            break

    # 地域・都道府県
    for region, prefs in REGION_MAP.items():
        if region in question:
            filtered = filtered[filtered["都道府県"].isin(prefs)]
            applied.append(f"地域={region}")
            break
    else:
        for pref in ["東京", "大阪", "京都", "神奈川", "愛知", "福岡", "北海道", "宮城",
                     "広島", "兵庫", "埼玉", "千葉", "静岡", "茨城", "栃木", "群馬"]:
            if pref in question:
                filtered = filtered[filtered["都道府県"] == pref]
                applied.append(f"都道府県={pref}")
                break

    # 在学者数
    num = extract_number(question, ["学生数", "在学者数", "人以上", "人超"])
    if num and ("以上" in question or "超" in question or "以上" in question):
        filtered = filtered[filtered["全体在学者数"] >= num]
        applied.append(f"在学者数≥{int(num)}")
    num = extract_number(question, ["学生数", "在学者数", "人以下", "人未満"])
    if num and ("以下" in question or "未満" in question):
        filtered = filtered[filtered["全体在学者数"] <= num]
        applied.append(f"在学者数≤{int(num)}")

    # 偏差値（上限・下限を文脈で判断）
    import re
    deviation_nums = re.findall(r"偏差値[^\d]*(\d+(?:\.\d+)?)", question)
    if deviation_nums:
        val = float(deviation_nums[0])
        if "最低" in question or "下限" in question or "最小" in question:
            if "以上" in question or "超" in question:
                filtered = filtered[filtered["偏差値下限"] >= val]
                applied.append(f"偏差値下限≥{val}")
            else:
                filtered = filtered[filtered["偏差値下限"] <= val]
                applied.append(f"偏差値下限≤{val}")
        elif "最高" in question or "上限" in question or "最大" in question:
            if "以上" in question or "超" in question:
                filtered = filtered[filtered["偏差値上限"] >= val]
                applied.append(f"偏差値上限≥{val}")
            else:
                filtered = filtered[filtered["偏差値上限"] <= val]
                applied.append(f"偏差値上限≤{val}")
        else:
            # デフォルト：偏差値上限で判定
            if "以上" in question or "超" in question:
                filtered = filtered[filtered["偏差値上限"] >= val]
                applied.append(f"偏差値上限≥{val}")
            else:
                filtered = filtered[filtered["偏差値上限"] <= val]
                applied.append(f"偏差値上限≤{val}")

    count = len(filtered)
    conds_str = "、".join(applied) if applied else "なし"

    rows = []
    for _, row in filtered.sort_values("全体在学者数", ascending=False).iterrows():
        rows.append(
            f"・{row['大学名']}（{row['学校区分']}／{row['都道府県']}）"
            f" 在学者数:{int(row['全体在学者数'])}人 偏差値:{row['偏差値下限']}〜{row['偏差値上限']}"
        )

    if count == 0:
        detail = "該当する大学はありませんでした。"
    elif count <= 100:
        detail = "\n".join(rows)
    else:
        detail = f"（{count}校あるため在学者数上位30校を表示）\n" + "\n".join(rows[:30])

    return f"【適用条件】{conds_str}\n\n【結果】{count}校が該当します。\n\n{detail}"


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


def answer(question: str, chunks: list[dict], claude_client: anthropic.Anthropic,
           history: list[dict] | None = None) -> str:
    context_parts = []
    for match in chunks:
        meta = match["metadata"]
        text = meta.get("text", "")
        source = meta.get("source", "不明")
        page = meta.get("page", "?")
        context_parts.append(f"【{source} p.{page}】\n{text}")

    context = "\n\n".join(context_parts)

    system_prompt = """あなたは大学認証評価・自己点検評価報告書の専門アシスタントです。
提供された参考資料と会話履歴をもとに質問に答えてください。
回答は日本語で、出典（大学名・報告書種別・ページ番号）を明記してください。
「この大学」「これらの大学」「前の回答」などの指示語は会話履歴を参照して解釈してください。"""

    user_content = f"""【参考資料】
{context}

【質問】
{question}"""

    # 会話履歴を含めてメッセージを構築
    messages = []
    if history:
        for msg in history[-6:]:  # 直近3往復分
            messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_content})

    response = claude_client.messages.create(
        model="claude-sonnet-5",
        max_tokens=2000,
        system=system_prompt,
        messages=messages,
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
    history: list[dict] | None = None,
) -> tuple[str, list[dict], dict]:
    claude_client = anthropic.Anthropic(api_key=anthropic_api_key)

    # 集計・一覧系の質問はExcelデータから直接回答
    if is_stats_question(question, claude_client):
        response_text = answer_from_excel(question, claude_client)
        return response_text, [], {"モード": "統計検索（Excelデータ）"}

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

    response_text = answer(question, matches, claude_client, history=history)
    return response_text, matches, filters
