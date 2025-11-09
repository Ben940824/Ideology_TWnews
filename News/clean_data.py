import json
import re

# 定義要排除的標題
EXCLUDE_TITLE = "立委罷免二階開始送件 連署門檻、送件進度一次看【不斷更新】 ｜ 公視新聞網 PNN"

def clean_news_data(data):
    cleaned_data = []
    gongshi_count = 0
    hit_titles = []

    for item in data:
        # 若為排除目標，直接跳過
        if item['title'].strip() == EXCLUDE_TITLE:
            continue

        # 清理標題中的「｜ 公視新聞網 PNN」或變形
        item['title'] = re.sub(r" ｜?\s*公視新聞網(?:\s*PNN)?", "", item['title'])

        # 檢查 content 是否包含「公視」
        if "公視" in item['content']:
            gongshi_count += 1
            hit_titles.append(item['title'])

        cleaned_data.append(item)

    return cleaned_data, gongshi_count, hit_titles

if __name__ == "__main__":
    # 輸入與輸出檔案名
    INPUT_FILE = "公視_兩岸新聞_cleaned.json"
    OUTPUT_FILE = "公視_兩岸新聞_cleaned.json"

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    cleaned_data, gongshi_count, hit_titles = clean_news_data(raw_data)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)

    print(f"✅ 清理完成，已寫入 {OUTPUT_FILE}")
    print(f"🔍 發現有「公視」出現在 {gongshi_count} 篇 content 中")
    if gongshi_count > 0:
        print("📌 標題如下：")
        for t in hit_titles:
            print(f" - {t}")