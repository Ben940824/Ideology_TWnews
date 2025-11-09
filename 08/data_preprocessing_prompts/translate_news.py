#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import openai
from openai import OpenAI
import time
from tqdm import tqdm
import json
import argparse
import re


def setup_openai_api():
    """Set up the OpenAI API with the API key."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        api_key = input("Please enter your OpenAI API key: ")
        os.environ["OPENAI_API_KEY"] = api_key


def create_translation_prompt(text):
    """Create a prompt for translating news articles from Chinese to English."""
    prompt = """
當然可以。以下是根據你提供的新聞內容設計的 prompt，並附上翻譯範例：

⸻

Prompt:

你是一位專業的新聞翻譯員，擅長處理台灣的政治新聞，能準確保留原文的語氣與用詞，並以清晰、自然的英文呈現。請將下列繁體中文政治新聞翻譯成英文：

請注意以下翻譯原則：
	1.	忠實呈現政治人物的發言與立場，不做美化或誇大。
	2.	保留專有名詞與機構名稱（如：立法院、國民黨、民進黨），可在首次出現時加上英文說明。
	3.	若有台灣特有文化或制度，請簡要加上註解或以英文描述解釋。
	4.	採用正式、新聞風格的英文語氣（如 Reuters 或 BBC 報導風格）。
	5.	保持段落結構清晰，方便後續編輯使用。

以下是舉例：

⸻

【新聞標題】
大罷免解國會衝突？　賴清德喊話在野「不必貼民團標籤」

【新聞內文】
總統賴清德上任周年，朝野衝突愈發嚴重，各地更掀起大罷免潮。賴清德今（17）日接受專訪表示，人民有選舉、罷免、創制、複決的權利，這是人民公民運動、自動自發，沒有人有辦法主導或停止，不必去刻意貼公民罷免運動的標籤，也不要形塑是執政黨在策劃；今天公民團體走上街頭，一定有其主張與客觀事實，如果針對執政黨，好比在野黨要罷免總統，這必須要尊重；同樣若民團要罷免立委，相信相關的政黨也必須要尊重。至於用什麼方式來解決國會僵局，賴清德說，民主的問題必然是要用更大的民主來解決。

對於國會朝野僵局怎麼解，賴清德今晚接受知名時事網紅「敏迪選讀」專訪表示，民主的問題必然是要用更大的民主來解決。有兩個方式，就是憲政體制，也就是有司法院憲法法庭可以來裁判，立法院的這些法律有沒有侵犯其他單位的權力是可以解決的。但是因為立法院太多通過的法案都涉及違憲。即便是前總統陳水扁朝小野大的時代，國民黨也知道這不行、是侵犯行政權，「所以案子太多了，都違憲，那自然而然就會引起公民力量的出發。」

⸻

翻譯結果：

[Headline]
Can Mass Recalls Resolve Legislative Gridlock? President Lai Urges Opposition Not to Label Civil Groups

[News Article]
As President Lai Ching-te marks his first year in office, political conflict between the ruling and opposition parties has intensified, with mass recall movements emerging across the country. In an interview today (May 17), Lai stated that citizens possess the constitutional rights to vote, recall, initiate legislation, and hold referenda. These are forms of civic action, carried out autonomously by the people, beyond anyone’s control or orchestration.

He emphasized that there is no need to deliberately label recall campaigns as being manipulated by civic groups or orchestrated by the ruling party. “When civic groups take to the streets, it is because they have their own claims and factual grounds,” he said. If such actions target the ruling party—for instance, if the opposition seeks to recall the president—this should be respected. Likewise, if civil groups aim to recall legislators, related political parties should also show respect.

As for how to resolve the legislative impasse, Lai reiterated, “Democratic problems must be solved with more democracy.”

In a separate interview later that evening with prominent current affairs podcaster “Mindy’s Selection,” Lai elaborated on the legislative deadlock, again stating that democratic challenges must be met with greater democratic solutions. He suggested two possible paths: constitutional mechanisms, specifically the Constitutional Court under the Judicial Yuan, could determine whether laws passed by the Legislative Yuan infringe on the powers of other branches.

However, Lai pointed out that many of the bills passed by the legislature appear unconstitutional. He recalled the era of former President Chen Shui-bian, when the Democratic Progressive Party (DPP) held a minority in the legislature, saying even the Kuomintang (KMT) then recognized that certain actions would violate executive powers. “With so many unconstitutional bills, it naturally triggers a response from civil society,” he concluded.

⸻
請開始翻譯：
"""
    return prompt + text


def translate_text(text, model="gpt-4.1-mini", max_retries=3, retry_delay=5):
    """Translate the given text using the OpenAI API."""
    if not text or pd.isna(text) or text.strip() == "":
        return ""

    prompt = create_translation_prompt(text)

    for attempt in range(max_retries):
        try:
            client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional translator from Traditional Chinese to English, specializing in news articles.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=1500,
            )

            translation = response.choices[0].message.content.strip()

            # Extract headline and article content from the translation
            headline_match = re.search(
                r"\[Headline\](.*?)(?=\[News Article\]|\Z)", translation, re.DOTALL
            )
            article_match = re.search(
                r"\[News Article\](.*?)(?=\Z)", translation, re.DOTALL
            )

            if headline_match and article_match:
                # Return structured format for separate title and content extraction
                return {
                    "headline": headline_match.group(1).strip(),
                    "article": article_match.group(1).strip(),
                }
            else:
                # If the structured format isn't found, return the full translation
                return translation

        except Exception as e:
            print(f"Error during translation (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print(f"Failed to translate after {max_retries} attempts.")
                return f"[TRANSLATION ERROR] {str(e)}"


def translate_dataframe(df, model="gpt-4.1-mini", sample_size=None, batch_size=10):
    """Translate the title and content columns of the dataframe."""
    if sample_size:
        df = df.sample(sample_size, random_state=42)

    # Create new columns for translated text
    df["title_en"] = ""
    df["content_en"] = ""

    # Process in batches to provide better progress tracking
    total_batches = (len(df) + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(total_batches), desc="Processing batches"):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(df))
        batch = df.iloc[start_idx:end_idx]

        for idx, row in batch.iterrows():
            print(f"Translating article {idx} (ID: {row['id']})")

            # Combine title and content into one translation request
            if pd.notna(row["title"]) and pd.notna(row["content"]):
                full_text = f"""【新聞標題】\n{row["title"]}\n\n【新聞內文】\n{row["content"]}"""
                translation_result = translate_text(full_text, model=model)

                if isinstance(translation_result, dict):
                    # Extract the headline and article from structured response
                    df.at[idx, "title_en"] = translation_result.get("headline", "")
                    df.at[idx, "content_en"] = translation_result.get("article", "")
                    print(
                        f"Title translated: {translation_result.get('headline', '')[:50]}..."
                    )
                    print(
                        f"Content translated: {translation_result.get('article', '')[:50]}..."
                    )
                else:
                    # Fallback to storing the entire translation in content_en
                    df.at[idx, "content_en"] = translation_result
                    print(
                        f"Translation result stored in content field: {translation_result[:50]}..."
                    )
            else:
                # Handle cases where either title or content is missing
                if pd.notna(row["title"]) and row["title"].strip():
                    title_en = translate_text(row["title"], model=model)
                    if isinstance(title_en, dict):
                        df.at[idx, "title_en"] = title_en.get(
                            "headline", title_en.get("article", "")
                        )
                    else:
                        df.at[idx, "title_en"] = title_en
                    print(
                        f"Title translated separately: {df.at[idx, 'title_en'][:50]}..."
                    )

                if pd.notna(row["content"]) and row["content"].strip():
                    content_en = translate_text(row["content"], model=model)
                    if isinstance(content_en, dict):
                        df.at[idx, "content_en"] = content_en.get(
                            "article", content_en.get("headline", "")
                        )
                    else:
                        df.at[idx, "content_en"] = content_en
                    print(
                        f"Content translated separately: {df.at[idx, 'content_en'][:50]}..."
                    )

            # Save intermediate results every batch
            if (idx % batch_size == 0 or idx == len(df) - 1) and idx > 0:
                print(f"Saving intermediate results...")
                intermediate_output_path = (
                    f"news_translated_intermediate_{batch_idx}.csv"
                )
                df.to_csv(intermediate_output_path, index=False, encoding="utf-8-sig")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Translate news articles from Chinese to English"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data_tagging/results/news_training_simplified_20250528_141937.csv",
        help="Input CSV file path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data_tagging/results/news_training_english_translated.csv",
        help="Output CSV file path",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4.1-mini",
        help="OpenAI model to use for translation",
    )
    parser.add_argument(
        "--sample", type=int, default=None, help="Sample size for testing (optional)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=10, help="Batch size for processing"
    )

    args = parser.parse_args()

    # Set up API
    setup_openai_api()

    # Read CSV file
    print(f"Reading input file: {args.input}")
    df = pd.read_csv(args.input)
    print(f"Total articles: {len(df)}")

    # Translate articles
    print(f"Starting translation using model: {args.model}")
    translated_df = translate_dataframe(
        df, model=args.model, sample_size=args.sample, batch_size=args.batch_size
    )

    # Save translated dataframe to CSV
    print(f"Saving translated articles to: {args.output}")
    translated_df.to_csv(args.output, index=False, encoding="utf-8-sig")
    print("Translation completed!")


if __name__ == "__main__":
    main()
