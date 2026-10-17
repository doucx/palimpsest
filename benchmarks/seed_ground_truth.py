#!/usr/bin/env python3
"""
Palimpsest - 黄金评测测试集生成与注入工具
用法:
  python benchmarks/seed_ground_truth.py
  python benchmarks/seed_ground_truth.py --output benchmarks/queries.jsonl --samples 25
"""

import argparse
import json
import os
import re
import sqlite3
from pathlib import Path


def seed_ground_truth(corpus_db_path: str, output_path: str, samples_per_category: int = 8):
    if not os.path.exists(corpus_db_path):
        raise FileNotFoundError(f"未找到提纯语料数据库: {corpus_db_path}，请先执行 extract_corpus.py")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    conn = sqlite3.connect(corpus_db_path)
    # 注入 Python 原生正则支持，防止 SQLite 缺少 REGEXP 扩展
    conn.create_function("REGEXP", 2, lambda expr, item: bool(re.search(expr, item or "")))
    cur = conn.cursor()

    cases = []

    # =========================================================================
    # 核心锚点 1: 真实翻车案例 (Bug Regression Anchor)
    # =========================================================================
    cur.execute("""
        SELECT t.file_id, t.turn_index, s.title, t.text_content
        FROM intent_turns t
        JOIN sessions s ON t.file_id = s.file_id
        WHERE t.text_content LIKE '%编译无任何错误%'
        LIMIT 1;
    """)
    翻车_row = cur.fetchone()
    if 翻车_row:
        fid, tidx, title, content = 翻车_row
        cases.append({
            "case_id": "TC-001-REGRESSION",
            "category": "1. 中英混排长句 (CJK Mixed)",
            "query": "执行 npm run check 确认 linter 和 TypeScript 编译无任何错误。",
            "target_file_id": fid,
            "target_turn": tidx,
            "target_title": title,
            "description": "导致现网系统 FTS5 发生 OR 降级并被高频词污染的真实翻车长句",
            "is_anchor": True,
        })
        cases.append({
            "case_id": "TC-002-REGRESSION",
            "category": "1. 中英混排长句 (CJK Mixed)",
            "query": "TypeScript 编译无任何错误。",
            "target_file_id": fid,
            "target_turn": tidx,
            "target_title": title,
            "description": "带中文全角句号的中英文混排短语检索",
            "is_anchor": True,
        })

    # =========================================================================
    # 类别 1: 中英文混排完整句子采样
    # =========================================================================
    cur.execute("""
        SELECT t.file_id, t.turn_index, s.title, t.text_content
        FROM intent_turns t
        JOIN sessions s ON t.file_id = s.file_id
        WHERE t.role = 'user' 
          AND length(t.text_content) BETWEEN 30 AND 400
          AND (t.text_content LIKE '%。%' OR t.text_content LIKE '%，%')
        LIMIT ?;
    """, (samples_per_category * 8,))
    
    # 纯 Python 内存正则过滤
    cjk_rows = cur.fetchall()
    count_cjk = 0
    for fid, tidx, title, text in cjk_rows:
        if count_cjk >= samples_per_category:
            break
        # 提取第一句完整的包含中文和英文单词的句子
        sentences = [s.strip() for s in re.split(r'[\n\r]+', text) if len(s.strip()) >= 15]
        for sent in sentences:
            if re.search(r'[\u4e00-\u9fa5]', sent) and re.search(r'[a-zA-Z]{3,}', sent) and sent.endswith(('。', '！', '？', '.')):
                cases.append({
                    "case_id": f"TC-CJK-{count_cjk+1:02d}",
                    "category": "1. 中英混排长句 (CJK Mixed)",
                    "query": sent[:80],
                    "target_file_id": fid,
                    "target_turn": tidx,
                    "target_title": title,
                    "description": "真实用户提问中的中英文混排句式",
                })
                count_cjk += 1
                break

    # =========================================================================
    # 类别 2: 多行连续代码块 (Multiline Block)
    # =========================================================================
    cur.execute("""
        SELECT t.file_id, t.turn_index, s.title, t.text_content
        FROM intent_turns t
        JOIN sessions s ON t.file_id = s.file_id
        WHERE t.has_code = 1 AND length(t.text_content) > 300
        LIMIT ?;
    """, (samples_per_category * 4,))
    
    code_rows = cur.fetchall()
    count_code = 0
    code_block_regex = re.compile(r"```(?:\w+)?\n([\s\S]+?)\n```")
    for fid, tidx, title, text in code_rows:
        if count_code >= samples_per_category:
            break
        blocks = code_block_regex.findall(text)
        for b in blocks:
            lines = [l.rstrip() for l in b.strip().splitlines() if l.strip()]
            if 3 <= len(lines) <= 8:
                query_code = "\n".join(lines[:4])
                cases.append({
                    "case_id": f"TC-CODE-{count_code+1:02d}",
                    "category": "2. 多行代码块 (Multiline Block)",
                    "query": query_code,
                    "target_file_id": fid,
                    "target_turn": tidx,
                    "target_title": title,
                    "description": "连续 3~4 行纯代码片段检索",
                })
                count_code += 1
                break

    # =========================================================================
    # 类别 3: 路径、异常 Traceback 与富符号 (Symbol Heavy)
    # =========================================================================
    cur.execute("""
        SELECT file_id, turn_index, symbol_text
        FROM context_symbols
        WHERE symbol_type = 'filepath' AND symbol_text LIKE '%.%' AND length(symbol_text) > 12
        LIMIT ?;
    """, (samples_per_category * 2,))
    sym_rows = cur.fetchall()
    count_sym = 0
    for fid, tidx, path_str in sym_rows:
        if count_sym >= samples_per_category:
            break
        cur.execute("SELECT title FROM sessions WHERE file_id = ?;", (fid,))
        s_row = cur.fetchone()
        title = s_row[0] if s_row else "Untitled"
        cases.append({
            "case_id": f"TC-PATH-{count_sym+1:02d}",
            "category": "3. 路径与调用符号 (Path & Symbol)",
            "query": path_str,
            "target_file_id": fid,
            "target_turn": tidx,
            "target_title": title,
            "description": f"精确定位文件工程路径: {path_str}",
        })
        count_sym += 1

    # =========================================================================
    # 类别 4: 代码缩进与排版扰动 (Whitespace Jitter)
    # =========================================================================
    count_jitter = 0
    for tc in [c for c in cases if c["category"] == "2. 多行代码块 (Multiline Block)"][:samples_per_category]:
        original_lines = tc["query"].splitlines()
        if len(original_lines) >= 2:
            # 改变缩进：增加前导空格并去掉末尾分号
            jittered = ["   " + l.strip().rstrip(";") for l in original_lines]
            cases.append({
                "case_id": f"TC-JITTER-{count_jitter+1:02d}",
                "category": "4. 缩进与符号扰动 (Jitter Fallback)",
                "query": "\n".join(jittered),
                "target_file_id": tc["target_file_id"],
                "target_turn": tc["target_turn"],
                "target_title": tc["target_title"],
                "description": "在真实代码块基础上人为改变缩进、移除分号",
            })
            count_jitter += 1

    # 保存至 JSONL
    with open(output_path, "w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"✨ 成功构建黄金评测集: {output_path} (共 {len(cases)} 个评测 Case)")
    categories = sorted(set(c["category"] for c in cases))
    for cat in categories:
        sub_c = [c for c in cases if c["category"] == cat]
        print(f"   • {cat}: {len(sub_c)} 条")


def main():
    parser = argparse.ArgumentParser(description="Palimpsest 黄金评测集种子生成器")
    parser.add_argument("--corpus-db", default="data/purified_corpus.db", help="提纯语料库路径")
    parser.add_argument("--output", default="benchmarks/queries.jsonl", help="输出 jsonl 路径")
    parser.add_argument("--samples", type=int, default=6, help="每个分类抽样规模")
    args = parser.parse_args()

    seed_ground_truth(args.corpus_db, args.output, args.samples)


if __name__ == "__main__":
    main()