#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import shutil

from pathlib import Path


def load_equivalence_map(equiv_path: str):
    """
    读取 strain_equivalence_classes.txt，构建：
    每个 stem -> 代表 stem 的映射。

    文件格式：每行是逗号分隔的 strain 名字（用下划线），例如：
    Clostridioides_difficile_NB95026,Clostridioides_difficile_CD630,Clostridioides_difficile_NAPCR1,

    约定：每一行的第一个菌株作为代表，其余视为与之等价。
    """
    mapping = {}

    if not os.path.exists(equiv_path):
        print(f"[INFO] 未找到等价类文件：{equiv_path}，不做等价优化。")
        return mapping

    with open(equiv_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",") if p.strip()]
            if len(parts) < 2:
                continue

            rep = parts[0]
            for p in parts:
                mapping.setdefault(p, rep)

    n_classes = len(set(mapping.values()))
    print(
        f"[INFO] 从 {equiv_path} 载入等价菌株类：{n_classes} 组，覆盖 {len(mapping)} 个菌株（stem）。"
    )
    return mapping


def main():
    current_path = os.getcwd()
    HOME_DIR = os.path.dirname(current_path)

    PROCESS_DIR = f"{HOME_DIR}/data/1-processed_data"
    OUTPUT_ROOT = f"{HOME_DIR}/data/3-string_ppi_data"

    equiv_path = os.path.join(PROCESS_DIR, "strain_equivalence_classes.txt")
    equivalence_map = load_equivalence_map(equiv_path)  # stem -> rep_stem

    if not equivalence_map:
        print("[INFO] 等价类信息为空，退出。")
        return

    # 构造：rep_stem -> [member_stem1, member_stem2, ...]
    rep2members = {}
    for stem, rep in equivalence_map.items():
        rep2members.setdefault(rep, []).append(stem)

    print(f"[INFO] 等价类代表数量：{len(rep2members)}")

    for rep_stem, members in rep2members.items():
        rep_dir = os.path.join(OUTPUT_ROOT, rep_stem)
        rep_ggi = os.path.join(rep_dir, f"{rep_stem}.ggi.tsv")
        rep_ggi_in = os.path.join(rep_dir, f"{rep_stem}.ggi.in_genes.tsv")

        if not (os.path.exists(rep_ggi) and os.path.exists(rep_ggi_in)):
            print(f"[WARN] 代表菌株 {rep_stem} 缺少 GGI 文件，跳过其等价类：")
            print(f"       {rep_ggi}")
            print(f"       {rep_ggi_in}")
            continue

        print(f"\n[REP] 代表菌株：{rep_stem}，成员数：{len(members)}")

        for stem in members:
            if stem == rep_stem:
                # 自己不需要复制
                continue

            member_dir = os.path.join(OUTPUT_ROOT, stem)
            os.makedirs(member_dir, exist_ok=True)

            member_ggi = os.path.join(member_dir, f"{stem}.ggi.tsv")
            member_ggi_in = os.path.join(member_dir, f"{stem}.ggi.in_genes.tsv")

            # 如果目标已经存在，可以选择跳过（避免覆盖）
            if os.path.exists(member_ggi) or os.path.exists(member_ggi_in):
                print(f"[SKIP] {stem} 目标文件已存在，跳过复制。")
                continue

            # 复制并改名：rep_stem.* -> stem.*
            shutil.copy2(rep_ggi, member_ggi)
            shutil.copy2(rep_ggi_in, member_ggi_in)

            print(f"[COPY] {rep_stem} -> {stem}")
            print(f"       {member_ggi}")
            print(f"       {member_ggi_in}")

    print("\n[DONE] 等价菌株 GGI 结果复制完成。")


if __name__ == "__main__":
    main()
