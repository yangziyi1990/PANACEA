#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import glob
import pandas as pd
import requests
from urllib.parse import quote
from requests.exceptions import SSLError, ConnectionError, Timeout

# ====================== 物种 TaxID 映射 ======================

STRAIN_TAXID_MAP = {
    "Klebsiella pneumoniae": 573,
    "Staphylococcus aureus": 1280,
    "Escherichia coli": 562,
    "Pseudomonas aeruginosa": 287,
    "Mycobacterium tuberculosis": 1773,
}


def get_taxid_from_taxmap(strain_name: str, df_taxmap: pd.DataFrame) -> int:
    """
    优先从 df_taxmap 中获取 TaxID：
    1) 先按 NormalizedSubject 精确匹配
    2) 如果没有，再用前两个 token 作为 species 前缀匹配
    """
    # 1) 精确匹配 NormalizedSubject
    rows = df_taxmap.loc[df_taxmap["NormalizedSubject"] == strain_name]
    if len(rows) > 0 and pd.notna(rows.iloc[0]["TaxID"]):
        taxid = int(rows.iloc[0]["TaxID"])
        print(f"[INFO] 从 df_taxmap 精确匹配：{strain_name} -> TaxID={taxid}")
        return taxid

    # 2) species 前缀匹配（前两个 token）
    tokens = strain_name.split()
    species = " ".join(tokens[:2]) if len(tokens) >= 2 else strain_name

    rows_species = df_taxmap[
        (df_taxmap["NormalizedSubject"].str.startswith(species))
        & (df_taxmap["TaxID"].notna())
    ]

    if len(rows_species) > 0:
        taxid = int(rows_species.iloc[0]["TaxID"])
        print(
            f"[INFO] 从 df_taxmap 物种前缀匹配：{strain_name} (species={species}) -> TaxID={taxid}"
        )
        return taxid

    # 3) 都没有就抛错 / 或者 fallback 到原来的 get_taxid_from_name
    raise ValueError(
        f"在 df_taxmap 中找不到 '{strain_name}' 的 TaxID，请检查 NormalizedSubject 命名或手动补充。"
    )


def _fetch_taxid_from_ena(species_name: str):
    """
    尝试使用 ENA (EBI) 的 taxonomy REST API，根据物种名获取 NCBI TaxID。
    如果失败，返回 None。
    """
    base_url = "https://www.ebi.ac.uk/ena/taxonomy/rest/scientific-name/"
    url = base_url + quote(species_name)

    for attempt in range(1, 4):
        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code != 200:
                print(f"[WARN] ENA HTTP {resp.status_code}, attempt={attempt}")
                continue

            try:
                data = resp.json()
            except Exception:
                print(f"[WARN] ENA 非 JSON 返回，前 300 字符：\n{resp.text[:300]}")
                continue

            if not data:
                print(f"[WARN] ENA 未找到该物种：{species_name}")
                return None

            taxid = int(data[0]["taxId"])
            print(f"[INFO] ENA TaxID 映射：{species_name} -> {taxid}")
            return taxid

        except (SSLError, ConnectionError, Timeout) as e:
            print(f"[WARN] ENA 网络错误 attempt={attempt}: {e}")
            continue

    return None


def get_taxid_from_name(strain_name: str) -> int:
    """
    根据菌株名称获取 NCBI TaxID：
    1) 优先使用本地 STRAIN_TAXID_MAP
    2) 若本地没有，尝试 ENA
    """
    if strain_name in STRAIN_TAXID_MAP:
        taxid = STRAIN_TAXID_MAP[strain_name]
        print(f"[INFO] 使用本地 TaxID 映射：{strain_name} -> {taxid}")
        return taxid

    print(f"[INFO] 本地未找到 {strain_name}，尝试通过 ENA 获取 TaxID...")
    taxid = _fetch_taxid_from_ena(strain_name)
    if taxid is not None:
        return taxid

    raise ValueError(
        f"暂时无法为菌株 '{strain_name}' 获取 NCBI TaxID。\n"
        f"请在 STRAIN_TAXID_MAP 中手动补充。"
    )


# ====================== Step 1: 提取基因列表 ======================


def extract_gene_list(tsv_path: str):
    """
    从 TSV 文件中提取第一列作为基因名列表：
    - 去掉 '-' 和空字符串
    - 去重
    """
    df = pd.read_csv(tsv_path, sep="\t", header=None)
    genes = df.iloc[:, 0].astype(str)

    genes = genes[genes != "-"]
    genes = genes[genes != ""]
    genes = genes.dropna().unique().tolist()

    print(f"[INFO] 从 {tsv_path} 中提取到 {len(genes)} 个基因（去重后）")
    return genes


# ====================== 通用安全请求封装 ======================


def safe_post(url: str, data: dict, max_retries: int = 3, timeout: int = 30):
    """
    对 requests.post 做一层简单的重试封装。
    - 只捕获网络相关错误（SSL / Connection / Timeout）
    - 重试 max_retries 次，仍失败则返回 None
    """
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(url, data=data, timeout=timeout)
            return resp
        except (SSLError, ConnectionError, Timeout) as e:
            if attempt == max_retries:
                print(f"[ERROR] POST {url} 重试 {attempt}/{max_retries} 仍失败：{e}")
                return None
            wait = 1.0 * attempt
            print(
                f"[WARN] POST {url} 失败，重试 {attempt}/{max_retries}，"
                f"等待 {wait:.1f}s，错误：{e}"
            )
            time.sleep(wait)


def safe_get(url: str, params: dict, max_retries: int = 3, timeout: int = 30):
    """
    对 requests.get 做一层简单的重试封装。
    """
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            return resp
        except (SSLError, ConnectionError, Timeout) as e:
            if attempt == max_retries:
                print(f"[ERROR] GET {url} 重试 {attempt}/{max_retries} 仍失败：{e}")
                return None
            wait = 1.0 * attempt
            print(
                f"[WARN] GET {url} 失败，重试 {attempt}/{max_retries}，"
                f"等待 {wait:.1f}s，错误：{e}"
            )
            time.sleep(wait)


# ====================== Step 2.5: 基因名 → STRING ID 映射 ======================


def map_gene_names_to_string_ids(taxid: int, gene_list, batch_size: int = 200):
    """
    使用 STRING /get_string_ids 将基因名映射到 STRING ID / preferredName

    返回：
    - mapped_genes: list of dict，字段包含：
        - query          原始输入的 gene 名
        - stringId       STRING 内部 ID
        - preferredName  推荐的基因名
    - unmapped_genes: set，所有没有找到的 gene 名
    """
    string_api_url = "https://string-db.org/api"
    output_format = "json"
    method = "get_string_ids"
    request_url = "/".join([string_api_url, output_format, method])

    mapped = []
    unmapped = set()

    n = len(gene_list)
    for start in range(0, n, batch_size):
        sub = gene_list[start : start + batch_size]
        identifiers = "\r".join(sub)
        params = {
            "identifiers": identifiers,
            "species": taxid,
            "caller_identity": "AResKG_pipeline",
        }

        resp = safe_post(request_url, data=params, max_retries=3, timeout=60)

        if resp is None:
            # 整个 batch 都视为 unmapped，继续下一批
            print(
                f"[WARN] get_string_ids: batch {start}-{start+len(sub)} 请求失败，"
                f"将 {len(sub)} 个基因全部记为 unmapped"
            )
            unmapped.update(sub)
            continue

        if resp.status_code != 200:
            print(
                f"[WARN] get_string_ids HTTP {resp.status_code}, body: {resp.text[:200]}"
            )
            unmapped.update(sub)
            continue

        try:
            data = resp.json()
        except ValueError:
            print(f"[WARN] get_string_ids JSON 解析失败，body: {resp.text[:200]}")
            unmapped.update(sub)
            continue

        # data 是一个 list，每个元素对应某个 query 的一个匹配
        sub_set = set(sub)
        matched_queries = set()

        for row in data:
            q = row.get("queryItem")
            if q is None:
                continue
            matched_queries.add(q)
            mapped.append(
                {
                    "query": q,
                    "stringId": row.get("stringId"),
                    "preferredName": row.get("preferredName"),
                }
            )

        unmapped_now = sub_set - matched_queries
        unmapped.update(unmapped_now)

        print(
            f"[INFO] get_string_ids: batch {start}-{start+len(sub)}，"
            f"mapped={len(matched_queries)}, unmapped+={len(unmapped_now)}"
        )

    return mapped, unmapped


# ====================== Step 3: interaction_partners 获取 GGI ======================


def get_string_ggi_by_partners(
    taxid: int,
    gene_id_list,
    limit_per_gene: int = 50,
    min_score: int = 400,
    sleep_sec: float = 0.1,
) -> pd.DataFrame:
    """
    使用 STRING /interaction_partners 端点，逐基因获取 GGI，并合并去重。

    参数：
    - taxid: 物种 NCBI taxid (例如 573)
    - gene_id_list: 传给 STRING 的 identifiers（可以是 stringId，也可以是基因名）
    - limit_per_gene: 每个基因最多取多少个互作 partner
    - min_score: 过滤低于该分数的互作 (0–1000)：
         在结果中 score 通常是 0–1，这里用 score >= min_score/1000 过滤
    - sleep_sec: 每次请求之间 pause，避免被限流

    返回：
    - pandas.DataFrame: [gene1, gene2, score, method]
    """
    string_api_url = "https://string-db.org/api"
    output_format = "json"
    method = "interaction_partners"
    request_url = "/".join([string_api_url, output_format, method])

    edges = []
    n_genes = len(gene_id_list)
    score_threshold = min_score / 1000.0 if min_score > 1 else min_score

    for idx, gene in enumerate(gene_id_list, start=1):
        params = {
            "identifiers": gene,
            "species": taxid,
            "limit": limit_per_gene,
            "caller_identity": "AResKG_pipeline",
            "required_score": min_score,
        }

        resp = safe_get(request_url, params=params, max_retries=3, timeout=60)
        if resp is None:
            print(f"[WARN] interaction_partners: 基因 {gene} 请求失败，跳过")
            continue

        if resp.status_code != 200:
            print(
                f"[WARN] interaction_partners HTTP {resp.status_code} for gene {gene}, "
                f"body: {resp.text[:200]}"
            )
            continue

        try:
            data = resp.json()
        except ValueError:
            print(
                f"[WARN] interaction_partners JSON 解析失败，基因 {gene}，body: {resp.text[:200]}"
            )
            continue

        # data 是一个列表，每个元素是一条互作关系
        for row in data:
            score = row.get("score")
            if score is None:
                continue
            try:
                score = float(score)
            except Exception:
                continue

            # 过滤分数
            if score < score_threshold:
                continue

            g1 = row.get("preferredName_A") or row.get("stringId_A")
            g2 = row.get("preferredName_B") or row.get("stringId_B")
            if g1 is None or g2 is None:
                continue

            edges.append((g1, g2, score))

        if idx % 100 == 0 or idx == n_genes:
            print(
                f"[INFO] interaction_partners: 已处理 {idx}/{n_genes} 个基因，"
                f"累计边数：{len(edges)}"
            )

        time.sleep(sleep_sec)

    # ------- 合并去重：无向边 (gene1, gene2) 归一化顺序 -------
    edge_dict = {}  # key: (min(g1,g2), max(g1,g2)), value: max_score

    for g1, g2, score in edges:
        if g1 == g2:
            continue
        key = tuple(sorted((g1, g2)))
        if key not in edge_dict:
            edge_dict[key] = score
        else:
            edge_dict[key] = max(edge_dict[key], score)

    rows = []
    for (g1, g2), score in edge_dict.items():
        rows.append(
            {
                "gene1": g1,
                "gene2": g2,
                "score": score,
                "method": "STRING-DB/interaction_partners",
            }
        )

    df = pd.DataFrame(rows)
    print(f"[INFO] 去重后得到 {len(df)} 条 GGI 关系")
    return df


def get_genes_from_df(df_gene: pd.DataFrame, strain_name: str):
    """
    从 df_gene 中提取某个菌种在三元组里出现过的基因（has gene 的 Object）。

    返回：set[str]
    """
    mask = (df_gene["NormalizedSubject"] == strain_name) & (
        df_gene["Predicate"] == "has gene"
    )
    genes = set(df_gene.loc[mask, "Object"].astype(str))
    print(f"[INFO] 在 df_gene 中，{strain_name} 共出现 {len(genes)} 个 has gene 基因")
    return genes


# ====================== 主流程 ======================
def run_pipeline(
    tsv_path: str,
    strain_name: str,
    output_dir: str = "./output",
    df_gene: pd.DataFrame = None,
    df_taxmap: pd.DataFrame = None,
) -> pd.DataFrame:

    print(f"\n===== Processing {strain_name} =====")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] 输出目录：{output_dir}")

    # Step 1: TSV gene list
    genes_from_tsv = extract_gene_list(tsv_path)
    gene_set_tsv = set(genes_from_tsv)

    # ===== 整合 df_gene 里的基因 =====
    if df_gene is not None:
        genes_from_df = get_genes_from_df(df_gene, strain_name)

        inter = gene_set_tsv & genes_from_df
        only_in_df = genes_from_df - gene_set_tsv
        only_in_tsv = gene_set_tsv - genes_from_df

        print(f"[CHECK] TSV 基因数：{len(gene_set_tsv)}")
        print(f"[CHECK] df_gene 中 has gene 基因数：{len(genes_from_df)}")
        print(f"[CHECK] 交集基因数：{len(inter)}")
        print(f"[CHECK] df_gene 中但 TSV 里没有的基因数：{len(only_in_df)}")
        print(f"[CHECK] TSV 中但 df_gene 没用到的基因数：{len(only_in_tsv)}")

        # 使用并集
        genes = sorted(gene_set_tsv | genes_from_df)
        print(f"[INFO] 实际送去 STRING 的基因数（并集 union）：{len(genes)}")

    else:
        genes = genes_from_tsv
        print(f"[INFO] 未提供 df_gene，直接使用 TSV 中的 {len(genes)} 个基因")

    # 这里开始下面逻辑都用新的 genes 列表
    gene_set = set(genes)

    # Step 2: TaxID
    if df_taxmap is not None:
        taxid = get_taxid_from_taxmap(strain_name, df_taxmap)
    else:
        taxid = get_taxid_from_name(strain_name)

    print(f"[INFO] 使用 TaxID = {taxid}")

    # Step 2.5: gene → STRING ID 映射
    mapped, unmapped = map_gene_names_to_string_ids(taxid, genes, batch_size=200)

    print(f"[INFO] 总基因数（送去 STRING）：{len(genes)}")
    print(f"[INFO] 映射到 STRING 的基因数：{len(mapped)}")
    print(f"[INFO] 未能映射的基因数：{len(unmapped)} ({len(unmapped)/len(genes):.1%})")

    # 后面保持不变
    unmapped_out = os.path.join(
        output_dir, f"{strain_name.replace(' ', '_')}.unmapped_genes.tsv"
    )
    pd.Series(sorted(unmapped)).to_csv(
        unmapped_out, sep="\t", index=False, header=False
    )
    print(f"[INFO] 未映射基因列表已保存到：{unmapped_out}")

    identifiers = []
    for m in mapped:
        sid = m.get("stringId")
        identifiers.append(sid if sid else m.get("query"))
    identifiers = sorted(set(identifiers))
    print(f"[INFO] 用于 interaction_partners 的 identifiers 数量：{len(identifiers)}")

    df_ggi = get_string_ggi_by_partners(
        taxid,
        identifiers,
        limit_per_gene=50,
        min_score=400,
        sleep_sec=0.1,
    )

    outname = os.path.join(output_dir, f"{strain_name.replace(' ', '_')}.ggi.tsv")
    df_ggi.to_csv(outname, sep="\t", index=False)
    print(f"[DONE] 完整 GGI 文件已保存到：{outname}")

    filtered_ggi = df_ggi[
        df_ggi["gene1"].isin(gene_set) & df_ggi["gene2"].isin(gene_set)
    ].copy()

    filtered_out = os.path.join(
        output_dir, f"{strain_name.replace(' ', '_')}.ggi.in_genes.tsv"
    )
    filtered_ggi.to_csv(filtered_out, sep="\t", index=False)

    print(
        f"[DONE] 过滤后 GGI 文件已保存到：{filtered_out}\n"
        f"       过滤后边数：{len(filtered_ggi)}，覆盖基因数："
        f"{len(set(filtered_ggi['gene1']) | set(filtered_ggi['gene2']))}"
    )

    return filtered_ggi


if __name__ == "__main__":
    current_path = os.getcwd()
    HOME_DIR = os.path.dirname(current_path)

    # 1) 目录配置
    GENELIST_DIR = f"{HOME_DIR}/data/2-strains_gene_list"
    OUTPUT_ROOT = f"{HOME_DIR}/data/3-string_ppi_data"
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    # 2) 读入三元组与 TaxID 映射表（只读一次，全局复用）
    df_gene = pd.read_csv(
        f"{HOME_DIR}/data/1-processed_data/strain_species_803_norm.tsv", sep="\t"
    )
    df_taxmap = pd.read_csv(
        f"{HOME_DIR}/data/1-processed_data/strain_taxid_result_all.tsv", sep="\t"
    )

    # 只保留有 TaxID 的菌株名字集合，便于快速判断
    df_taxmap["NormalizedSubject"] = df_taxmap["NormalizedSubject"].astype(str)
    valid_taxid_subjects = set(
        df_taxmap.loc[df_taxmap["TaxID"].notna(), "NormalizedSubject"]
    )

    # 3) 找到所有 *.genes.tsv 文件
    all_tsvs = sorted(glob.glob(os.path.join(GENELIST_DIR, "*.genes.tsv")))
    print(f"[INFO] 在 {GENELIST_DIR} 中共发现 {len(all_tsvs)} 个 gene list TSV 文件")

    # 如果输出目录已存在且非空，是否跳过
    SKIP_IF_EXISTS = True

    failed_strains = []  # 记录失败的菌株名，最后汇总

    for i, tsv_path in enumerate(all_tsvs, start=1):
        fname = os.path.basename(
            tsv_path
        )  # 例如 Clostridioides_difficile_NAPCR1.genes.tsv
        stem = fname[: -len(".genes.tsv")]  # Clostridioides_difficile_NAPCR1
        # 约定：文件名下划线 → 空格，对应到 df_gene / df_taxmap 中的 NormalizedSubject
        strain_name = stem.replace("_", " ")  # Clostridioides difficile NAPCR1

        OUTPUT_STRAIN_DIR = os.path.join(OUTPUT_ROOT, stem)

        print(f"\n[{i}/{len(all_tsvs)}] 处理菌株：{strain_name}")
        print(f"    - GENE_TSV = {tsv_path}")
        print(f"    - OUTPUT_STRAIN_DIR = {OUTPUT_STRAIN_DIR}")

        # 3.1 若该菌株目录已存在且非空，直接跳过
        if (
            SKIP_IF_EXISTS
            and os.path.exists(OUTPUT_STRAIN_DIR)
            and os.listdir(OUTPUT_STRAIN_DIR)
        ):
            print(f"[SKIP] {OUTPUT_STRAIN_DIR} 已存在且非空，跳过该菌株。")
            continue

        os.makedirs(OUTPUT_STRAIN_DIR, exist_ok=True)

        # 3.2 检查是否在 df_taxmap 中有 TaxID
        if strain_name not in valid_taxid_subjects:
            print(f"[WARN] 在 df_taxmap 中找不到 {strain_name} 的 TaxID，跳过该菌株。")
            failed_strains.append(strain_name)
            continue

        # 3.3 调用主流程
        try:
            run_pipeline(
                tsv_path=tsv_path,
                strain_name=strain_name,
                output_dir=OUTPUT_STRAIN_DIR,
                df_gene=df_gene,
                df_taxmap=df_taxmap,
            )
        except Exception as e:
            print(f"[ERROR] {strain_name} 计算 GGI 失败：{e}")
            failed_strains.append(strain_name)
            # 不中断整个批处理，继续下一个菌株
            continue
        if i > 2:
            break

    # 4) 批处理总结
    if failed_strains:
        failed_strains = sorted(set(failed_strains))
        fail_out = os.path.join(OUTPUT_ROOT, "failed_strains.tsv")
        pd.Series(failed_strains).to_csv(fail_out, sep="\t", index=False, header=False)
        print(f"\n[SUMMARY] 共 {len(failed_strains)} 个菌株未成功，已写入：{fail_out}")
    else:
        print("\n[SUMMARY] 所有菌株 GGI 计算完成，无失败。")
