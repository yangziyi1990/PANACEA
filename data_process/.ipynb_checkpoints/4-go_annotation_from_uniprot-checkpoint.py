#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import time
import requests
from tqdm import tqdm
import pandas as pd
from requests.exceptions import RequestException

# ====================== UniProt 查询函数（用 TaxID） ======================

def query_uniprot_gene_to_go_by_taxid(gene_name: str, taxid: int):
    """
    使用 UniProt REST API，根据 gene_name + NCBI TaxID 查询 GO 注释。

    返回：list[tuple]
        (Gene, TaxID, UniProtID, GO_ID, GO_Type)
    """
    base_url = "https://rest.uniprot.org/uniprotkb/search"

    # 用 organism_id 精确限定物种
    query = f"gene_exact:{gene_name} AND organism_id:{taxid}"
    params = {
        "query": query,
        "fields": "accession,gene_names,go_id,go_p,go_c,go_f",
        "format": "json",
        "size": 1,      # 只取一个最匹配的 entry
    }

    try:
        resp = requests.get(base_url, params=params, timeout=20)
        resp.raise_for_status()
        results = resp.json().get("results", [])
        if not results:
            return []

        entry = results[0]
        uid = entry.get("primaryAccession", "")

        go_entries = []
        for xref in entry.get("uniProtKBCrossReferences", []):
            if xref.get("database") != "GO":
                continue
            go_id = xref.get("id", "")
            go_type = ""
            props = xref.get("properties") or []
            if props:
                go_type = props[0].get("value", "")
            go_entries.append((gene_name, taxid, uid, go_id, go_type))

        return go_entries

    except RequestException as e:
        print(f"[WARN] UniProt 请求失败：gene={gene_name}, taxid={taxid}, err={e}")
        return []
    except Exception as e:
        print(f"[WARN] 解析 UniProt 返回失败：gene={gene_name}, taxid={taxid}, err={e}")
        return []


def aggregate_go(go_df: pd.DataFrame) -> pd.DataFrame:
    """
    对原始 GO 行（每行一个 GO term）做聚合：
    输出列：
        Genes, TaxID, UniProtID,
        GO_Cellular_Component,
        GO_Molecular_Function,
        GO_Biological_Process
    """
    if go_df.empty:
        return pd.DataFrame(
            columns=[
                "Genes", "TaxID", "UniProtID",
                "GO_Cellular_Component",
                "GO_Molecular_Function",
                "GO_Biological_Process",
            ]
        )

    go_df = go_df.copy()
    go_df["GO_Merged"] = go_df["GO_ID"].astype(str) + "_" + go_df["GO_Type"].astype(str)

    def _agg_one(df_sub: pd.DataFrame, prefix: str):
        # prefix: "C:", "F:", "P:"
        return "; ".join(
            df_sub.loc[df_sub["GO_Type"].str.startswith(prefix, na=False), "GO_Merged"]
            .dropna()
            .unique()
        )

    agg = (
        go_df.groupby(["Genes", "TaxID", "UniProtID"])
        .apply(
            lambda d: pd.Series(
                {
                    "GO_Cellular_Component": _agg_one(d, "C:"),
                    "GO_Molecular_Function": _agg_one(d, "F:"),
                    "GO_Biological_Process": _agg_one(d, "P:"),
                }
            )
        )
        .reset_index()
    )
    return agg


# ====================== 主流程：针对每个菌株的 GGI 做 GO 注释 ======================

def annotate_go_for_all_strains(ppi_root: str,
                                taxmap_path: str,
                                go_root: str,
                                sleep_sec: float = 0.3):
    """
    对 ppi_root 下每个菌株目录里的 *.ggi.in_genes.tsv 做 GO 注释。

    - ppi_root:   data/3-string_ppi_data
    - taxmap_path: data/1-processed_data/strain_taxid_result_all.tsv
                   需要包含列：NormalizedSubject, TaxID
    - go_root:    data/4-go_annotation
    """

    os.makedirs(go_root, exist_ok=True)

    # 读入 TaxID 映射表
    df_taxmap = pd.read_csv(taxmap_path, sep="\t")
    df_taxmap["NormalizedSubject"] = df_taxmap["NormalizedSubject"].astype(str)

    # 建一个 dict: NormalizedSubject -> TaxID
    subject2taxid = {}
    for _, row in df_taxmap[df_taxmap["TaxID"].notna()].iterrows():
        subject2taxid[row["NormalizedSubject"]] = int(row["TaxID"])

    print(f"[INFO] TaxID 映射表中有 {len(subject2taxid)} 个菌株有合法 TaxID")

    # 遍历 PPI 目录下每个菌株的子目录
    strain_dirs = sorted(
        d for d in glob.glob(os.path.join(ppi_root, "*"))
        if os.path.isdir(d)
    )
    print(f"[INFO] 在 {ppi_root} 下发现 {len(strain_dirs)} 个菌株子目录")

    failed_strains = []

    for i, strain_dir in enumerate(strain_dirs, start=1):
        stem = os.path.basename(strain_dir)              # 如 Clostridioides_difficile_NAPCR1
        normalized_subject = stem.replace("_", " ")      # Clostridioides difficile NAPCR1

        print(f"\n[{i}/{len(strain_dirs)}] 处理菌株：{normalized_subject}")
        print(f"    - strain_dir = {strain_dir}")

        # 查 TaxID
        taxid = subject2taxid.get(normalized_subject)
        if taxid is None:
            print(f"[WARN] 在 TaxID 表中找不到 {normalized_subject} 的 TaxID，跳过。")
            failed_strains.append(normalized_subject)
            continue

        print(f"[INFO] 使用 TaxID = {taxid}")

        # 找 ggi.in_genes.tsv
        ggi_path = os.path.join(strain_dir, f"{stem}.ggi.in_genes.tsv")
        if not os.path.exists(ggi_path):
            print(f"[WARN] 未找到 GGI 文件：{ggi_path}，跳过。")
            failed_strains.append(normalized_subject)
            continue

        # 输出目录 & 文件
        strain_go_dir = os.path.join(go_root, stem)
        os.makedirs(strain_go_dir, exist_ok=True)
        go_raw_out = os.path.join(strain_go_dir, f"{stem}.go_raw.tsv")
        go_agg_out = os.path.join(strain_go_dir, f"{stem}.go_agg.tsv")

        # 如果已经有聚合结果，可以选择跳过
        if os.path.exists(go_agg_out):
            print(f"[SKIP] 已存在 GO 注释结果：{go_agg_out}，跳过。")
            continue

        # 1) 读取 GGI 文件，拿到基因集合（gene1 + gene2 并集）
        df_ggi = pd.read_csv(ggi_path, sep="\t")
        genes = sorted(set(df_ggi["gene1"].astype(str)) | set(df_ggi["gene2"].astype(str)))
        print(f"[INFO] GGI 中共涉及 {len(genes)} 个基因，将对其做 GO 注释")

        # 2) 针对每个基因查询 UniProt GO
        all_entries = []
        for g in tqdm(genes):
            terms = query_uniprot_gene_to_go_by_taxid(g, taxid)
            all_entries.extend(terms)
            time.sleep(sleep_sec)

        if not all_entries:
            print(f"[WARN] {normalized_subject} 无任何 GO 注释结果。")
            # 也可以写个空文件占位
            pd.DataFrame(columns=["Genes", "TaxID", "UniProtID", "GO_ID", "GO_Type"]).to_csv(
                go_raw_out, sep="\t", index=False
            )
            pd.DataFrame().to_csv(go_agg_out, sep="\t", index=False)
            continue

        # 3) 构建 DataFrame 并保存原始行
        go_df = pd.DataFrame(
            all_entries,
            columns=["Genes", "TaxID", "UniProtID", "GO_ID", "GO_Type"]
        )
        go_df.to_csv(go_raw_out, sep="\t", index=False)
        print(f"[INFO] 原始 GO 注释已写入：{go_raw_out}")

        # 4) 聚合并保存
        go_agg_df = aggregate_go(go_df)
        go_agg_df.to_csv(go_agg_out, sep="\t", index=False)
        print(f"[DONE] 聚合 GO 注释已写入：{go_agg_out}")

    # 总结
    if failed_strains:
        fail_out = os.path.join(go_root, "failed_strains_for_go.tsv")
        pd.Series(sorted(set(failed_strains))).to_csv(
            fail_out, sep="\t", index=False, header=False
        )
        print(f"\n[SUMMARY] 共 {len(set(failed_strains))} 个菌株 GO 注释失败，列表写入：{fail_out}")
    else:
        print("\n[SUMMARY] 所有菌株 GO 注释完成，无失败。")


# ====================== 脚本入口 ======================

if __name__ == "__main__":
    current_path = os.getcwd()
    HOME_DIR = os.path.dirname(current_path)

    PPI_ROOT   = f"{HOME_DIR}/data/3-string_ppi_data"
    TAXMAP_TSV = f"{HOME_DIR}/data/1-processed_data/strain_taxid_result_all.tsv"
    GO_ROOT    = f"{HOME_DIR}/data/4-go_annotation"

    annotate_go_for_all_strains(
        ppi_root=PPI_ROOT,
        taxmap_path=TAXMAP_TSV,
        go_root=GO_ROOT,
        sleep_sec=0.3,          # 可以根据请求速度调
    )