import os

os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')

import sys
import json
import gc
import zlib
import traceback
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)

# case 6/12 缺乏必要例句，跳过；其余 case 均有 LLM_contexts/case{n}.json
ALL_CASES = (1, 2, 3, 4, 5, 7, 8, 9, 10, 11)

sys.path.insert(0, PARENT_DIR)
from SMM import SemanticMap
from scipy.stats import pearsonr, spearmanr


def case_paths(case):
    prompts_file = os.path.join(BASE_DIR, 'LLM_contexts', f'case{case}.json')
    gt_file = os.path.join(BASE_DIR, '..', 'data_pure', f'case{case}.xlsx')
    sim_dir = os.path.join(BASE_DIR, 'LLM_similarity', f'case{case}')
    return prompts_file, gt_file, sim_dir


def parse_args(argv):
    dry_run = '--dry-run' in argv
    center = '--center' in argv
    selected = None
    if '--models' in argv:
        i = argv.index('--models')
        if i + 1 < len(argv):
            selected = [m.strip() for m in argv[i + 1].split(',') if m.strip()]
    selected_cases = None
    if '--cases' in argv:
        i = argv.index('--cases')
        if i + 1 < len(argv):
            try:
                selected_cases = [int(x.strip()) for x in argv[i + 1].split(',') if x.strip()]
            except ValueError:
                sys.exit('--cases 参数格式错误，应为逗号分隔的数字，如 --cases 1,2,3')
    return dry_run, center, selected, selected_cases


DRY_RUN, CENTER, SELECTED_MODELS, SELECTED_CASES = parse_args(sys.argv[1:])
CASES = list(ALL_CASES if SELECTED_CASES is None else SELECTED_CASES)
if SELECTED_CASES is not None:
    unknown = [c for c in CASES if c not in ALL_CASES]
    if unknown:
        sys.exit(f'未知 case: {unknown}，可选: {list(ALL_CASES)}')

ENCODE_BATCH_SIZE = 2
TARGET_BATCH_SIZE = 32
TARGET_MAX_LENGTH = 512

TEXT_CONTEXT_TYPES = ('流畅综合', '单独定义', '简单综合')
TOKEN_CONTEXT_TYPE = '单独例句'
TOKEN_MATRIX_MEAN = '单独例句_均值池化'
TOKEN_MATRIX_PAIRWISE = '单独例句_两两余弦'

# Sheet1 前导标签列（其后才是功能列）：case2 多 type 列，case8 多 语组 列
LEADING_LABEL_COLS = {'languages', '语组', 'forms', 'type'}

# name: 输出子目录与结果表中的模型名; pooling: 句向量池化 (cls/mean/last_token);
# max_length: 句子编码上限; dim: dry-run 伪向量维度 (真实向量自动取模型隐层维度);
# note: 运行时打印的注意事项
MODEL_SPECS = [
    {'name': 'bge-m3', 'repo': 'BAAI/bge-m3',
     'pooling': 'cls', 'max_length': 8192, 'dim': 1024,
     'note': 'CLS pooling，与 FlagEmbedding dense_vecs 等价'},
    {'name': 'xlm-roberta-base', 'repo': 'FacebookAI/xlm-roberta-base',
     'pooling': 'mean', 'max_length': 512, 'dim': 768,
     'note': '无专用句向量，mean pooling；max_length=512，长 context 会被截断；'
             '原始向量各向异性强（相似度趋近 1），建议配合 --center 使用'},
    {'name': 'xlm-roberta-large', 'repo': 'FacebookAI/xlm-roberta-large',
     'pooling': 'mean', 'max_length': 512, 'dim': 1024,
     'note': '无专用句向量，mean pooling；max_length=512，长 context 会被截断；'
             '原始向量各向异性强（相似度趋近 1），建议配合 --center 使用'},
    {'name': 'mmBERT-small', 'repo': 'jhu-clsp/mmBERT-small',
     'pooling': 'mean', 'max_length': 8192, 'dim': 384,
     'note': 'ModernBERT 架构，需 transformers>=4.48；Volta 无 flash-attn，回退 sdpa/eager'},
    {'name': 'mmBERT-base', 'repo': 'jhu-clsp/mmBERT-base',
     'pooling': 'mean', 'max_length': 8192, 'dim': 768,
     'note': 'ModernBERT 架构，需 transformers>=4.48；Volta 无 flash-attn，回退 sdpa/eager'},
    {'name': 'embeddinggemma-300m', 'repo': 'google/embeddinggemma-300m',
     'pooling': 'mean', 'max_length': 2048, 'dim': 768,
     'dtype': 'float32', 'prompt': 'title: none | text: {}', 'st_encode': True,
     'note': '双向编码器（T5Gemma 改造），官方句向量 = mean pooling + Dense(768→3072→768) + L2，'
             '故句向量走 sentence-transformers 官方管线（需安装）；fp16 不支持，强制 float32；'
             'token 向量取 backbone 最高层隐状态；gated repo：token 写入 ARR_Oct/.hf_token 或设 HF_TOKEN'},
    {'name': 'Qwen3-Embedding-0.6B', 'repo': 'Qwen/Qwen3-Embedding-0.6B',
     'pooling': 'last_token', 'max_length': 8192, 'dim': 1024, 'left_padding': True,
     'note': '需 transformers>=4.51；官方文档格式不加 instruction'},
    {'name': 'Qwen3-Embedding-4B', 'repo': 'Qwen/Qwen3-Embedding-4B',
     'pooling': 'last_token', 'max_length': 8192, 'dim': 2560, 'left_padding': True,
     'note': '需 transformers>=4.51；fp16 约 8GB 显存'},
    {'name': 'Qwen3-Embedding-8B', 'repo': 'Qwen/Qwen3-Embedding-8B',
     'pooling': 'last_token', 'max_length': 8192, 'dim': 4096, 'left_padding': True,
     'note': '需 transformers>=4.51；fp16 约 16GB 显存'},
]


def l2_normalize(vecs):
    vecs = np.asarray(vecs, dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


def center_vecs(vecs):
    """减去向量集的公共均值方向 (all-but-the-top)，缓解未微调 MLM 的各向异性。"""
    vecs = np.asarray(vecs, dtype=np.float32)
    return vecs - vecs.mean(axis=0, keepdims=True)


def cosine_matrix(func_vecs, center=False):
    if center:
        func_vecs = center_vecs(func_vecs)
    func_vecs = l2_normalize(func_vecs)
    sim = func_vecs @ func_vecs.T
    np.fill_diagonal(sim, 1.0)
    return sim


def fake_vector(key, dim):
    seed = zlib.crc32(key.encode('utf-8'))
    rng = np.random.default_rng(seed)
    return rng.normal(size=dim).astype(np.float32)


def get_hf_token():
    token_file = os.path.join(BASE_DIR, '.hf_token')
    if os.path.exists(token_file):
        with open(token_file, encoding='utf-8') as f:
            tok = f.read().strip()
            if tok:
                return tok
    return os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN') or None


def load_model(spec, device):
    import torch
    from transformers import AutoModel, AutoTokenizer

    token = get_hf_token()
    auth_kwargs = {'token': token} if token else {}
    if token:
        print('  [info] 使用本地 HF token (ARR_Oct/.hf_token 或环境变量)')

    if spec.get('st_encode'):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                f'{spec["repo"]} 的官方句向量管线 (mean pooling + Dense 投影 + L2) '
                f'需 sentence-transformers，请 pip install -U sentence-transformers') from e
        st = SentenceTransformer(
            spec['repo'], device=device,
            model_kwargs={'torch_dtype': torch.float32},
            token=token)
        model, tokenizer = st[0].auto_model, st[0].tokenizer
        return model, tokenizer, st

    dtype = torch.float32
    if spec.get('dtype') == 'float32':
        dtype = torch.float32
    elif device == 'cuda':
        dtype = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(
        spec['repo'], use_fast=True,
        padding_side='left' if spec.get('left_padding') else 'right',
        **auth_kwargs)
    model = AutoModel.from_pretrained(spec['repo'], torch_dtype=dtype, **auth_kwargs)
    model.to(device).eval()
    return model, tokenizer, None


def encode_sentences(model, tokenizer, st, spec, texts, device):
    import torch

    if spec.get('st_encode') and st is not None:
        try:
            emb = st.encode_document(list(texts), batch_size=ENCODE_BATCH_SIZE,
                                     show_progress_bar=False, convert_to_numpy=True)
        except Exception as e:
            print(f'  [WARN] encode_document 不可用 ({type(e).__name__}: {e})，'
                  f'回退手动前缀 + encode')
            emb = None
        if emb is None:
            emb = st.encode([spec['prompt'].format(t) for t in texts],
                            batch_size=ENCODE_BATCH_SIZE,
                            show_progress_bar=False, convert_to_numpy=True)
        return np.asarray(emb, dtype=np.float32)

    if spec['pooling'] == 'last_token':
        tokenizer.padding_side = 'left'
    else:
        tokenizer.padding_side = 'right'

    n_trunc = 0
    vecs = []
    for b in range(0, len(texts), ENCODE_BATCH_SIZE):
        batch_texts = texts[b:b + ENCODE_BATCH_SIZE]
        if spec.get('prompt'):
            batch_texts = [spec['prompt'].format(t) for t in batch_texts]
        for t in batch_texts:
            n_trunc += len(tokenizer(t, add_special_tokens=False)['input_ids']) + 2 > spec['max_length']
        enc = tokenizer(batch_texts, padding=True, truncation=True,
                        max_length=spec['max_length'], return_tensors='pt')
        enc.pop('token_type_ids', None)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            hidden = model(**enc).last_hidden_state
        if spec['pooling'] == 'cls':
            emb = hidden[:, 0]
        elif spec['pooling'] == 'mean':
            m = enc['attention_mask'].unsqueeze(-1).to(hidden.dtype)
            emb = (hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-9)
        else:
            emb = hidden[:, -1]
        vecs.append(emb.float().cpu().numpy())
    if n_trunc:
        print(f'  [WARN] {n_trunc}/{len(texts)} 条文本超过 max_length={spec["max_length"]}，已截断')
    return np.concatenate(vecs, axis=0)


def compute_target_vectors(model, tokenizer, spec, examples, device):
    import torch

    tokenizer.padding_side = 'right'
    vecs = []
    for b in range(0, len(examples), TARGET_BATCH_SIZE):
        batch = examples[b:b + TARGET_BATCH_SIZE]
        enc = tokenizer([e['句子'] for e in batch], padding=True, truncation=True,
                        max_length=TARGET_MAX_LENGTH, return_offsets_mapping=True,
                        return_tensors='pt')
        offsets = enc.pop('offset_mapping')
        enc.pop('token_type_ids', None)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            hidden = model(**enc).last_hidden_state
        for i, e in enumerate(batch):
            start, end = e['开头索引'], e['末尾索引']
            offs = offsets[i].tolist()
            sel = [k for k, (a, z) in enumerate(offs) if z > start and a < end and z > a]
            if not sel:
                mid = (start + end) / 2.0
                sel = [min((k for k, (a, z) in enumerate(offs) if z > a),
                           key=lambda k: abs((offs[k][0] + offs[k][1]) / 2.0 - mid))]
                print(f"[WARN] 目标词无覆盖token，改用最近token [{e['func']}] {e['句子']}")
            vecs.append(hidden[i, sel].float().mean(dim=0).cpu().numpy())
    return np.stack(vecs)


def save_heatmap(sim, func_names, matrix_name, model_name, out_dir):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
        from matplotlib.font_manager import FontProperties, fontManager

        for font_path in ('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
                          '/usr/share/fonts/wqy-microhei/wqy-microhei.ttc',
                          '/System/Library/Fonts/PingFang.ttc',
                          '/System/Library/Fonts/STHeiti Light.ttc'):
            if os.path.exists(font_path):
                font = FontProperties(fname=font_path)
                fontManager.addfont(font_path)
                plt.rcParams['font.family'] = font.get_name()
                break
        plt.rcParams['axes.unicode_minus'] = False

        title = f'功能间相似性矩阵 - {matrix_name} ({model_name})'
        if DRY_RUN:
            title += ' [DRY-RUN]'
        out_png = os.path.join(out_dir, f'function_similarity_{matrix_name}_heatmap.png')

        plt.figure(figsize=(14, 12))
        sns.heatmap(sim, xticklabels=func_names, yticklabels=func_names,
                    annot=True, fmt='.2f', cmap='RdYlBu_r', vmin=0, vmax=1,
                    square=True, linewidths=0.5, linecolor='gray',
                    cbar_kws={'label': 'Cosine Similarity'})
        plt.title(title, fontsize=16)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(out_png, dpi=150)
        plt.close()
        print(f'  热力图已保存: {out_png}')
    except Exception as e:
        print(f'  [WARN] 热力图生成失败 ({matrix_name}): {e}')


def setup_case(case):
    """准备单个 case：读取 contexts、GT(Sheet2)、SMM(Sheet1)，展平单独例句。"""
    prompts_file, gt_file, sim_dir = case_paths(case)
    os.makedirs(sim_dir, exist_ok=True)

    # 1. 读取 case{n}.json (四种 context 类型)
    with open(prompts_file, encoding='utf-8') as f:
        contexts = json.load(f)
    func_names = list(contexts['流畅综合'].keys())
    print(f'\n{"=" * 70}')
    print(f'=== case{case}: 共 {len(func_names)} 个功能 ===')
    print(f'{"=" * 70}')
    for ctype in TEXT_CONTEXT_TYPES + (TOKEN_CONTEXT_TYPE,):
        if ctype not in contexts:
            raise KeyError(f'context文件缺少类型: {ctype} (现有: {list(contexts.keys())})')
        print(f'  {ctype}: {len(contexts[ctype])} 个功能')

    # 2. 读取 GT (Sheet2: 直接功能邻接矩阵 0/1)
    df_gt = pd.read_excel(gt_file, sheet_name='Sheet2', header=0)
    df_gt = df_gt.rename(columns={df_gt.columns[0]: 'func'})
    df_gt = df_gt.set_index('func')
    df_gt = df_gt.fillna(0).astype(float)
    gt_names = list(df_gt.index)
    gt_out = os.path.join(sim_dir, 'gt_matrix.csv')
    df_gt.to_csv(gt_out, encoding='utf-8-sig', float_format='%.6f')
    print(f'\nGT矩阵 (Sheet2): {df_gt.shape[0]}x{df_gt.shape[1]}, 已保存: {gt_out}')

    # 3. SMM G/J/D (Sheet1 form-feature矩阵；type 列仅 case2 有，其余全部为数据行)
    df_supp = pd.read_excel(gt_file, sheet_name='Sheet1', header=0)
    header = [str(c).strip() for c in df_supp.columns]
    k = 0
    while k < len(header) and header[k] in LEADING_LABEL_COLS:
        k += 1
    label_cols = list(df_supp.columns[:k])
    func_cols = list(df_supp.columns[k:])
    if not func_cols:
        raise ValueError('Sheet1 未识别到功能列')
    if 'type' in df_supp.columns:
        data_mask = df_supp['type'].astype(str).str.lower() == 'data'
    else:
        data_mask = pd.Series(True, index=df_supp.index)
    df_data = df_supp[data_mask].copy()
    tfM = df_data[func_cols].apply(pd.to_numeric, errors='coerce').fillna(0).astype(int).values.astype(float)
    form_names = []
    for _, r in df_data.iterrows():
        parts = [str(r[c]) for c in label_cols if c != 'type']
        parts = [p for p in parts if p and p != 'nan']
        form_names.append('-'.join(parts))

    smm_dfs = {}
    for ct in ('G', 'J', 'D'):
        try:
            sm = SemanticMap(tfM=tfM.copy(), featNames=list(func_cols),
                             formNames=form_names, calc_type=ct, blend=None)
            adj = sm.calculate_semantic_relations(sm.tfM, calc_type=ct)
            if adj.shape[0] < len(func_cols):
                adj = sm.get_unmerged_matrix(adj)
            adj = sm.norm_matrix(adj)
            smm_dfs[ct] = pd.DataFrame(adj, index=func_cols, columns=func_cols)
            smm_out = os.path.join(sim_dir, f'smm_matrix_{ct}.csv')
            smm_dfs[ct].to_csv(smm_out, encoding='utf-8-sig', float_format='%.6f')
            print(f'SMM矩阵({ct})已保存: {smm_out}')
        except ValueError as e:
            print(f'[WARN] SMM {ct} 不可用，已跳过: {e}')

    common = [f for f in gt_names if f in func_names]
    n = len(common)
    triu_idx = np.triu_indices(n, k=1)
    gt_vals = df_gt.loc[common, common].values[triu_idx]
    smm_vals = {ct: smm_dfs[ct].loc[common, common].values[triu_idx] for ct in smm_dfs}

    # 4. 单独例句扁平化 (所有模型共用；无目标词索引的条目跳过)
    flat = []
    func_indices = {}
    skipped = {}
    for func, items in contexts[TOKEN_CONTEXT_TYPE].items():
        func_indices[func] = []
        for it in items:
            if it.get('开头索引', -1) == -1:
                skipped[func] = skipped.get(func, 0) + 1
                continue
            func_indices[func].append(len(flat))
            flat.append({'func': func, **it})
    for func, cnt in skipped.items():
        print(f'[WARN] case{case} [{func}] {cnt} 条例句未定位目标词，已跳过')
    print(f'\ncase{case} 单独例句共 {len(flat)} 条可用')
    if not flat:
        print(f'[WARN] case{case} 无已定位目标词的例句，'
              f'{TOKEN_MATRIX_MEAN} 与 {TOKEN_MATRIX_PAIRWISE} 将缺省')

    return {
        'case': case, 'sim_dir': sim_dir, 'contexts': contexts, 'func_names': func_names,
        'common': common, 'triu_idx': triu_idx, 'gt_vals': gt_vals, 'smm_vals': smm_vals,
        'flat': flat, 'func_indices': func_indices, 'has_targets': bool(flat),
    }


def run_model_on_case(cd, spec, device, model=None, tokenizer=None, st=None):
    """在单个 case 上用已加载的模型计算全部 context 类型的相似性矩阵与相关性。"""
    case, sim_dir = cd['case'], cd['sim_dir']
    contexts, func_names = cd['contexts'], cd['func_names']
    common, triu_idx = cd['common'], cd['triu_idx']
    model_name = spec['name']
    model_label = f'{model_name}_center' if CENTER else model_name
    model_dir = os.path.join(sim_dir, model_label)
    os.makedirs(model_dir, exist_ok=True)

    sim_matrices = {}
    for ctype in TEXT_CONTEXT_TYPES:
        texts = [str(contexts[ctype].get(f, '')) for f in func_names]
        print(f'  编码 {ctype}: 文本长度 min={min(map(len, texts))} max={max(map(len, texts))}')
        if DRY_RUN:
            func_vecs = np.stack([fake_vector(f'{model_name}|{ctype}|{f}', spec['dim'])
                                   for f in func_names])
        else:
            func_vecs = encode_sentences(model, tokenizer, st, spec, texts, device)
        sim_matrices[ctype] = cosine_matrix(func_vecs, center=CENTER)

    # 单独例句两类矩阵：仅当存在已定位目标词的例句时计算，否则缺省
    if cd['has_targets']:
        flat, func_indices = cd['flat'], cd['func_indices']
        if DRY_RUN:
            tok_vecs = np.stack([
                fake_vector(f"{model_name}|tok|{e['func']}|{e['句子']}|{e['开头索引']}", spec['dim'])
                for e in flat])
        else:
            tok_vecs = compute_target_vectors(model, tokenizer, spec, flat, device)
        print(f'  token向量: {tok_vecs.shape}')

        normed = l2_normalize(tok_vecs)
        if CENTER:
            # token 矩阵同样去公共方向：对全体例句 token 向量去均值后再归一化
            normed = l2_normalize(center_vecs(normed))

        mean_vecs = np.zeros((len(func_names), normed.shape[1]), dtype=np.float32)
        for i, f in enumerate(func_names):
            ii = func_indices.get(f, [])
            if ii:
                mean_vecs[i] = normed[ii].mean(axis=0)
            else:
                print(f'  [WARN] 功能 {f} 无有效例句，向量置零')
        sim_matrices[TOKEN_MATRIX_MEAN] = cosine_matrix(mean_vecs)

        cross = normed @ normed.T
        sim_pair = np.zeros((len(func_names), len(func_names)), dtype=np.float32)
        for i, fi in enumerate(func_names):
            for j, fj in enumerate(func_names):
                ii, jj = func_indices.get(fi, []), func_indices.get(fj, [])
                if ii and jj:
                    sim_pair[i, j] = cross[np.ix_(ii, jj)].mean()
        np.fill_diagonal(sim_pair, 1.0)
        sim_matrices[TOKEN_MATRIX_PAIRWISE] = sim_pair
    else:
        print(f'  [WARN] case{case} 缺少目标词索引，跳过 {TOKEN_MATRIX_MEAN} 与 {TOKEN_MATRIX_PAIRWISE}')

    model_results = []
    for matrix_name, sim in sim_matrices.items():
        df_sim = pd.DataFrame(sim, index=func_names, columns=func_names)
        csv_path = os.path.join(model_dir, f'function_similarity_matrix_{matrix_name}.csv')
        df_sim.to_csv(csv_path, encoding='utf-8-sig', float_format='%.6f')

        bge_vals = df_sim.loc[common, common].values[triu_idx]
        r_p, p_p = pearsonr(cd['gt_vals'], bge_vals)
        r_s, p_s = spearmanr(cd['gt_vals'], bge_vals)
        print(f'  [{matrix_name}] GT: Pearson r={r_p:.4f} (p={p_p:.2e}), '
              f'Spearman ρ={r_s:.4f} (p={p_s:.2e})')
        model_results.append({'case': f'case{case}', 'model': model_label, 'context_type': matrix_name,
                              'calc_type': 'GT', 'pearson_r': r_p, 'pearson_p': p_p,
                              'spearman_rho': r_s, 'spearman_p': p_s})
        for ct, smm_val in cd['smm_vals'].items():
            r_p2, p_p2 = pearsonr(smm_val, bge_vals)
            r_s2, p_s2 = spearmanr(smm_val, bge_vals)
            print(f'    SMM-{ct}: Pearson r={r_p2:.4f} (p={p_p2:.2e}), '
                  f'Spearman ρ={r_s2:.4f} (p={p_s2:.2e})')
            model_results.append({'case': f'case{case}', 'model': model_label, 'context_type': matrix_name,
                                  'calc_type': ct, 'pearson_r': r_p2, 'pearson_p': p_p2,
                                  'spearman_rho': r_s2, 'spearman_p': p_s2})
        save_heatmap(sim, func_names, matrix_name, model_label, model_dir)

    # 每个模型目录单独保存一份相关性结果，便于独立查阅
    pd.DataFrame(model_results).to_csv(
        os.path.join(model_dir, 'correlation_results.csv'),
        encoding='utf-8-sig', float_format='%.6f', index=False)
    return model_results


def main():
    if DRY_RUN:
        print('=' * 70)
        print('[DRY-RUN] 跳过模型加载，使用确定性随机向量代替编码结果')
        print('[DRY-RUN] 相似度与相关系数无实际意义，仅用于本地流程验证')
        print('[DRY-RUN] 实际运行请在 GPU 服务器上不带 --dry-run 参数执行')
        print('[DRY-RUN] 用法: python compute_similarity_whole.py [--dry-run] '
              '[--center] [--models bge-m3,Qwen3-Embedding-0.6B,...] [--cases 1,2,3]')
        print('=' * 70)

    if CENTER:
        print('=' * 70)
        print('[CENTER] 已启用向量去均值 (all-but-the-top)：计算余弦前减去向量的公共均值方向，')
        print('[CENTER] 用于缓解未微调 MLM (xlm-roberta 等) 的各向异性/表征退化问题')
        print('[CENTER] 结果写入 {模型}_center/ 目录与 correlation_results_center.csv，不覆盖原始结果')
        print('=' * 70)

    specs = MODEL_SPECS
    if SELECTED_MODELS is not None:
        known = {s['name'] for s in MODEL_SPECS}
        unknown = [m for m in SELECTED_MODELS if m not in known]
        if unknown:
            sys.exit(f'未知模型: {unknown}，可选: {sorted(known)}')
        specs = [s for s in MODEL_SPECS if s['name'] in SELECTED_MODELS]
    print(f'待处理模型 ({len(specs)}): {[s["name"] for s in specs]}')
    print(f'待处理case ({len(CASES)}): {CASES}')

    # 逐 case 准备（contexts/GT/SMM/例句扁平化，与模型无关，只做一次）
    case_data = {}
    for case in CASES:
        prompts_file, gt_file, _ = case_paths(case)
        for p in (prompts_file, gt_file):
            if not os.path.exists(p):
                print(f'[WARN] case{case} 缺少文件，跳过: {p}')
                break
        else:
            try:
                case_data[case] = setup_case(case)
            except Exception:
                print(f'[WARN] case{case} 准备失败，已跳过:')
                print(traceback.format_exc())
    if not case_data:
        sys.exit('无可用 case，退出')

    device = None
    if not DRY_RUN:
        import torch
        import transformers
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'\ndevice={device}, transformers={transformers.__version__}')

    # 模型为外层循环（只加载一次），case 为内层循环
    results_by_case = {case: [] for case in case_data}
    for spec in specs:
        model_name = spec['name']
        print(f"\n{'=' * 70}")
        print(f'=== 模型: {spec["repo"]} (pooling={spec["pooling"]}, '
              f'max_length={spec["max_length"]}) ===')
        if spec.get('note'):
            print(f'  [note] {spec["note"]}')
        print(f"{'=' * 70}")

        model = tokenizer = st = None
        try:
            if not DRY_RUN:
                model, tokenizer, st = load_model(spec, device)
        except Exception:
            print(f'  [WARN] 模型 {model_name} 加载失败，已跳过:')
            print(traceback.format_exc())
            continue

        try:
            for case, cd in case_data.items():
                print(f'\n--- case{case} / {model_name} ---')
                try:
                    results_by_case[case].extend(
                        run_model_on_case(cd, spec, device, model, tokenizer, st))
                except Exception:
                    print(f'  [WARN] case{case} 处理失败 (模型 {model_name})，已跳过:')
                    print(traceback.format_exc())
        finally:
            if not DRY_RUN:
                del model, tokenizer, st
                gc.collect()
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(f'  模型 {model_name} 完成，显存已释放')

    # 逐 case 汇总输出
    for case, res in results_by_case.items():
        sim_dir = case_data[case]['sim_dir']
        df_results = pd.DataFrame(res)
        results_csv = os.path.join(
            sim_dir, 'correlation_results_center.csv' if CENTER else 'correlation_results.csv')
        df_results.to_csv(results_csv, encoding='utf-8-sig', float_format='%.6f', index=False)
        print(f'\ncase{case} 相关性结果已保存: {results_csv} ({len(df_results)} 行)')
        try:
            gt_rows = df_results[df_results['calc_type'] == 'GT']
            pivot = gt_rows.pivot_table(index='model', columns='context_type', values='pearson_r')
            print(f'case{case} GT Pearson r 汇总 (模型 x context类型):')
            with pd.option_context('display.max_columns', None, 'display.width', 200):
                print(pivot.round(4).to_string())
        except Exception as e:
            print(f'[WARN] case{case} 汇总表生成失败: {e}')


if __name__ == '__main__':
    main()
