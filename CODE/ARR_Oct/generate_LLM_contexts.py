import pandas as pd
import json
import os
import re
import unicodedata

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data_pure')
OUTPUT_DIR = os.path.join(BASE_DIR, 'LLM_contexts')

# case 1-12 均已提供例句；case 5/6/8/12 另带目标词（见 CASE_CONFIGS）
CASES = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)

CONTEXT_TYPES = ('流畅综合', '单独例句', '单独定义', '简单综合')

# Sheet1 前导标签列（其后才是功能列）：case2 多一个 type，case8 多一个 语组
LEADING_LABEL_COLS = {'languages', '语组', 'forms', 'type'}

# 每个case的概念描述（沿用此前 data/LLM_examples/caseN/generate_prompts.py 的表述）
#   concept      概念总述；example_word 概念/功能；extra 额外说明（case1）
#   default_targets 无显式目标词时依次尝试的默认目标词；
#                  'zh_name' 表示用 Sheet3 中文名称作目标词（case3）
#   fluent_template 自定义流畅综合模板（case3 为动词+双及物结构的特殊表述）
#   line_format  例句中目标词的位置：
#                  'colon'      "目标词：句子"（默认，case2/7/8）
#                  'target_lang' "目标词：语言名：句子"（case6，语言名段会被剥掉）
#                  'lang_target' "语言名 目标词: 句子"（case12，先剥掉行首语言名）
#   targets_from_row 目标词取自 Sheet3 的 Target word 行，与例句逐行对应（case5）
CASE_CONFIGS = {
    1: {
        'concept': '一种跨汉语方言共通的概念，核心概念表示"吃"的动作或行为',
        'extra': '在汉语的不同方言中，这个概念会有不同的词来表达，例如："吃、喫、食、馌"等。',
        'default_targets': ['吃', '喫', '食', '馌'],
    },
    2: {
        'concept': '表示补充义、重复义等的多功能语法词的跨语言及方言共通的概念',
        'default_targets': None,  # 例句自带 "序号、目标词：句子" 格式
    },
    3: {
        'fluent_template': (
            '动词"{func}"在汉语中被称为"{zh_name}"，'
            '它之后往往可以带一个双宾语构成双及物结构，'
            '形成"施事，受事，与事"的三重结构，'
            '但是具体各个语义角色的顺序取决于该动词的语义等特征。'
            '这一动词的语义解释为"{explanation}"，'
            '并且有下面的例句：{examples}。'
        ),
        'default_targets': 'zh_name',
    },
    4: {
        'concept': '汉语多功能语法词"得"的跨汉语方言共通的概念',
        'default_targets': ['得'],
    },
    5: {
        'concept': '汉语多功能语法词"和"的跨语言及方言共通的概念',
        # 目标词来自 Sheet3 的 Target word 行，与例句逐行对应
        'targets_from_row': True,
        'default_targets': None,
    },
    6: {
        'concept': '汉语多功能疑问词"什么"的跨语言及方言共通的概念',
        'example_word': '该功能',
        # 例句形如 "目标词：语言名：句子"
        'line_format': 'target_lang',
        'default_targets': None,
    },
    7: {
        'concept': '汉语表示"终竟义"的多功能语法词的跨语言及方言共通的概念',
        'default_targets': None,  # 例句多为 "目标词：句子" 格式
    },
    8: {
        'concept': '汉语量词的跨语言及方言共通的功能',
        'example_word': '该功能',
        'default_targets': None,  # 例句形如 "目标词：句子"，无目标词的行索引记 -1
    },
    9: {
        'concept': '汉语泛义动词"做"的跨语言及方言共通的功能',
        'example_word': '该功能',
        'default_targets': ['做'],
    },
    10: {
        'concept': '汉语泛义动词"打"的跨语言及方言共通的功能',
        'example_word': '该功能',
        'default_targets': ['打'],
    },
    11: {
        'concept': '汉语泛义动词"来"的跨语言及方言共通的功能',
        'example_word': '该功能',
        'default_targets': ['来'],
    },
    12: {
        'concept': '与"树、木材、柴火、树林、森林"相关的跨语言及方言共通概念',
        # 例句形如 "Danish træ: Et godt træ bærer god frugt."
        'line_format': 'lang_target',
        'default_targets': None,
    },
}

DEFAULT_FLUENT_TEMPLATE = '"{func}"是{concept}，{meaning_clause}{extra}以下是体现{example_word}的一些例句：{examples}。'


def lower_safe(s):
    out = []
    for ch in s:
        low = ch.lower()
        out.append(low if len(low) == 1 else ch)
    return ''.join(out)


def fold_graphemes(s):
    folded = []
    starts = []
    ends = []
    i = 0
    while i < len(s):
        j = i + 1
        while j < len(s) and unicodedata.combining(s[j]):
            j += 1
        for ch in unicodedata.normalize('NFC', s[i:j]):
            folded.append(ch)
            starts.append(i)
            ends.append(j)
        i = j
    return ''.join(folded), starts, ends


def norm_form(s):
    return lower_safe(fold_graphemes(s)[0])


ALT_SPLIT_RE = re.compile(r'[/-]+|\s+and\s+', re.IGNORECASE)


def target_alternatives(t):
    """目标词标签若并列多个变体（'Apa/sesuatu'、'孰-之'、'bois and forêt'），
    返回标签本身及按 /、-、and 拆出的各变体。"""
    out, seen = [], set()
    for a in [t] + ALT_SPLIT_RE.split(t):
        a = a.strip()
        if a and a not in seen:
            seen.add(a)
            out.append(a)
    return out


def locate_target(target, sentence):
    if not target:
        return -1, -1

    candidates = target_alternatives(target)
    stripped = re.sub(r'[()（）]', '', target)
    if stripped and stripped not in candidates:
        candidates.append(stripped)

    for cand in candidates:
        idx = sentence.find(cand)
        if idx != -1:
            return idx, idx + len(cand)

    folded_sent, starts, ends = fold_graphemes(sentence)
    lowered_sent = lower_safe(folded_sent)
    folded_cands = [fold_graphemes(c)[0] for c in candidates]
    lowered_cands = [lower_safe(c) for c in folded_cands]

    for fc, lc in zip(folded_cands, lowered_cands):
        for hay, needle in ((folded_sent, fc), (lowered_sent, lc)):
            idx = hay.find(needle)
            if idx != -1:
                return starts[idx], ends[idx + len(needle) - 1]

    for length in range(max(len(c) for c in folded_cands), 0, -1):
        for fc, lc in zip(folded_cands, lowered_cands):
            if length > len(fc):
                continue
            for hay, needle in ((folded_sent, fc[:length]), (lowered_sent, lc[:length])):
                idx = hay.find(needle)
                if idx != -1:
                    return starts[idx], ends[idx + length - 1]

    return -1, -1


NUM_RE = re.compile(r'^\s*\d+\s*[、.．]\s*')
PIPE_RE = re.compile(r'[｜|]')
MAX_TARGET_LABEL_LEN = 30


def cell_str(v):
    s = str(v).strip() if pd.notna(v) else ''
    return '' if s == 'nan' else s


def split_example_lines_indexed(cell_text):
    """同 split_example_lines，但附带该例句所在原始换行序号（0 起），
    供与 Sheet3 的 Target word 行逐行对齐（case5）。"""
    lines = []
    for idx, raw in enumerate(cell_str(cell_text).split('\n')):
        raw = raw.strip()
        if not raw:
            continue
        for part in PIPE_RE.split(raw):
            part = part.strip()
            if part:
                lines.append((idx, part))
    return lines


def split_example_lines(cell_text):
    """按换行切分单元格外，再把一行内用 ｜ / | 分隔的多个例句拆开（如 case10/11）。
    返回保留原始编号的行。"""
    return [part for _, part in split_example_lines_indexed(cell_text)]


def strip_numbering(line):
    # 顺带去掉剥掉编号后残留在行首的分隔符（如 case8 的 "11、：tu²tu² din¹"）
    return re.sub(r'^[：:]+', '', NUM_RE.sub('', line)).strip()


def try_explicit(body):
    """识别 "目标词：句子" 格式（如 case2/case7），返回 (目标词, 句子) 或 None。
    分隔符前的内容须能在分隔符后的句子中定位到，否则视为方言名等标签而非目标词。"""
    for sep in ('：', ':'):
        i = body.find(sep)
        if i <= 0:
            continue
        cand = body[:i].strip()
        rest = body[i + 1:].strip()
        if 0 < len(cand) <= MAX_TARGET_LABEL_LEN and rest and locate_target(cand, rest)[0] != -1:
            return cand, rest
        break
    return None


def strip_lang_label(target, rest):
    """case6 形如 "目标词：语言名：句子"，剥掉不含目标词的行首标签（语言名）。
    仅当剥掉后的片段仍能定位到目标词时才生效，最多剥 3 段。"""
    sentence = rest
    for _ in range(3):
        seps = [j for j in (sentence.find('：'), sentence.find(':')) if j > 0]
        if not seps:
            break
        j = min(seps)
        label, tail = sentence[:j], sentence[j + 1:].strip()
        if not tail or locate_target(target, label)[0] != -1 or locate_target(target, tail)[0] == -1:
            break
        sentence = tail
    return sentence


def parse_lang_target(body):
    """识别 "语言名 目标词: 句子" 格式（case12，如 "Danish træ: Et godt træ…"），
    返回 (目标词, 句子) 或 None。行首语言名（ASCII 字母开头的词）先剥掉再作目标词；
    无语言名时按整段标签处理，与 try_explicit 一致。"""
    for sep in ('：', ':'):
        i = body.find(sep)
        if i == -1:
            continue
        label, sentence = body[:i].strip(), body[i + 1:].strip()
        if not label or not sentence:
            return None
        parts = label.split(None, 1)
        if len(parts) == 2 and parts[0].isascii() and parts[0].isalpha():
            target = parts[1].strip()
            if 0 < len(target) <= MAX_TARGET_LABEL_LEN and locate_target(target, sentence)[0] != -1:
                return target, sentence
        if 0 < len(label) <= MAX_TARGET_LABEL_LEN and locate_target(label, sentence)[0] != -1:
            return label, sentence
        return None
    return None


def ends_mid_word(text, end):
    """text[end] 处若为字母（Lo 之外的类别，即拉丁/西里尔等字母），说明匹配停在词中间；
    CJK 等 Lo 类字符不算，因其本身即为词。"""
    return end < len(text) and unicodedata.category(text[end]) in ('Lu', 'Ll', 'Lt', 'Lm')


def match_kind(body, target, require_boundary):
    """在 body 中匹配 target，返回 'full' / 'partial' / None（视为无匹配）。
    完整匹配指命中 target 本身或其任一并列变体；
    部分匹配须覆盖该变体一半以上且不在词中间；
    require_boundary=True 时完整匹配也要落在词边界，用于跨列候选（证据较弱）。"""
    found = None
    for alt in target_alternatives(target):
        start, end = locate_target(alt, body)
        if start == -1:
            continue
        matched = body[start:end]
        if norm_form(matched) in (norm_form(alt), norm_form(re.sub(r'[()（）]', '', alt))):
            if require_boundary and ends_mid_word(body, end):
                continue
            return 'full'
        if len(matched.strip()) * 2 >= len(alt) and not ends_mid_word(body, end):
            found = found or 'partial'
    return found


def is_substantial(sentence, target):
    """标签是否在句中完整出现（含并列变体），而非只匹配到很短的前缀片段。"""
    return match_kind(sentence, target, False) == 'full'


def pick_fallback(body, own_candidates, cross_candidates):
    """从候选目标词中选一个：先看完整匹配（本列候选优先于跨列候选，同层内更长者优先），
    找不到再看部分匹配。跨列候选（整个 case 其它列的目标词）证据较弱，须落在词边界上。"""
    for want in ('full', 'partial'):
        for pool, require_boundary in ((own_candidates, False), (cross_candidates, True)):
            best = None
            for t in pool:
                if not t or match_kind(body, t, require_boundary) != want:
                    continue
                if best is None or (want == 'full' and len(t) > len(best)):
                    best = t
            if best is not None:
                return best
    return None


def sheet1_functions(df1):
    start = 0
    while start < df1.shape[1] and cell_str(df1.iloc[0, start]) in LEADING_LABEL_COLS:
        start += 1
    funcs = []
    for j in range(start, df1.shape[1]):
        name = cell_str(df1.iloc[0, j])
        if name:
            funcs.append(name)
    return funcs


def sheet3_layout(df3):
    """按首列标签定位各行；例句行取解释行紧挨着的下一行（case1 即 例句（论文））。"""
    rows = {}
    for i in range(df3.shape[0]):
        label = cell_str(df3.iloc[i, 0])
        if label and label not in rows:
            rows[label] = i
    expl_row = rows.get('解释')
    if expl_row is None:
        raise ValueError('Sheet3 缺少 解释 行')
    ex_row = expl_row + 1
    if cell_str(df3.iloc[ex_row, 0]) in ('英文名称', '中文名称', '解释'):
        raise ValueError(f'Sheet3 解释 行({expl_row})之后不是例句行')
    return {
        'en_row': rows.get('英文名称'),
        'zh_row': rows.get('中文名称'),
        'expl_row': expl_row,
        'ex_row': ex_row,
        'fluent_row': rows.get('综合'),  # 仅 case2 有：流畅综合使用的论文综合例句
        'target_row': rows.get('Target word'),  # case5 有：与例句行逐行对应的目标词
    }


def process_case(n, cfg):
    input_file = os.path.join(DATA_DIR, f'case{n}.xlsx')
    output_file = os.path.join(OUTPUT_DIR, f'case{n}.json')

    df1 = pd.read_excel(input_file, sheet_name='Sheet1', header=None)
    functions = sheet1_functions(df1)

    df3 = pd.read_excel(input_file, sheet_name='Sheet3', header=None)
    layout = sheet3_layout(df3)
    n_cols = df3.shape[1]

    def row_values(row):
        return [cell_str(df3.iloc[row, j]) for j in range(n_cols)]

    en_names = row_values(layout['en_row']) if layout['en_row'] is not None else [''] * n_cols
    zh_names = row_values(layout['zh_row']) if layout['zh_row'] is not None else [''] * n_cols
    explanations = row_values(layout['expl_row'])
    example_raw = row_values(layout['ex_row'])
    fluent_raw = row_values(layout['fluent_row']) if layout['fluent_row'] is not None else [''] * n_cols

    # 英文名称/中文名称 均可索引到 Sheet3 列（case1/3 的 Sheet1 功能名是英文，其余是中文）
    name_to_col = {}
    for c in range(1, n_cols):
        for name in (en_names[c], zh_names[c]):
            if name and name not in name_to_col:
                name_to_col[name] = c

    default_targets = cfg.get('default_targets')
    line_format = cfg.get('line_format', 'colon')
    from_row = bool(cfg.get('targets_from_row'))
    concept = cfg.get('concept', '')
    extra = cfg.get('extra', '')
    example_word = cfg.get('example_word', '该概念')
    fluent_template = cfg.get('fluent_template', DEFAULT_FLUENT_TEMPLATE)

    # case5：Sheet3 的 Target word 行与例句行逐行对应
    if from_row:
        if layout['target_row'] is None:
            raise ValueError(f'case{n}: 配置了 targets_from_row 但 Sheet3 缺少 Target word 行')
        target_raw = row_values(layout['target_row'])
    else:
        target_raw = [''] * n_cols

    contexts = {context_type: {} for context_type in CONTEXT_TYPES}
    stats = {'missing': 0, 'partial': 0, 'fallback': 0, 'row_miss': 0}

    # 第一遍：逐条识别目标词，并收集各功能及整个case出现过的目标词
    func_lines = {}   # func -> [(body, explicit or None), ...]
    func_provided = {}  # func -> 显式给出但未能定位的目标词，仅作本列候选
    case_target_freq = {}

    def parse_explicit(func, raw_index, body):
        """识别一行中的 (目标词, 句子)，识别不到返回 None。"""
        if from_row:
            provided = func_provided[func]
            target = provided[raw_index] if raw_index < len(provided) else ''
            if not target:
                return None
            if locate_target(target, body)[0] != -1:
                return target, body
            stats['row_miss'] += 1
            print(f'[INFO] Target word 未能定位 [{func}] "{target}" 句子="{body}"，改用候选目标词')
            return None
        if line_format == 'lang_target':
            hit = parse_lang_target(body)
        else:
            hit = try_explicit(body)
            if hit and line_format == 'target_lang':
                hit = hit[0], strip_lang_label(hit[0], hit[1])
        if hit is None:
            return None
        target, sentence = hit
        if line_format != 'colon' and not is_substantial(sentence, target):
            # 标签只匹配到句中很短的片段：该行其实没有该目标词（目标词多出现在相邻行），
            # 这里仅把标签留作本列候选，供相邻行使用
            func_provided[func].append(target)
            print(f'[INFO] 标签仅部分匹配 [{func}] "{target}" 句子="{body}"，只留作候选')
            return None
        return target, sentence

    for func in functions:
        col = name_to_col.get(func)
        if col is None:
            print(f'[WARN] case{n}: No Sheet3 entry for: {func}')
            continue
        func_provided[func] = [t.strip() for t in target_raw[col].split('\n')] if from_row else []
        lines = []
        for raw_index, line in split_example_lines_indexed(example_raw[col]):
            body = strip_numbering(line)
            if body:
                lines.append((body, parse_explicit(func, raw_index, body)))
        func_lines[func] = lines
        for _, hit in lines:
            if hit:
                case_target_freq[hit[0]] = case_target_freq.get(hit[0], 0) + 1
    case_targets = [t for t, _ in sorted(case_target_freq.items(), key=lambda kv: -kv[1])]

    for func in functions:
        if func not in func_lines:
            continue
        col = name_to_col.get(func)
        explanation = explanations[col]
        zh_name = zh_names[col] or func

        # 流畅综合的例句来源：有 综合 行（case2）用之，否则用例句行原文（保留编号）
        if layout['fluent_row'] is not None and fluent_raw[col]:
            fluent_example_lines = split_example_lines(fluent_raw[col])
        else:
            fluent_example_lines = split_example_lines(example_raw[col])

        # 单独例句：优先行内显式目标词（目标词行 / "目标词：句子" 等格式），其余依次尝试候选目标词
        example_list = []
        sentences = []
        func_targets = [hit[0] for _, hit in func_lines[func] if hit]
        base_targets = list(default_targets) if isinstance(default_targets, list) else []
        if default_targets == 'zh_name':
            base_targets.append(zh_name)
        own_targets = []
        cross_targets = []
        for t in base_targets + func_provided.get(func, []) + func_targets:
            if t and t not in own_targets:
                own_targets.append(t)
        for t in case_targets:
            if t and t not in own_targets and t not in cross_targets:
                cross_targets.append(t)
        for body, hit in func_lines[func]:
            if hit:
                target, sentence = hit
                start, end = locate_target(target, sentence)
                stripped = re.sub(r'[()（）]', '', target)
                if start != -1 and norm_form(sentence[start:end]) not in (norm_form(target), norm_form(stripped)):
                    stats['partial'] += 1
                    print(f'[INFO] Partial match [{func}] "{target}" -> "{sentence[start:end]}" 句子="{sentence}"')
            else:
                used = pick_fallback(body, own_targets, cross_targets)
                if used is None:
                    stats['missing'] += 1
                    print(f'[WARN] Target not found [{func}] 句子="{body}"')
                    target, sentence, start, end = '', body, -1, -1
                else:
                    target, sentence = used, body
                    start, end = locate_target(used, body)
                    stripped = re.sub(r'[()（）]', '', used)
                    if norm_form(body[start:end]) not in (norm_form(used), norm_form(stripped)):
                        stats['partial'] += 1
                        print(f'[INFO] Partial match [{func}] "{used}" -> "{body[start:end]}" 句子="{body}"')
                    if used not in base_targets:
                        stats['fallback'] += 1
                        print(f'[INFO] Fallback target [{func}] "{used}" 句子="{body}"')
            example_list.append({
                '目标词': target,
                '句子': sentence,
                '开头索引': start,
                '末尾索引': end,
            })
            sentences.append(sentence)

        examples_str = '；'.join(fluent_example_lines)
        if fluent_template == DEFAULT_FLUENT_TEMPLATE:
            meaning_clause = f'它的基本含义是"{explanation}"。' if explanation else ''
            fluent_context = fluent_template.format(
                func=func, concept=concept, meaning_clause=meaning_clause,
                extra=extra, example_word=example_word, examples=examples_str)
        else:
            fluent_context = fluent_template.format(
                func=func, zh_name=zh_name, explanation=explanation, examples=examples_str)

        contexts['流畅综合'][func] = fluent_context
        contexts['单独例句'][func] = example_list
        contexts['单独定义'][func] = explanation
        simple_parts = [f'定义：{explanation}'] if explanation else []
        if sentences:
            simple_parts.append(f'例句：{"；".join(sentences)}')
        contexts['简单综合'][func] = '\n'.join(simple_parts)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(contexts, f, ensure_ascii=False, indent=2)

    n_examples = sum(len(v) for v in contexts['单独例句'].values())
    n_indexed = sum(1 for v in contexts['单独例句'].values() for e in v if e['开头索引'] != -1)
    print(f'case{n}: functions={len(contexts["流畅综合"])} examples={n_examples} indexed={n_indexed} '
          f'missing={stats["missing"]} partial={stats["partial"]} fallback={stats["fallback"]} '
          f'row_miss={stats["row_miss"]} -> {os.path.basename(output_file)}')
    return contexts, stats


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    total = {'missing': 0, 'partial': 0, 'fallback': 0, 'row_miss': 0}
    for n in CASES:
        if n not in CASE_CONFIGS:
            print(f'[SKIP] case{n}: 未配置')
            continue
        contexts, stats = process_case(n, CASE_CONFIGS[n])
        for k in total:
            total[k] += stats[k]
        first = next(iter(contexts['流畅综合']))
        sample = contexts['流畅综合'][first]
        print(f'  [{first}] {sample[:160]}...')

    print(f'\nAll done. missing={total["missing"]} partial={total["partial"]} '
          f'fallback={total["fallback"]} row_miss={total["row_miss"]}')


if __name__ == '__main__':
    main()
