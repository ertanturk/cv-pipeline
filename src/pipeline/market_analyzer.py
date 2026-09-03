"""Market Skill Analysis module with prompt batching."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence

from src.llm import GeminiEngine
from src.models import (
    BatchJobExtractionResult,
    JobExtractionResult,
    JobPosting,
    MarketAnalysisResult,
    MarketSkill,
)
from src.utils.logger import log

TAIL_CUTOFF_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"(?im)^\s*(?:[#*\-•\d\.\s]*)\b(?:"
        r"what\s+(?:can\s+)?we\s+offer|what\s+we\s+(?:can\s+)?offer|"
        r"(?:our\s+)?perks(?:\s+(?:and|&)\s+benefits)?|"
        r"(?:the\s+)?benefits(?:\s+of\s+working(?:\s+with\s+us)?)?|"
        r"compensation(?:\s+(?:and|&)\s+benefits)?|rewards(?:\s+package)?|"
        r"neler\s+sunuyoruz|neler\s+vadediyoruz|yan\s+haklar|sağlanan\s+imkanlar|"
        r"bizimle\s+çalışmanın\s+avantajları"
        r")\b.*$"
    ),
    re.compile(
        r"(?im)^\s*(?:[#*\-•\d\.\s]*)\b(?:"
        r"(?:the\s+)?recruiting\s+process|recruitment\s+process|application\s+process|"
        r"how\s+to\s+apply|interview\s+process|hiring\s+process|"
        r"işe\s+alım\s+süreci|başvuru\s+süreci"
        r")\b.*$"
    ),
    re.compile(
        r"(?im)^\s*(?:[#*\-•\d\.\s]*)\b(?:"
        r"diversity\s*(?:&|and)\s*inclusion|equal\s+opportunity(?:\s+employer)?|"
        r"be\s+aware\s+of\s+recruitment\s+scams|recruitment\s+(?:fraud|scams)|"
        r"statement\s+of\s+personal\s+data|we\s+respect\s+your\s+privacy"
        r")\b.*$"
    ),
    re.compile(
        r"(?im)^\s*(?:#{1,6}|\*{1,2})?\s*(?:privacy\s+policy|gizlilik\s+politikası)\s*(?:\*{1,2})?\s*$"
    ),
    re.compile(
        r"(?im)^\s*.*(?:"
        r"6698\s+sayılı\s+kişisel\s+verilerin\s+korunması|"
        r"bu\s+aydınlatma\s+metni|"
        r"kişisel\s+verilerin\s+korunması\s+kanunu"
        r").*$"
    ),
    re.compile(r"(?im)^\s*(?:#{1,6}|\*{1,2})\s*kvkk\s*(?:\*{1,2})?.*$"),
]


def _stitch_wrapped_markdown(text: str) -> str:
    """Reconstructs lines broken mid-sentence around bold tags."""
    stitched: list[str] = []
    prev_was_blank = False

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            prev_was_blank = True
            continue
        if not stitched:
            stitched.append(line)
            prev_was_blank = False
            continue

        is_bullet = bool(re.match(r"^(?:[\-•]|\*(?!\*))\s+", line)) or bool(
            re.match(r"^\d+[\.\)]\s+", line)
        )
        is_header = bool(re.match(r"^(?:#{1,6}\s+|\*\*[^*]+\*\*\s*:?$)", line))
        prev = stitched[-1]

        if prev_was_blank:
            prev_was_blank = False
            is_soft_continuation = (
                not (is_bullet or is_header)
                and (line[0].islower() or line[0] in {",", ")", "]", "}", "."})
                and not prev.endswith((".", ":", ";", "!", "?"))
            )
            if is_soft_continuation:
                stitched[-1] = f"{prev} {line}"
            else:
                stitched.append(line)
            continue

        is_continuation = (
            not (is_bullet or is_header)
            and (line[0].islower() or line[0] in {",", ")", "]", "}", "."})
            and not prev.endswith((".", ":", ";", "!", "?"))
        )
        if is_continuation:
            stitched[-1] = f"{prev} {line}"
        else:
            stitched.append(line)

    return "\n".join(stitched)


def prune_job_description(raw_text: str, max_chars: int = 2500) -> str:
    """Extracts technical requirements while removing corporate fluff and perks."""
    if not raw_text or not raw_text.strip():
        return ""

    text = (
        raw_text.replace(r"\-", "-")
        .replace(r"\+", "+")
        .replace(r"\'", "'")
        .replace(r"\"", '"')
        .replace(r"\_", "_")
    )
    healed_text = _stitch_wrapped_markdown(text)

    earliest_cutoff = len(healed_text)
    for pattern in TAIL_CUTOFF_PATTERNS:
        match = pattern.search(healed_text)
        if match and match.start() < earliest_cutoff:
            earliest_cutoff = match.start()

    pruned = healed_text[:earliest_cutoff].strip() or healed_text.strip()

    tech_start_match = re.search(
        r"(?im)^\s*(?:[#*\-•\d\.\s\W]*)\b(?:"
        r".*?(?:responsibilit|qualification|requirement|aranan\s+nitelik|görev\s+tanım)"
        r"|what\s+we\s+(?:are\s+)?look(?:ing)?\s+for"
        r"|what\s+you(?:'ll|\s+will)?\s+(?:bring|do|need)"
        r"|desired\s+skills|must\s+have|nice\s+to\s+have|tech(?:nical)?\s+stack"
        r")\b.*$",
        pruned,
    )
    if tech_start_match and tech_start_match.start() > 150:
        pruned = pruned[tech_start_match.start() :].strip()

    if len(pruned) > max_chars:
        pruned = pruned[:max_chars].rsplit("\n", 1)[0] + "\n[Content truncated for analysis]"

    return pruned


def _elect_canonical(variants: Counter[str]) -> str:
    """Elects the most frequent properly capitalized variant from observed names."""
    properly_cased = [(name, count) for name, count in variants.items() if name != name.lower()]
    if properly_cased:
        return max(properly_cased, key=lambda x: x[1])[0]
    return variants.most_common(1)[0][0]


def run_market_analyzer(
    postings: Sequence[JobPosting],
    max_analyze: int = 15,
    batch_size: int = 8,
    engine: GeminiEngine | None = None,
) -> MarketAnalysisResult:
    """Executes batched Map-Reduce skill extraction across in-memory jobs."""
    if not postings:
        log.warning("No job postings provided for market analysis; returning empty result.")
        return MarketAnalysisResult(total_analyzed=0)

    target_pool: list[JobPosting] = list(postings)[:max_analyze]
    log.info(
        f"Initiating Stage 3: Market Analysis across {len(target_pool)} postings "
        f"(Batched, max_analyze={max_analyze}, batch_size={batch_size})..."
    )

    active_engine: GeminiEngine = engine or GeminiEngine()

    # Prune all postings upfront to reduce token volume
    prepared_payloads: list[dict[str, str]] = []
    total_raw_chars = 0
    total_pruned_chars = 0
    for idx, job in enumerate(target_pool, start=1):
        raw_len = len(job.description)
        total_raw_chars += raw_len
        pruned = prune_job_description(job.description)
        pruned_len = len(pruned)
        total_pruned_chars += pruned_len
        if pruned_len >= 40:
            prepared_payloads.append({"title": job.title, "company": job.company, "text": pruned})
            log.debug(
                f"Job #{idx} ('{job.title}' at '{job.company}'): pruned {raw_len} -> {pruned_len} chars"
            )
        else:
            log.warning(
                f"Job #{idx} ('{job.title}') description too short after pruning ({pruned_len} chars); skipping."
            )

    log.debug(
        f"Pruning complete: {len(prepared_payloads)}/{len(target_pool)} postings retained "
        f"({total_raw_chars} -> {total_pruned_chars} total characters)"
    )

    if not prepared_payloads:
        log.warning("No postings met minimum character requirements after pruning.")
        return MarketAnalysisResult(total_analyzed=0)

    # Process in batches to minimize rate-limiter sleeps
    extractions: list[JobExtractionResult] = []
    for chunk_start in range(0, len(prepared_payloads), batch_size):
        chunk = prepared_payloads[chunk_start : chunk_start + batch_size]
        log.info(f"Analyzing batch of {len(chunk)} job postings in a single prompt...")

        try:
            if hasattr(active_engine, "extract_batch_job_skills"):
                batch_res: BatchJobExtractionResult = active_engine.extract_batch_job_skills(chunk)
                extractions.extend(batch_res.jobs)
            else:
                for item in chunk:
                    res = active_engine.extract_job_skills(item["text"], JobExtractionResult)
                    extractions.append(res)
        except Exception as exc:
            log.warning(f"Batch extraction failed: {exc}")

    # Reduce Phase
    tool_mentions: Counter[str] = Counter()
    tool_required_counts: Counter[str] = Counter()
    tool_name_variants: dict[str, Counter[str]] = {}
    methodology_mentions: Counter[str] = Counter()
    methodology_name_variants: dict[str, Counter[str]] = {}
    successful_parses = 0

    for extraction in extractions:
        if extraction is None or extraction.status == "no_data":
            continue

        successful_parses += 1
        seen_tools_in_job: dict[str, bool] = {}

        for tool in extraction.tools:
            name = tool.name.strip()
            if not name:
                continue

            lower_key = name.lower()
            tool_name_variants.setdefault(lower_key, Counter())[name] += 1
            is_required = tool.priority.lower() == "required"

            if lower_key not in seen_tools_in_job:
                tool_mentions[lower_key] += 1
                seen_tools_in_job[lower_key] = is_required
                if is_required:
                    tool_required_counts[lower_key] += 1
            elif is_required and not seen_tools_in_job[lower_key]:
                seen_tools_in_job[lower_key] = True
                tool_required_counts[lower_key] += 1

        seen_methods_in_job: set[str] = set()
        for method in extraction.methodologies:
            method_name = method.strip()
            if not method_name:
                continue

            lower_key = method_name.lower()
            methodology_name_variants.setdefault(lower_key, Counter())[method_name] += 1

            if lower_key not in seen_methods_in_job:
                methodology_mentions[lower_key] += 1
                seen_methods_in_job.add(lower_key)

    if successful_parses == 0:
        log.warning("No technical skills could be extracted. Returning empty result.")
        return MarketAnalysisResult(total_analyzed=0)

    canonical_tool_names = {k: _elect_canonical(v) for k, v in tool_name_variants.items()}
    canonical_methodology_names = {
        k: _elect_canonical(v) for k, v in methodology_name_variants.items()
    }

    ranked_tools: list[MarketSkill] = [
        MarketSkill(
            name=canonical_tool_names[k],
            mentions=count,
            share_percentage=round((count / successful_parses) * 100, 1),
            required_count=tool_required_counts[k],
        )
        for k, count in tool_mentions.most_common()
    ]

    ranked_methodologies: list[MarketSkill] = [
        MarketSkill(
            name=canonical_methodology_names[k],
            mentions=count,
            share_percentage=round((count / successful_parses) * 100, 1),
            required_count=0,
        )
        for k, count in methodology_mentions.most_common()
    ]

    log.info(
        f"Analyzed {successful_parses} listings. "
        f"Discovered {len(ranked_tools)} unique tools and {len(ranked_methodologies)} methodologies."
    )
    for tool in ranked_tools[:8]:
        log.debug(
            f"Top market tool: '{tool.name}' | {tool.mentions} mentions "
            f"({tool.share_percentage}%) | {tool.required_count} required"
        )

    return MarketAnalysisResult(
        total_analyzed=successful_parses,
        top_skills=ranked_tools,
        top_methodologies=ranked_methodologies,
        recommended_keywords=[tool.name for tool in ranked_tools[:12]],
    )
