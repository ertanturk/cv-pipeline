"""Live job scraping module consuming search queries into memory."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from jobspy import scrape_jobs

from src.models import JobPosting, RoleMatchResult
from src.utils.exit_codes import ExitCode
from src.utils.logger import SessionLogger, log

_STRUCTURAL_TERMS: frozenset[str] = frozenset(
    {
        "intern",
        "internship",
        "junior",
        "senior",
        "entry",
        "level",
        "associate",
        "engineer",
        "engineering",
        "developer",
        "development",
        "specialist",
        "coordinator",
        "manager",
        "analyst",
        "director",
        "head",
        "vp",
        "chief",
        "lead",
        "principal",
        "officer",
        "consultant",
        "architect",
        "and",
        "or",
        "the",
        "a",
        "an",
        "of",
        "in",
        "at",
        "for",
        "&",
        "to",
        "with",
        "system",
        "systems",
    }
)

_SENIOR_GUARD_TERMS: frozenset[str] = frozenset(
    {
        "senior",
        "lead",
        "principal",
        "director",
        "head",
        "vp",
        "vice",
        "chief",
        "president",
        "staff",
        "distinguished",
        "fellow",
        "kıdemli",
        "yönetici",
        "direktör",
        "müdür",
    }
)

_EARLY_CAREER_INDICATORS: frozenset[str] = frozenset(
    {
        "intern",
        "internship",
        "junior",
        "entry",
        "staj",
        "stajyer",
        "stajyeri",
        "stajyerleri",
        "student",
        "co-op",
        "coop",
        "trainee",
        "candidate",
        "aday",
        "öğrenci",
        "part-time",
        "yarı zamanlı",
    }
)

_TURKISH_TECH_TERMS: frozenset[str] = frozenset(
    {
        "yazılım",
        "bilişim",
        "bilgisayar",
        "geliştirici",
        "kod",
        "programcı",
        "teknoloji",
        "teknolojileri",
    }
)

_NON_SOFTWARE_DOMAINS: frozenset[str] = frozenset(
    {
        "thermal",
        "mechanical",
        "civil",
        "chemical",
        "hvac",
        "simulation",
        "supply chain",
        "sales",
        "marketing",
        "accounting",
        "recruiter",
        "hr",
        "talent acquisition",
        "legal",
        "customer success",
        "credit risk",
        "solidworks",
        "biorefinery",
        "account manager",
        "ot systems",
        "operational technology",
    }
)

_FETCH_BUFFER: int = 3


def _normalize_text(text: str) -> str:
    """Normalizes compound terms and hyphenated phrases."""
    t = text.lower()
    t = re.sub(r"back[\s\-_]+end", "backend", t)
    t = re.sub(r"front[\s\-_]+end", "frontend", t)
    t = re.sub(r"full[\s\-_]+stack", "fullstack", t)
    t = re.sub(r"dev[\s\-_]+ops", "devops", t)
    return t


def _tokenize(text: str) -> set[str]:
    """Tokenizes text extracting clean word tokens, stripping punctuation and brackets."""
    return set(re.findall(r"[a-zA-ZçğıöşüÇĞİÖŞÜ0-9\+#]+", _normalize_text(text)))


def _build_domain_keywords(texts: Sequence[str]) -> frozenset[str]:
    """Extracts subject-matter domain keywords from multiple text sources (role title + search queries)."""
    result: set[str] = set()
    for text in texts:
        for w in _tokenize(text):
            if w not in _STRUCTURAL_TERMS and len(w) > 2:
                result.add(w)
    return frozenset(result)


def _is_early_career(job_type: str | None, role_title: str | None) -> bool:
    """Returns True when the search context targets an intern or junior-level position."""
    if job_type and job_type.lower() == "internship":
        return True
    if role_title:
        return bool(_tokenize(role_title) & _EARLY_CAREER_INDICATORS)
    return False


def _is_senior_posting(job_title: str) -> bool:
    """Returns True if the job title contains a seniority-guard term."""
    return bool(_tokenize(job_title) & _SENIOR_GUARD_TERMS)


def _is_domain_relevant(
    job_title: str,
    domain_keywords: frozenset[str],
    description: str = "",
    is_early_career: bool = False,
    is_turkey: bool = False,
) -> bool:
    """Returns True if the posting is domain relevant to the target role."""
    norm_title = _normalize_text(job_title)
    title_tokens = _tokenize(job_title)

    # 1. Reject titles with clearly non-software domain terms unless explicit software keywords present
    has_excluded = any(ex in norm_title for ex in _NON_SOFTWARE_DOMAINS)
    has_explicit_tech = bool(
        title_tokens
        & {
            "software",
            "backend",
            "frontend",
            "fullstack",
            "yazılım",
            "python",
            "developer",
            "engineer",
            "c++",
            "java",
        }
    )
    if has_excluded and not has_explicit_tech:
        return False

    # 2. Check direct domain keywords match on title
    if domain_keywords and bool(domain_keywords & title_tokens):
        return True

    # 3. Core software engineering keywords match
    if bool(title_tokens & {"software", "backend", "developer", "python", "java", "c++"}):
        return True

    # 4. Turkish tech equivalents if target market is Turkey
    if is_turkey and bool(title_tokens & _TURKISH_TECH_TERMS):
        return True

    # 5. Generic early career postings (intern, stajyer, co-op, student)
    if is_early_career:
        is_early_title = bool(title_tokens & _EARLY_CAREER_INDICATORS)
        if is_early_title:
            if any(
                term in norm_title
                for term in (
                    "engineer",
                    "mühendis",
                    "developer",
                    "geliştirici",
                    "tech",
                    "bilişim",
                    "teknoloji",
                    "bilgi",
                    "r&d",
                    "network",
                    "data",
                    "cloud",
                    "it",
                )
            ):
                return True
            if description:
                desc_tokens = _tokenize(description[:1500])
                tech_probe = domain_keywords | {
                    "software",
                    "backend",
                    "python",
                    "code",
                    "database",
                    "api",
                    "git",
                    "linux",
                    "yazılım",
                }
                if bool(desc_tokens & tech_probe):
                    return True
            # For Turkey, branded tech internship programs without explicit exclude
            if is_turkey and not has_excluded:
                return True

    # 6. Check description for strong domain signals if title was borderline
    if description and domain_keywords:
        desc_tokens = _tokenize(description[:1000])
        if len(desc_tokens & domain_keywords) >= 2:
            return True

    return False


def _sanitize_queries(raw_queries: Sequence[str], location: str = "") -> list[str]:
    """Cleans search queries by removing redundant words and extraneous location tokens."""
    loc_words = set(re.findall(r"[a-zA-ZçğıöşüÇĞİÖŞÜ]+", location.lower())) if location else set()
    sanitized: list[str] = []

    for q in raw_queries:
        clean = q.strip()
        words = [w for w in clean.split() if w.lower() not in loc_words]
        if not words:
            continue

        if len(words) > 3:
            lower_words = [w.lower() for w in words]
            filtered = []
            for w in words:
                wl = w.lower()
                if wl == "engineering" and ("software" in lower_words or "backend" in lower_words):
                    continue
                filtered.append(w)
            words = filtered[:3] if len(filtered) > 3 else filtered

        candidate = " ".join(words)
        if candidate and candidate not in sanitized:
            sanitized.append(candidate)

    return sanitized


def _scrape_query(
    query: str,
    query_idx: int,
    total_queries: int,
    sites: list[str],
    per_query_target: int,
    location: str,
    hours_old: int,
    country_indeed: str,
    job_type: str | None,
) -> list[pd.DataFrame]:
    """Scrapes all sites for a single query. Runs in its own thread."""
    frames: list[pd.DataFrame] = []
    log.info(f"[{query_idx}/{total_queries}] Querying '{query}' across {sites}...")
    for site in sites:
        try:
            df: pd.DataFrame = scrape_jobs(
                site_name=[site],
                search_term=query,
                location=location,
                results_wanted=per_query_target,
                hours_old=hours_old,
                country_indeed=country_indeed,
                linkedin_fetch_description=True,
                is_remote=False,
                job_type=job_type,
                verbose=0,
            )
            if not df.empty:
                log.debug(f"Retrieved {len(df)} listings from {site} for '{query}'")
                frames.append(df)
            else:
                log.debug(f"Site '{site}' returned 0 listings for '{query}'")
        except Exception as exc:
            log.warning(f"Scraper error on {site} for query '{query}': {exc}")
    return frames


def _scrape_raw(
    queries: list[str],
    sites: list[str],
    per_query_target: int,
    location: str,
    hours_old: int,
    country_indeed: str,
    job_type: str | None,
) -> list[pd.DataFrame]:
    """Dispatches one thread per query, collects all DataFrames into a single pool."""
    total: int = len(queries)
    all_frames: list[pd.DataFrame] = []

    with ThreadPoolExecutor(max_workers=min(max(1, total), 8)) as executor:
        futures = {
            executor.submit(
                _scrape_query,
                query,
                idx,
                total,
                sites,
                per_query_target,
                location,
                hours_old,
                country_indeed,
                job_type,
            ): query
            for idx, query in enumerate(queries, start=1)
        }
        for future in as_completed(futures):
            query: str = futures[future]
            try:
                all_frames.extend(future.result())
            except Exception as exc:
                log.warning(f"Thread failed for query '{query}': {exc}")

    return all_frames


def _normalize_and_deduplicate(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenates raw frames, normalizes columns, applies fallback descriptions, and deduplicates."""
    combined: pd.DataFrame = pd.concat(frames, ignore_index=True)

    for col in ["title", "company", "location", "description", "job_url", "site"]:
        if col not in combined.columns:
            combined[col] = ""
        combined[col] = combined[col].fillna("").astype(str).str.strip()

    short_desc: pd.Series = combined["description"].str.len() < 30
    combined.loc[short_desc, "description"] = (
        combined.loc[short_desc, "title"]
        + " at "
        + combined.loc[short_desc, "company"]
        + ". Role based in "
        + combined.loc[short_desc, "location"]
        + "."
    )

    has_url: pd.Series = combined["job_url"] != ""
    df_with_url: pd.DataFrame = combined[has_url].drop_duplicates(subset=["job_url"], keep="first")
    df_no_url: pd.DataFrame = combined[~has_url]
    combined = pd.concat([df_with_url, df_no_url], ignore_index=True)

    has_identity: pd.Series = (combined["title"] != "") & (combined["company"] != "")
    df_with_id: pd.DataFrame = combined[has_identity].drop_duplicates(
        subset=["title", "company"], keep="first"
    )
    df_no_id = combined[~has_identity]
    combined = pd.concat([df_with_id, df_no_id], ignore_index=True)

    return combined[combined["title"] != ""].reset_index(drop=True)


def run_scraper(
    queries: Sequence[str] | RoleMatchResult,
    target_count: int = 40,
    location: str = "Istanbul, Turkey",
    sites: list[str] | None = None,
    hours_old: int = 4320,  # last 6 months
    country_indeed: str = "turkey",
    job_type: str | None = None,
    role_title: str | None = None,
) -> list[JobPosting]:
    """Scrapes live job postings across compatible platforms and retains them in memory."""
    raw_queries: list[str]
    if isinstance(queries, RoleMatchResult):
        if queries.status == "insufficient_data" or not queries.primary_role:
            SessionLogger.fatal(
                "Cannot scrape jobs: RoleMatchResult contains insufficient data or empty primary role.",
                code=ExitCode.SCRAPER_ERROR,
            )
        raw_queries = queries.primary_role.search_queries
        if role_title is None:
            role_title = queries.primary_role.title
    else:
        raw_queries = [q.strip() for q in queries if q.strip()]

    if not raw_queries:
        SessionLogger.fatal(
            "No valid search queries provided for scraping.",
            code=ExitCode.SCRAPER_ERROR,
        )

    is_turkey: bool = country_indeed.lower() in {"turkey", "tr"} or any(
        loc_term in location.lower() for loc_term in ("turkey", "türkiye", "istanbul")
    )

    resolved_queries: list[str] = _sanitize_queries(raw_queries, location=location)
    is_early: bool = _is_early_career(job_type, role_title)

    if is_turkey and is_early:
        local_additions = ["Software Intern", "Yazılım Stajyer"]
        for q in local_additions:
            if q not in resolved_queries and len(resolved_queries) < 6:
                resolved_queries.append(q)

    default_sites: list[str] = ["linkedin", "indeed", "google"]
    scrape_sites: list[str]
    if country_indeed.lower() in {"turkey", "tr"}:
        scrape_sites = [
            s for s in (sites or default_sites) if s.lower() not in {"glassdoor", "ziprecruiter"}
        ]
    else:
        scrape_sites = sites or default_sites

    job_type_label: str = job_type if job_type else "any"
    role_label: str = role_title if role_title else "unfiltered"
    log.info(
        f"Initiating Stage 2: In-Memory Job Scraping | Target: {target_count} postings | "
        f"Type: {job_type_label} | Role filter: {role_label} | Sites: {scrape_sites}"
    )
    log.debug(f"Resolved queries: {resolved_queries}")

    per_query_target: int = max(10, math.ceil(target_count * _FETCH_BUFFER / len(resolved_queries)))

    raw_frames: list[pd.DataFrame] = _scrape_raw(
        resolved_queries,
        scrape_sites,
        per_query_target,
        location,
        hours_old,
        country_indeed,
        job_type,
    )

    if not raw_frames:
        log.warning(f"All scraper queries returned zero listings for location='{location}'.")
        return []

    combined_df: pd.DataFrame = _normalize_and_deduplicate(raw_frames)
    initial_count: int = len(combined_df)
    log.debug(f"Total raw listings collected before filtering: {initial_count}")

    if combined_df.empty:
        log.warning(
            f"No valid job postings remained after deduplication for location='{location}'."
        )
        return []

    domain_keywords: frozenset[str] = (
        _build_domain_keywords([role_title, *resolved_queries]) if role_title else frozenset()
    )
    log.debug(f"Domain keywords for relevancy filtering: {sorted(domain_keywords)}")

    pre_filter_count: int = len(combined_df)
    if is_early:
        combined_df = combined_df[~combined_df["title"].apply(_is_senior_posting)]
        log.debug(
            f"Seniority guard removed {pre_filter_count - len(combined_df)} senior listings; "
            f"{len(combined_df)} remain."
        )

    if domain_keywords:
        pre_domain_count: int = len(combined_df)
        combined_df = combined_df[
            combined_df.apply(
                lambda row: _is_domain_relevant(
                    job_title=row["title"],
                    domain_keywords=domain_keywords,
                    description=row.get("description", ""),
                    is_early_career=is_early,
                    is_turkey=is_turkey,
                ),
                axis=1,
            )
        ]
        log.debug(f"Domain filter kept {len(combined_df)}/{pre_domain_count} listings.")

    if len(combined_df) < target_count:
        deficit: int = target_count - len(combined_df)
        log.info(
            f"Primary pool has {len(combined_df)}/{target_count} listings. "
            f"Backfilling deficit of {deficit} with broader postings..."
        )

        backfill_candidates: list[str]
        if is_early:
            if is_turkey:
                backfill_candidates = [
                    "Software Intern",
                    "Yazılım Stajyer",
                    "Backend Developer",
                    "Python Developer",
                    "Software Engineer",
                ]
            else:
                backfill_candidates = [
                    "Software Intern",
                    "Junior Software Engineer",
                    "Backend Developer",
                    "Python Developer",
                ]
        else:
            backfill_candidates = [
                role_title or "Software Engineer",
                "Backend Developer",
                "Software Engineer",
                "Python Developer",
            ]

        backfill_queries = [q for q in backfill_candidates if q not in resolved_queries]
        if not backfill_queries:
            backfill_queries = ["Software Developer", "Python Developer", "Backend Developer"]

        backfill_per_query: int = max(
            10, math.ceil(deficit * _FETCH_BUFFER / len(backfill_queries))
        )
        backfill_frames: list[pd.DataFrame] = _scrape_raw(
            backfill_queries,
            scrape_sites,
            backfill_per_query,
            location,
            hours_old,
            country_indeed,
            job_type=None,
        )

        if backfill_frames:
            backfill_df: pd.DataFrame = _normalize_and_deduplicate(backfill_frames)
            if is_early:
                backfill_df = backfill_df[~backfill_df["title"].apply(_is_senior_posting)]
            if domain_keywords:
                backfill_df = backfill_df[
                    backfill_df.apply(
                        lambda row: _is_domain_relevant(
                            job_title=row["title"],
                            domain_keywords=domain_keywords,
                            description=row.get("description", ""),
                            is_early_career=is_early,
                            is_turkey=is_turkey,
                        ),
                        axis=1,
                    )
                ]
            combined_df = pd.concat([combined_df, backfill_df], ignore_index=True)

            has_url: pd.Series = combined_df["job_url"] != ""
            combined_df = pd.concat(
                [
                    combined_df[has_url].drop_duplicates(subset=["job_url"], keep="first"),
                    combined_df[~has_url],
                ],
                ignore_index=True,
            )
            has_identity: pd.Series = (combined_df["title"] != "") & (combined_df["company"] != "")
            df_with_id = combined_df[has_identity].drop_duplicates(
                subset=["title", "company"], keep="first"
            )
            df_no_id = combined_df[~has_identity]
            combined_df = pd.concat([df_with_id, df_no_id], ignore_index=True)
            log.info(f"Backfill complete; pool now at {len(combined_df)} listings.")

    final_df: pd.DataFrame = combined_df.head(target_count)

    job_postings: list[JobPosting] = [
        JobPosting(
            title=row["title"],
            company=row["company"],
            location=row["location"],
            description=row["description"],
            job_url=row["job_url"],
            site=row["site"],
        )
        for _, row in final_df.iterrows()
    ]

    log.info(
        f"Scraping completed: {len(job_postings)} valid listings retained in memory "
        f"(deduplicated from {initial_count} raw listings)."
    )
    for idx, jp in enumerate(job_postings[:10], start=1):
        log.debug(
            f"Retained listing #{idx}: '{jp.title}' at '{jp.company}' ({jp.location}) via {jp.site or 'unknown'}"
        )
    if len(job_postings) > 10:
        log.debug(f"... and {len(job_postings) - 10} additional listings retained in memory.")

    return job_postings
