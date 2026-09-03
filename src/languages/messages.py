"""Translation dictionary for interactive CLI prompts and warnings."""

from typing import Literal

UiLanguage = Literal["en", "tr"]

_MESSAGES: dict[str, dict[UiLanguage, str]] = {
    "ui_language_prompt": {
        "en": "Select CLI language (en/tr)",
        "tr": "CLI dilini seçin (en/tr)",
    },
    "ui_language_box_hint": {
        "en": "Press [1] for TR or [2] for EN.",
        "tr": "[1] ile TR, [2] ile EN seçin.",
    },
    "ui_language_invalid": {
        "en": "Invalid choice. Please enter 'en' or 'tr'.",
        "tr": "Geçersiz seçim. Lütfen 'en' veya 'tr' girin.",
    },
    "ui_language_selected": {
        "en": "CLI language selected: {lang}",
        "tr": "CLI dili seçildi: {lang}",
    },
    "profile_not_found": {
        "en": "Candidate profile not found at: {path}",
        "tr": "Aday profil dosyası bulunamadı: {path}",
    },
    "api_key_missing": {
        "en": "Gemini API key is missing. Run 'cv auth set' first.",
        "tr": "Gemini API anahtarı eksik. Önce 'cv auth set' çalıştırın.",
    },
    "pdflatex_missing": {
        "en": "System dependency missing: 'pdflatex' executable not found in PATH.",
        "tr": "Sistem bağımlılığı eksik: PATH içinde 'pdflatex' bulunamadı.",
    },
    "template_not_found": {
        "en": "LaTeX template not found at: {path}",
        "tr": "LaTeX şablonu bulunamadı: {path}",
    },
    "step_role": {
        "en": "Analyzing Profile & Target Role Resolution",
        "tr": "Profil Analizi ve Hedef Rol Çözümleme",
    },
    "step_role_exec": {
        "en": "Resolving optimal market role...",
        "tr": "En uygun pazar rolü belirleniyor...",
    },
    "insufficient_role_warning": {
        "en": "Profile data was insufficient for automatic role detection.",
        "tr": "Profil verisi otomatik rol tespiti için yetersiz kaldı.",
    },
    "role_prompt": {
        "en": "Role could not be determined from profile. Which role do you want to search for?",
        "tr": "Profilden rol belirlenemedi. Hangi rol için arama yapmak istersiniz?",
    },
    "invalid_role_error": {
        "en": "No valid role was provided.",
        "tr": "Geçerli bir rol girilmedi.",
    },
    "resolved_role": {
        "en": "Resolved Role: {role}",
        "tr": "Belirlenen Rol: {role}",
    },
    "role_confidence": {
        "en": "Role confidence: {score}%",
        "tr": "Rol güven skoru: {score}%",
    },
    "step_scrape": {
        "en": "Scraping Live Market Data ({location})",
        "tr": "Canlı Pazar Verisi Taranıyor ({location})",
    },
    "step_scrape_exec": {
        "en": "Scraping top {jobs} active listings...",
        "tr": "En iyi {jobs} aktif ilan taranıyor...",
    },
    "retained_listings": {
        "en": "Retained {count} active listings in memory.",
        "tr": "Bellekte {count} aktif ilan tutuldu.",
    },
    "step_market": {
        "en": "Extracting Market Competencies",
        "tr": "Pazar Yetkinlikleri Çıkarılıyor",
    },
    "step_market_exec": {
        "en": "Running Map-Reduce market analysis...",
        "tr": "Map-Reduce pazar analizi çalıştırılıyor...",
    },
    "market_missing_confirm": {
        "en": "No market data could be extracted. Do you want to continue?",
        "tr": "Market verisi bulunamadı. Devam etmek ister misiniz?",
    },
    "market_aborted_error": {
        "en": "Pipeline stopped because market data was missing and continuation was not approved.",
        "tr": "Market verisi olmadığı ve devam onayı verilmediği için pipeline durduruldu.",
    },
    "market_continue_warning": {
        "en": "Continuing without market signals; resume generation will use profile-based tailoring only.",
        "tr": "Market sinyali olmadan devam ediliyor; CV üretimi yalnızca profil tabanlı yapılacak.",
    },
    "discovered_skills": {
        "en": "Analyzed {analyzed} listings; discovered {count} unique technical skills.",
        "tr": "{analyzed} ilan incelendi; {count} benzersiz teknik yetkinlik bulundu.",
    },
    "step_tailor": {
        "en": "Tailoring Projects & Experience",
        "tr": "Projeler ve Deneyim Özelleştiriliyor",
    },
    "step_tailor_exec": {
        "en": "Reconciling profile with market signals...",
        "tr": "Profil, pazar sinyalleriyle uyumlandırılıyor...",
    },
    "tailored_projects": {
        "en": "Tailored {count} engineering projects.",
        "tr": "{count} teknik proje özelleştirildi.",
    },
    "step_compile": {
        "en": "Compiling ATS-Optimized PDF ({template})",
        "tr": "ATS Uyumlu PDF Derleniyor ({template})",
    },
    "step_compile_exec": {
        "en": "Rendering Jinja2 and compiling via pdflatex...",
        "tr": "Jinja2 render ve pdflatex derleme yapılıyor...",
    },
    "compact_mode_start": {
        "en": "Initial output is {pages} pages; enabling compact mode and recompiling...",
        "tr": "İlk çıktı {pages} sayfa; compact mod açılıyor ve yeniden derleniyor...",
    },
    "compact_pass_exec": {
        "en": "Running compact pass {current}/{total}...",
        "tr": "Compact geçişi çalışıyor {current}/{total}...",
    },
    "compact_new_numbers": {
        "en": "Compact pass {pass_index} introduced new numeric values; retrying with stricter constraints.",
        "tr": "Compact geçişi {pass_index} yeni sayısal değer üretti; daha sıkı kurallarla tekrar deneniyor.",
    },
    "compact_recompile_exec": {
        "en": "Recompiling compact candidate...",
        "tr": "Compact aday yeniden derleniyor...",
    },
    "compact_pass_still_oversize": {
        "en": "Compact pass {pass_index} still produced {pages} pages.",
        "tr": "Compact geçişi {pass_index} hâlâ {pages} sayfa üretti.",
    },
    "compact_failed_limit": {
        "en": "Page budget exceeded after {passes} compact passes: generated resume is {pages} pages. Pipeline enforces a strict 1-page limit and removed the oversized PDF.",
        "tr": "{passes} compact geçişinden sonra sayfa limiti aşıldı: üretilen CV {pages} sayfa. Pipeline 1 sayfa sınırını uygular ve büyük PDF'i siler.",
    },
    "compact_applied": {
        "en": "Compact mode applied: {initial} → {final} page (pass {pass_used}).",
        "tr": "Compact mod uygulandı: {initial} → {final} sayfa (geçiş {pass_used}).",
    },
    "pdf_generated": {
        "en": "PDF generated: {name} ({size_kb:.1f} KB)",
        "tr": "PDF oluşturuldu: {name} ({size_kb:.1f} KB)",
    },
    "latex_saved": {
        "en": "LaTeX source saved: {name}",
        "tr": "LaTeX kaynağı kaydedildi: {name}",
    },
    "pipeline_success": {
        "en": "Pipeline completed successfully in {seconds:.1f}s",
        "tr": "Pipeline başarıyla tamamlandı: {seconds:.1f}s",
    },
    "pipeline_failed": {
        "en": "Pipeline execution failed: {error}",
        "tr": "Pipeline çalıştırması başarısız oldu: {error}",
    },
    "market_col_num": {
        "en": "#",
        "tr": "#",
    },
    "market_col_skill": {
        "en": "Skill / Keyword",
        "tr": "Yetkinlik / Anahtar Kelime",
    },
    "market_col_mentions": {
        "en": "Mentions",
        "tr": "Geçiş Sayısı",
    },
    "market_col_share": {
        "en": "Share",
        "tr": "Pazar Payı",
    },
    "market_col_saturation": {
        "en": "Market Saturation",
        "tr": "Pazar Doygunluğu",
    },
}


def tr_message(key: str, language: str = "en") -> str:
    """Returns translated message text, defaulting to English."""
    lang: UiLanguage = "tr" if language == "tr" else "en"
    return _MESSAGES.get(key, {}).get(lang, _MESSAGES.get(key, {}).get("en", key))
