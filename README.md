# cv-pipeline

Automated, ATS-optimized resume generator powered by Google Gemini and LaTeX. Matches candidate profiles against live job market demand to produce clean, high-density, single-page resumes.

> [!WARNING]
> **Important Notice & Limitations:**
> - **LLM Mistakes:** Because this tool uses Large Language Models (LLMs), it can make mistakes. Always review and verify the generated resume before using it.
> - **Target Audience:** This tool is currently intended **only for Computer Science (CS) students**.
> - **Region Support:** Currently supports the **Turkey region only**.
> - **Expectations:** Do not expect flawless or extraordinary resumes right out of the box. Based on the skillset provided in the candidate profile and current market trends, the LLM will provide a **good starting template** that you should refine and adapt.

---

## Features

- **Market-Driven Tailoring**: Analyzes active job listings from LinkedIn, Indeed, and Google Jobs to prioritize relevant skills and frame project highlights.
- **Strict 1-Page Guarantee**: Formats content to fit precisely on a single page using LaTeX, with automated multi-pass text compaction if content overflows.
- **Bilingual Support**: Generates resumes in English or Turkish with appropriate professional terminology and formatting.
- **Metric Preservation**: Retains quantitative accomplishments, numbers, and technical stack details without hallucination.
- **Flexible Authentication**: Supports persistent OS keyring storage as well as `GEMINI_API_KEY` environment variable for headless environments and CI/CD.

---

## Prerequisites

### 1. LaTeX Engine (`pdflatex`)

A working TeX distribution is required to compile PDF resumes:

- **Ubuntu / Debian**:
    ```bash
    sudo apt-get update && sudo apt-get install -y texlive-latex-base texlive-fonts-recommended texlive-fonts-extra texlive-latex-extra poppler-utils
    ```
- **macOS**:
    ```bash
    brew install --cask mactex-no-gui
    brew install poppler
    ```
- **Arch Linux**:
    ```bash
    sudo pacman -S texlive-basic texlive-fontsrecommended poppler
    ```

### 2. Python

- Python 3.12 or newer.

### 3. Gemini API Key

- An active API key from Google AI Studio.

---

## Installation

Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/ertantuncturk/cv-pipeline.git
cd cv-pipeline

# Using uv (recommended)
uv sync

# Or using standard venv
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

---

## Configuration

Set your Gemini API key using either method:

### Method 1: System Keyring (Recommended for Local Use)

Store the API key securely in your operating system's keyring:

```bash
cv auth set
```

### Method 2: Environment Variable (CI/CD / Headless)

```bash
export GEMINI_API_KEY="your_api_key_here"
```

Verify your authentication status:

```bash
cv auth status
```

---

## Quick Start

### 1. Prepare Your Profile YAML

Create a candidate profile file (e.g., `candidate_profile.yaml`):

```yaml
candidate_tier: "student" # "student" or "junior"

personal_info:
    name: "John Doe"
    title: "Backend Engineer"
    location: "Istanbul, Turkey"
    email: "john@example.com"
    phone: "+90 555 123 4567"
    linkedin_url: "https://linkedin.com/in/johndoe"
    github_url: "https://github.com/johndoe"
    portfolio_url: "https://johndoe.dev"
    summary: "Computer engineering undergraduate focused on distributed systems and high-throughput backend services."

education:
    - institution: "University Name"
      location: "Istanbul, Turkey"
      degree: "B.S. in Computer Engineering"
      gpa: "3.80 / 4.00"
      start_date: "09/2023"
      end_date: "06/2027"
      coursework:
          - "Operating Systems"
          - "Distributed Systems"
          - "Database Systems"

skills:
    languages:
        - "Go"
        - "Python"
        - "C++"
        - "SQL"
    frameworks:
        - "FastAPI"
        - "gRPC"
        - "Gin"
    tools_and_systems:
        - "Docker"
        - "Kubernetes"
        - "Linux"
        - "Git"
    databases:
        - "PostgreSQL"
        - "Redis"

projects:
    - name: "Distributed Task Queue"
      stack:
          - "Go"
          - "Redis"
          - "Docker"
      team_size: 1
      link: "https://github.com/johndoe/task-queue"
      description: "Distributed task scheduler with exponential backoff and dead-letter queue support."
      highlights:
          - "Engineered Redis-backed worker pool processing 15,000 tasks/sec with under 5ms latency."
          - "Implemented at-least-once delivery guarantees and automatic retry with jitter."

experience:
    - role: "Software Engineering Intern"
      company: "Tech Company"
      location: "Istanbul, Turkey"
      employment_type: "Part-time"
      start_date: "06/2025"
      end_date: "09/2025"
      bullets:
          - "Designed REST APIs serving 2M daily requests using FastAPI and PostgreSQL."
          - "Reduced database query latency by 40% via composite indexing and query profiling."

languages:
    - name: "English"
      level: "Fluent"
    - name: "Turkish"
      level: "Native"
```

### 2. Validate Profile Syntax

```bash
cv validate candidate_profile.yaml
```

### 3. Generate Resume

```bash
# Generate an English resume
cv run candidate_profile.yaml output/ --lang en

# Generate a Turkish resume targeting specific location
cv run candidate_profile.yaml output/ --lang tr --location "Istanbul, Turkey" --jobs 15

# Verbose output with real-time logs
cv run candidate_profile.yaml output/ --verbose
```

The compiled PDF and LaTeX source files will be saved in your specified output directory (e.g., `output/John_Doe_Resume.pdf`).

---

## CLI Reference

### `cv run`

Executes the resume generation pipeline:

```bash
cv run <profile_path> <output_dir> [options]
```

| Option          | Flag   | Default            | Description                                      |
| :-------------- | :----- | :----------------- | :----------------------------------------------- |
| `--lang`        | `-l`   | `en`               | Resume output language (`en` or `tr`).           |
| `--jobs`        | `-j`   | `15`               | Number of job postings to analyze.               |
| `--location`    | `-loc` | `Istanbul, Turkey` | Target geographic job market.                    |
| `--ui-lang`     |        | Interactive        | CLI display language (`en` or `tr`).             |
| `--target-role` |        | Auto-detected      | Override target role title.                      |
| `--verbose`     | `-v`   | `False`            | Show detailed debug logs during execution.       |
| `--quiet`       | `-q`   | `False`            | Run in non-interactive mode with minimal output. |

### `cv validate`

Validates profile YAML structure and fields without making external network calls:

```bash
cv validate <profile_path>
```

### `cv clean`

Cleans up intermediate LaTeX build files (`.aux`, `.log`, `.tex`):

```bash
cv clean output/
cv clean output/ --all  # Also removes compiled PDF files
```

### `cv auth`

Manages Gemini API authentication:

```bash
cv auth set       # Store or update API key
cv auth show      # View active masked key
cv auth validate  # Test connection and model availability
cv auth status    # Check if a key is configured
cv auth delete    # Remove key from system keyring
```

---

## License

This project is licensed under the MIT License.
