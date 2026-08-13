#!/usr/bin/env bash
# ============================================================
#  deepseek-api-kit — OpenAI Proxy launcher
#  اجرای سرور محلی پروکسی سازگار با OpenAI
# ============================================================
set -euo pipefail

# مسیر پوشهٔ پروژه (همان پوشه‌ای که این اسکریپت در آن است)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"

# فعال‌سازی محیط مجازی
if [ ! -d ".venv" ]; then
    echo "⚠️  محیط مجازی (.venv) پیدا نشد. در حال ساخت..."
    python3 -m venv .venv
    ./.venv/bin/pip install --upgrade pip -q
    ./.venv/bin/pip install -r requirements.txt
fi
source .venv/bin/activate

# بررسی وجود فایل .env
if [ ! -f ".env" ]; then
    echo "⚠️  فایل .env پیدا نشد. از روی .env.example ساخته می‌شود."
    cp .env.example .env
    echo "   ⚠️  لطفاً کلید API را در فایل .env قرار دهید."
fi

echo "=============================================="
echo "  🚀 OpenAI Proxy در حال اجرا"
echo "  Base URL : http://${HOST}:${PORT}/v1"
echo "  Models   : http://${HOST}:${PORT}/v1/models"
echo "  Chat     : http://${HOST}:${PORT}/v1/chat/completions"
echo "  Docs     : http://${HOST}:${PORT}/docs"
echo "=============================================="
echo "  برای متوقف کردن: Ctrl+C"
echo ""

exec uvicorn openai_proxy.main:app --host "$HOST" --port "$PORT"
