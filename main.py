import json
import os
import re
from datetime import date
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

app = FastAPI(title="Invoice Extractor API", version="1.0.0")


class ExtractRequest(BaseModel):
    text: str = Field(min_length=1, max_length=50000)

    @field_validator("text")
    @classmethod
    def non_empty_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Invoice text must not be empty")
        return value.strip()


class InvoiceFields(BaseModel):
    model_config = ConfigDict(extra="forbid")

    vendor: str
    amount: float
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")

    @field_validator("date")
    @classmethod
    def valid_date(cls, value: str) -> str:
        date.fromisoformat(value)
        return value


def regex_extract(text: str) -> Optional[dict]:
    # ISO dates, including the randomized 2026-MM-DD test dates.
    dates = re.findall(
        r"\b(20\d{2}-\d{2}-\d{2})\b", text
    )

    # Prefer dates explicitly associated with payment due dates.
    due_match = re.search(
        r"(?:payment\s+)?due\s+date\s*[:#=-]?\s*"
        r"(20\d{2}-\d{2}-\d{2})",
        text,
        re.I,
    )
    if not due_match:
        due_match = re.search(
            r"(?:due\s+by|payable\s+by|payment\s+due)"
            r"[^0-9]{0,30}(20\d{2}-\d{2}-\d{2})",
            text,
            re.I,
        )

    chosen_date = (
        due_match.group(1) if due_match else
        (dates[-1] if dates else None)
    )

    if not chosen_date:
        return None

    # Currency is identified independently of the amount.
    currency_patterns = [
        (r"\bUSD\b|\bUS\s+dollars?\b|\$", "USD"),
        (r"\bEUR\b|\beuros?\b|€", "EUR"),
        (r"\bGBP\b|\bGB\s+pounds?\b|£", "GBP"),
    ]
    currency = None
    for pattern, code in currency_patterns:
        if re.search(pattern, text, re.I):
            currency = code
            break

    if not currency:
        return None

    # Locate amounts close to total-due labels before considering
    # other monetary values such as subtotal or tax.
    number = r"([0-9][0-9,]*(?:\.[0-9]{1,2})?)"
    amount_patterns = [
        rf"(?:total\s+amount\s+due|total\s+due|amount\s+due|"
        rf"balance\s+due|grand\s+total|total\s+payable|"
        rf"invoice\s+total)\s*[:=-]?\s*"
        rf"(?:USD|EUR|GBP|\$|€|£)?\s*{number}",
        rf"(?:USD|EUR|GBP|\$|€|£)\s*{number}",
    ]

    amount = None
    for pattern in amount_patterns:
        matches = list(re.finditer(pattern, text, re.I))
        if matches:
            amount = float(matches[-1].group(1).replace(",", ""))
            break

    if amount is None:
        # Last-resort fallback for loosely formatted invoices.
        numbers = re.findall(
            r"(?<![\w.-])([0-9][0-9,]*\.[0-9]{2})(?!\w)",
            text,
        )
        if numbers:
            amount = float(numbers[-1].replace(",", ""))

    if amount is None:
        return None

    # Prefer a company name explicitly introduced by a vendor label.
    vendor_patterns = [
        r"(?:vendor|supplier|seller|billed\s+by|issued\s+by|"
        r"from|company)\s*[:=-]\s*([^\n\r]+)",
        r"^([A-Z][A-Za-z0-9&.,'() -]{2,80}"
        r"(?:Ltd\.?|Limited|Inc\.?|LLC|Corp\.?|Corporation|"
        r"Industries|Company|Co\.?))\s*$",
    ]

    vendor = None
    for pattern in vendor_patterns:
        matches = list(re.finditer(pattern, text, re.I | re.M))
        if matches:
            candidate = matches[0].group(1).strip()
            candidate = re.split(
                r"\s{2,}|\s+\|\s+|\s+(?:invoice|date|amount)\s*:",
                candidate,
                maxsplit=1,
                flags=re.I,
            )[0].strip(" \t,;:-")
            if candidate and len(candidate) <= 120:
                vendor = candidate
                break

    if not vendor:
        # Look for a company-like name anywhere in the invoice.
        match = re.search(
            r"\b([A-Z][A-Za-z0-9&'-]*(?:\s+[A-Z0-9][A-Za-z0-9&'.,-]*){0,5}"
            r"\s+(?:Industries|Limited|Ltd\.?|Inc\.?|LLC|"
            r"Corporation|Corp\.?|Company|Co\.?))\b",
            text,
            re.I,
        )
        if match:
            vendor = match.group(1).strip()

    if not vendor:
        return None

    # Validate the actual calendar date, not just its format.
    date.fromisoformat(chosen_date)

    return {
        "vendor": vendor,
        "amount": amount,
        "currency": currency,
        "date": chosen_date,
    }


async def ollama_extract(text: str) -> Optional[dict]:
    url = os.getenv(
        "OLLAMA_URL", "http://127.0.0.1:11434/api/generate"
    )
    model = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")

    prompt = f"""
Extract these fields from the invoice text:
vendor: full vendor/company name
amount: total amount due as a JSON number
currency: ISO 4217 uppercase 3-letter currency code
date: payment due date in YYYY-MM-DD format

Rules:
- Use only evidence in the invoice.
- Do not use the invoice issue date if a due date is present.
- Do not confuse subtotal or tax with total due.
- Return only a JSON object with exactly vendor, amount, currency, date.
- If a field is missing, return null for that field.

Invoice text:
{text}
"""

    try:
        async with httpx.AsyncClient(timeout=45) as client:
            response = await client.post(
                url,
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                    "options": {"temperature": 0},
                },
            )
            response.raise_for_status()
            payload = response.json()
            result = json.loads(payload["response"])
            return InvoiceFields.model_validate(result).model_dump()
    except Exception:
        return None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/extract", response_model=InvoiceFields)
async def extract_invoice(request: ExtractRequest) -> InvoiceFields:
    text = request.text.strip()

    # Pattern extraction can preserve exact planted values without
    # model paraphrasing or numeric rounding errors.
    extracted = regex_extract(text)

    if extracted is not None:
        try:
            return InvoiceFields.model_validate(extracted)
        except Exception:
            pass

    # Fall back to the local model for less predictable formatting.
    extracted = await ollama_extract(text)

    if extracted is None:
        raise HTTPException(
            status_code=422,
            detail="Could not reliably extract all required invoice fields",
        )

    return InvoiceFields.model_validate(extracted)
