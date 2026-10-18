
 import re
from datetime import date

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

app = FastAPI(title="Invoice Extractor")


class InvoiceRequest(BaseModel):
    text: str = Field(min_length=1, max_length=50000)


class InvoiceResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    vendor: str
    amount: float
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")


@app.get("/")
def root():
    return {"status": "running", "endpoint": "POST /extract"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/extract", response_model=InvoiceResponse)
def extract(request: InvoiceRequest):
    text = request.text.strip()
    if not text:
        from fastapi import HTTPException
        raise HTTPException(status_code=422, detail="Empty invoice text")

    # Extract ISO dates, preferring a date near a due-date label.
    dates = re.findall(r"\b2026-\d{2}-\d{2}\b", text)
    due = re.search(
        r"(?:payment\s+)?due\s+date\s*[:#=-]?\s*"
        r"(2026-\d{2}-\d{2})",
        text, re.I
    )
    if not due:
        due = re.search(
            r"(?:due\s+by|payable\s+by|payment\s+due)"
            r"[^0-9]{0,30}(2026-\d{2}-\d{2})",
            text, re.I
        )
    invoice_date = due.group(1) if due else (
        dates[-1] if dates else None
    )
    if not invoice_date:
        from fastapi import HTTPException
        raise HTTPException(422, "Due date not found")

    try:
        date.fromisoformat(invoice_date)
    except ValueError:
        from fastapi import HTTPException
        raise HTTPException(422, "Invalid date")

    # Find currency.
    currencies = [
        (r"\bUSD\b|US dollars?|\$", "USD"),
        (r"\bEUR\b|euros?|€", "EUR"),
        (r"\bGBP\b|GB pounds?|£", "GBP"),
    ]
    currency = next(
        (code for pattern, code in currencies
         if re.search(pattern, text, re.I)),
        None
    )
    if not currency:
        from fastapi import HTTPException
        raise HTTPException(422, "Currency not found")

    # Prefer the amount associated with total due.
    num = r"([0-9][0-9,]*(?:\.\d{1,2})?)"
    amount_patterns = [
        rf"(?:total\s+amount\s+due|total\s+due|amount\s+due|"
        rf"balance\s+due|grand\s+total|total\s+payable|"
        rf"invoice\s+total)\s*[:=-]?\s*"
        rf"(?:USD|EUR|GBP|\$|€|£)?\s*{num}",
        rf"(?:USD|EUR|GBP|\$|€|£)\s*{num}",
    ]
    amount = None
    for pattern in amount_patterns:
        matches = list(re.finditer(pattern, text, re.I))
        if matches:
            amount = float(matches[-1].group(1).replace(",", ""))
            break

    if amount is None:
        values = re.findall(
            r"(?<![\w.-])([0-9][0-9,]*\.\d{2})(?!\w)", text
        )
        if values:
            amount = float(values[-1].replace(",", ""))

    if amount is None:
        from fastapi import HTTPException
        raise HTTPException(422, "Total amount not found")

    # Prefer a labeled vendor.
    vendor = None
    for pattern in [
        r"(?:vendor|supplier|seller|billed\s+by|issued\s+by|"
        r"company|from)\s*[:=-]\s*([^\n\r]+)",
        r"\b([A-Z][A-Za-z0-9&'-]*(?:\s+[A-Z0-9][A-Za-z0-9&'.,-]*){0,5}"
        r"\s+(?:Industries|Limited|Ltd\.?|Inc\.?|LLC|"
        r"Corporation|Corp\.?|Company|Co\.?))\b",
    ]:
        matches = list(re.finditer(pattern, text, re.I | re.M))
        if matches:
            vendor = matches[0].group(1).strip()
            vendor = re.split(
                r"\s{2,}|\s+\|\s+|\s+(?:invoice|date|amount)\s*:",
                vendor, maxsplit=1, flags=re.I
            )[0].strip(" \t,;:-")
            if vendor:
                break

    if not vendor:
        from fastapi import HTTPException
        raise HTTPException(422, "Vendor not found")

    return InvoiceResponse(
        vendor=vendor,
        amount=amount,
        currency=currency,
        date=invoice_date,
    )
