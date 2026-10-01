import csv
import io
import logging
import os
from datetime import date as DateType
from decimal import Decimal
from pathlib import Path
from typing import Any, Generator

import plaid
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from plaid.api import plaid_api
from plaid.exceptions import ApiException
from plaid.model.country_code import CountryCode
from plaid.model.item_public_token_exchange_request import (
    ItemPublicTokenExchangeRequest,
)
from plaid.model.link_token_create_request import LinkTokenCreateRequest
from plaid.model.link_token_create_request_user import LinkTokenCreateRequestUser
from plaid.model.products import Products
from plaid.model.transactions_sync_request import TransactionsSyncRequest
from sqlalchemy import Boolean, Date, ForeignKey, Integer, String, create_engine, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
DATABASE_PATH = DATA_DIR / "raptorbudget.sqlite3"
load_dotenv(BASE_DIR / ".env")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="RaptorBudget")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

engine = create_engine(
    f"sqlite:///{DATABASE_PATH.as_posix()}",
    connect_args={"check_same_thread": False},
)
SessionLocal = sessionmaker(bind=engine, autoflush=False)


class Base(DeclarativeBase):
    pass


class PlaidItem(Base):
    __tablename__ = "plaid_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    item_id: Mapped[str] = mapped_column(String, unique=True)
    access_token: Mapped[str] = mapped_column(String)
    cursor: Mapped[str | None] = mapped_column(String, nullable=True)


class Bucket(Base):
    __tablename__ = "buckets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(60), unique=True, index=True)
    allocations: Mapped[list["Allocation"]] = relationship(back_populates="bucket")


class Transaction(Base):
    __tablename__ = "transactions"

    transaction_id: Mapped[str] = mapped_column(String, primary_key=True)
    account_id: Mapped[str] = mapped_column(String, index=True)
    date: Mapped[DateType] = mapped_column(Date, index=True)
    authorized_date: Mapped[DateType | None] = mapped_column(Date, nullable=True)
    name: Mapped[str] = mapped_column(String)
    merchant_name: Mapped[str | None] = mapped_column(String, nullable=True)
    amount_cents: Mapped[int] = mapped_column(Integer)
    currency: Mapped[str] = mapped_column(String(3), default="USD")
    pending: Mapped[bool] = mapped_column(Boolean, default=False)
    category: Mapped[str | None] = mapped_column(String, nullable=True)
    allocations: Mapped[list["Allocation"]] = relationship(
        back_populates="transaction", cascade="all, delete-orphan"
    )


class Allocation(Base):
    __tablename__ = "allocations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    transaction_id: Mapped[str] = mapped_column(
        ForeignKey("transactions.transaction_id", ondelete="CASCADE"), index=True
    )
    bucket_id: Mapped[int] = mapped_column(ForeignKey("buckets.id"))
    amount_cents: Mapped[int] = mapped_column(Integer)
    transaction: Mapped[Transaction] = relationship(back_populates="allocations")
    bucket: Mapped[Bucket] = relationship(back_populates="allocations")


class BucketCreate(BaseModel):
    name: str = Field(min_length=1, max_length=60)


class AllocationInput(BaseModel):
    bucket_id: int
    amount_cents: int = Field(gt=0)


class AllocationUpdate(BaseModel):
    allocations: list[AllocationInput]


Base.metadata.create_all(engine)
with SessionLocal() as session:
    if session.scalar(select(func.count()).select_from(Bucket)) == 0:
        session.add_all(
            Bucket(name=name)
            for name in ("Housing", "Food", "Transport", "Bills", "Savings", "Income", "Other")
        )
        session.commit()


def get_db() -> Generator[Session, None, None]:
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def get_plaid_client() -> plaid_api.PlaidApi:
    client_id = os.getenv("PLAID_CLIENT_ID")
    secret = os.getenv("PLAID_SECRET")
    if not client_id or not secret:
        raise HTTPException(
            status_code=500,
            detail="Set PLAID_CLIENT_ID and PLAID_SECRET in your local .env file.",
        )

    configuration = plaid.Configuration(
        host=plaid.Environment.Sandbox,
        api_key={"clientId": client_id, "secret": secret},
    )
    return plaid_api.PlaidApi(plaid.ApiClient(configuration))


def parse_optional_date(value: str | DateType | None) -> DateType | None:
    if value is None:
        return None
    if isinstance(value, DateType):
        return value
    return DateType.fromisoformat(value)


def dollars_to_cents(amount: Any) -> int:
    return int(Decimal(str(amount)) * 100)


def upsert_transaction(db: Session, item: dict[str, Any]) -> Transaction:
    transaction_id = item["transaction_id"]
    amount_cents = dollars_to_cents(item["amount"])
    transaction = db.get(Transaction, transaction_id)
    if transaction is None:
        transaction = Transaction(transaction_id=transaction_id, account_id=item["account_id"], date=DateType.today(), name="")
        db.add(transaction)
    elif transaction.amount_cents != amount_cents:
        transaction.allocations.clear()

    transaction.account_id = item["account_id"]
    transaction.date = parse_optional_date(item.get("date")) or DateType.today()
    transaction.authorized_date = parse_optional_date(item.get("authorized_date"))
    transaction.name = item.get("name") or item.get("merchant_name") or "Transaction"
    transaction.merchant_name = item.get("merchant_name")
    transaction.amount_cents = amount_cents
    transaction.currency = item.get("iso_currency_code") or "USD"
    transaction.pending = bool(item.get("pending", False))
    pfc = item.get("personal_finance_category") or {}
    transaction.category = pfc.get("primary") or ((item.get("category") or [None])[0])
    return transaction


def transaction_json(transaction: Transaction) -> dict[str, Any]:
    allocated = sum(allocation.amount_cents for allocation in transaction.allocations)
    return {
        "transaction_id": transaction.transaction_id,
        "date": transaction.date.isoformat(),
        "authorized_date": transaction.authorized_date.isoformat() if transaction.authorized_date else None,
        "name": transaction.name,
        "merchant_name": transaction.merchant_name,
        "amount_cents": transaction.amount_cents,
        "currency": transaction.currency,
        "pending": transaction.pending,
        "category": transaction.category,
        "allocations": [
            {
                "bucket_id": allocation.bucket_id,
                "bucket_name": allocation.bucket.name,
                "amount_cents": abs(allocation.amount_cents),
            }
            for allocation in transaction.allocations
        ],
        "allocated_cents": abs(allocated),
        "unassigned_cents": max(0, abs(transaction.amount_cents) - abs(allocated)),
    }


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/api/status")
def status(db: Session = Depends(get_db)) -> dict[str, bool]:
    return {"connected": db.get(PlaidItem, 1) is not None}


@app.post("/api/link-token")
def create_link_token() -> dict[str, str]:
    request = LinkTokenCreateRequest(
        client_name="RaptorBudget",
        language="en",
        country_codes=[CountryCode("US")],
        user=LinkTokenCreateRequestUser(client_user_id="raptor-budget-local-user"),
        products=[Products("transactions")],
    )
    try:
        response = get_plaid_client().link_token_create(request)
    except ApiException as error:
        logger.exception("Plaid link-token creation failed")
        raise HTTPException(status_code=502, detail="Plaid could not create a Link token.") from error
    return {"link_token": response.link_token}


@app.post("/api/exchange-public-token")
def exchange_public_token(
    payload: dict[str, str], db: Session = Depends(get_db)
) -> dict[str, str]:
    public_token = payload.get("public_token")
    if not public_token:
        raise HTTPException(status_code=400, detail="Missing public_token.")

    try:
        response = get_plaid_client().item_public_token_exchange(
            ItemPublicTokenExchangeRequest(public_token=public_token)
        )
    except ApiException as error:
        logger.exception("Plaid public-token exchange failed")
        raise HTTPException(status_code=502, detail="Plaid could not exchange the Link token.") from error

    item = db.get(PlaidItem, 1)
    if item is None:
        item = PlaidItem(id=1, item_id=response.item_id, access_token=response.access_token)
        db.add(item)
    else:
        item.item_id = response.item_id
        item.access_token = response.access_token
        item.cursor = None
    db.commit()
    return {"item_id": response.item_id}


@app.post("/api/transactions/sync")
def sync_transactions(db: Session = Depends(get_db)) -> dict[str, int]:
    item = db.get(PlaidItem, 1)
    if item is None:
        raise HTTPException(status_code=400, detail="Connect a Sandbox account first.")

    cursor = item.cursor
    counts = {"added": 0, "modified": 0, "removed": 0}
    try:
        has_more = True
        while has_more:
            request_data: dict[str, str] = {"access_token": item.access_token}
            if cursor is not None:
                request_data["cursor"] = cursor
            response = get_plaid_client().transactions_sync(
                TransactionsSyncRequest(**request_data)
            )
            for transaction in response.added:
                upsert_transaction(db, transaction.to_dict())
                counts["added"] += 1
            for transaction in response.modified:
                upsert_transaction(db, transaction.to_dict())
                counts["modified"] += 1
            for removed in response.removed:
                transaction_id = removed.transaction_id
                db.query(Allocation).filter_by(transaction_id=transaction_id).delete()
                db.query(Transaction).filter_by(transaction_id=transaction_id).delete()
                counts["removed"] += 1
            cursor = response.next_cursor
            has_more = response.has_more
    except ApiException as error:
        db.rollback()
        logger.exception("Plaid transaction sync failed")
        raise HTTPException(status_code=502, detail="Plaid could not sync transactions.") from error

    item.cursor = cursor
    db.commit()
    return counts


@app.get("/api/transactions")
def list_transactions(db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    transactions = db.scalars(
        select(Transaction).order_by(Transaction.date.desc(), Transaction.name)
    ).all()
    return [transaction_json(transaction) for transaction in transactions]


@app.get("/api/buckets")
def list_buckets(db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    buckets = db.scalars(select(Bucket).order_by(Bucket.name)).all()
    balances = dict(
        db.execute(
            select(Allocation.bucket_id, func.sum(Allocation.amount_cents)).group_by(
                Allocation.bucket_id
            )
        ).all()
    )
    return [
        {"id": bucket.id, "name": bucket.name, "balance_cents": balances.get(bucket.id, 0) or 0}
        for bucket in buckets
    ]


@app.post("/api/buckets", status_code=201)
def create_bucket(payload: BucketCreate, db: Session = Depends(get_db)) -> dict[str, Any]:
    name = payload.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Bucket name cannot be blank.")
    bucket = Bucket(name=name)
    db.add(bucket)
    try:
        db.commit()
    except IntegrityError as error:
        db.rollback()
        raise HTTPException(status_code=409, detail="That bucket already exists.") from error
    db.refresh(bucket)
    return {"id": bucket.id, "name": bucket.name, "balance_cents": 0}


@app.put("/api/transactions/{transaction_id}/allocations")
def update_allocations(
    transaction_id: str,
    payload: AllocationUpdate,
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    transaction = db.get(Transaction, transaction_id)
    if transaction is None:
        raise HTTPException(status_code=404, detail="Transaction not found.")
    if sum(row.amount_cents for row in payload.allocations) != abs(transaction.amount_cents):
        raise HTTPException(
            status_code=422,
            detail="Split amounts must add up exactly to the transaction amount.",
        )
    bucket_ids = {row.bucket_id for row in payload.allocations}
    known_ids = set(db.scalars(select(Bucket.id).where(Bucket.id.in_(bucket_ids))).all()) if bucket_ids else set()
    if bucket_ids != known_ids:
        raise HTTPException(status_code=422, detail="Choose an existing bucket for every split.")

    transaction.allocations.clear()
    direction = -1 if transaction.amount_cents > 0 else 1
    for row in payload.allocations:
        transaction.allocations.append(
            Allocation(bucket_id=row.bucket_id, amount_cents=direction * row.amount_cents)
        )
    db.commit()
    return transaction_json(transaction)


@app.get("/api/export.csv")
def export_csv(db: Session = Depends(get_db)) -> StreamingResponse:
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(
        ["date", "merchant", "description", "amount", "currency", "pending", "allocations", "unassigned_amount"]
    )
    transactions = db.scalars(select(Transaction).order_by(Transaction.date.desc())).all()
    for transaction in transactions:
        allocated = sum(allocation.amount_cents for allocation in transaction.allocations)
        allocation_text = "; ".join(
            f"{allocation.bucket.name}: {allocation.amount_cents / 100:.2f}"
            for allocation in transaction.allocations
        )
        unassigned = abs(transaction.amount_cents) - abs(allocated)
        writer.writerow(
            [
                transaction.date.isoformat(),
                transaction.merchant_name or transaction.name,
                transaction.name,
                f"{transaction.amount_cents / 100:.2f}",
                transaction.currency,
                transaction.pending,
                allocation_text,
                f"{unassigned / 100:.2f}" if unassigned else "0.00",
            ]
        )
    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=raptorbudget-transactions.csv"},
    )