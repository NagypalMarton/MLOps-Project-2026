from __future__ import annotations

import pandera.pandas as pa
from pandera.typing import DataFrame, Series


class Ingestion(pa.DataFrameModel):
    """Contract for the raw PMS export before cleaning."""

    hotel: Series[str] = pa.Field(isin=["City Hotel", "Resort Hotel"])
    is_canceled: Series[int] = pa.Field(ge=0, le=1)
    lead_time: Series[int] = pa.Field(ge=0, le=1000)
    arrival_date_year: Series[int] = pa.Field(ge=2015, le=2030)
    arrival_date_month: Series[str] = pa.Field(
        isin=[
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December",
        ]
    )
    arrival_date_week_number: Series[int] = pa.Field(ge=1, le=53)
    arrival_date_day_of_month: Series[int] = pa.Field(ge=1, le=31)
    stays_in_weekend_nights: Series[int] = pa.Field(ge=0, le=365)
    stays_in_week_nights: Series[int] = pa.Field(ge=0, le=365)
    adults: Series[int] = pa.Field(ge=0, le=100)
    children: Series[float] = pa.Field(nullable=True, ge=0, le=20)
    babies: Series[int] = pa.Field(ge=0, le=20)
    meal: Series[str] = pa.Field(
        isin=["BB", "HB", "FB", "SC", "Undefined", "NULL"], nullable=True
    )
    country: Series[str] = pa.Field(nullable=True)
    market_segment: Series[str] = pa.Field(
        isin=[
            "Online TA", "Offline TA/TO", "Groups", "Direct", "Corporate",
            "Complementary", "Aviation", "Undefined", "NULL",
        ]
    )
    distribution_channel: Series[str] = pa.Field(
        isin=["TA/TO", "Direct", "Corporate", "GDS", "Undefined", "NULL"]
    )
    is_repeated_guest: Series[int] = pa.Field(ge=0, le=1)
    previous_cancellations: Series[int] = pa.Field(ge=0, le=100)
    previous_bookings_not_canceled: Series[int] = pa.Field(ge=0, le=500)
    reserved_room_type: Series[str] = pa.Field(nullable=True)
    assigned_room_type: Series[str] = pa.Field(nullable=True)
    booking_changes: Series[int] = pa.Field(ge=0, le=100)
    deposit_type: Series[str] = pa.Field(
        isin=["No Deposit", "Non Refund", "Refundable", "NULL"]
    )
    agent: Series[float] = pa.Field(nullable=True, ge=1, le=1000)
    company: Series[float] = pa.Field(nullable=True, ge=1, le=1000)
    days_in_waiting_list: Series[int] = pa.Field(ge=0, le=400)
    customer_type: Series[str] = pa.Field(
        isin=["Transient", "Transient-Party", "Contract", "Group", "NULL"]
    )
    # The raw export contains one -6.38 ADR, so the ingestion boundary accepts
    # that documented anomaly while still rejecting impossible magnitudes.
    adr: Series[float] = pa.Field(ge=-10, le=10000)
    required_car_parking_spaces: Series[int] = pa.Field(ge=0, le=20)
    total_of_special_requests: Series[int] = pa.Field(ge=0, le=20)
    reservation_status: Series[str] = pa.Field(
        isin=["Check-Out", "Canceled", "No-Show"]
    )
    reservation_status_date: Series[str] = pa.Field()

    class Config:
        strict = True


class ModelInput(pa.DataFrameModel):
    """Contract for cleaned rows passed to feature preprocessing."""

    hotel: Series[str] = pa.Field(isin=["City Hotel", "Resort Hotel"])
    is_canceled: Series[int] = pa.Field(ge=0, le=1)
    lead_time: Series[int] = pa.Field(ge=0, le=1000)
    arrival_date_year: Series[int] = pa.Field(ge=2015, le=2030)
    arrival_date_month: Series[str] = pa.Field(
        isin=[
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December",
        ]
    )
    arrival_date_week_number: Series[int] = pa.Field(ge=1, le=53)
    arrival_date_day_of_month: Series[int] = pa.Field(ge=1, le=31)
    stays_in_weekend_nights: Series[int] = pa.Field(ge=0, le=365)
    stays_in_week_nights: Series[int] = pa.Field(ge=0, le=365)
    adults: Series[int] = pa.Field(ge=0, le=100)
    children: Series[float] = pa.Field(ge=0, le=20)
    babies: Series[int] = pa.Field(ge=0, le=20)
    meal: Series[str] = pa.Field(
        isin=["BB", "HB", "FB", "SC", "Undefined", "NULL"]
    )
    country: Series[str] = pa.Field()
    market_segment: Series[str] = pa.Field(
        isin=[
            "Online TA", "Offline TA/TO", "Groups", "Direct", "Corporate",
            "Complementary", "Aviation", "Undefined", "NULL",
        ]
    )
    distribution_channel: Series[str] = pa.Field(
        isin=["TA/TO", "Direct", "Corporate", "GDS", "Undefined", "NULL"]
    )
    is_repeated_guest: Series[int] = pa.Field(ge=0, le=1)
    previous_cancellations: Series[int] = pa.Field(ge=0, le=100)
    previous_bookings_not_canceled: Series[int] = pa.Field(ge=0, le=500)
    reserved_room_type: Series[str] = pa.Field()
    assigned_room_type: Series[str] = pa.Field()
    booking_changes: Series[int] = pa.Field(ge=0, le=100)
    deposit_type: Series[str] = pa.Field(
        isin=["No Deposit", "Non Refund", "Refundable", "NULL"]
    )
    agent: Series[float] = pa.Field(ge=0, le=1000)
    company: Series[float] = pa.Field(ge=0, le=1000)
    days_in_waiting_list: Series[int] = pa.Field(ge=0, le=400)
    customer_type: Series[str] = pa.Field(
        isin=["Transient", "Transient-Party", "Contract", "Group", "NULL"]
    )
    # Model input is stricter because negative prices are not meaningful to the
    # classifier, even though the raw PMS export contains one such anomaly.
    adr: Series[float] = pa.Field(ge=0, le=10000)
    required_car_parking_spaces: Series[int] = pa.Field(ge=0, le=20)
    total_of_special_requests: Series[int] = pa.Field(ge=0, le=20)
    total_guests: Series[float] = pa.Field(gt=0, le=120)

    class Config:
        strict = True


def validate_ingestion(frame: DataFrame[Ingestion]) -> DataFrame[Ingestion]:
    return Ingestion.validate(frame)


def validate_model_input(frame: DataFrame[ModelInput]) -> DataFrame[ModelInput]:
    return ModelInput.validate(frame)
