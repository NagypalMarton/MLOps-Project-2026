# Data Dictionary — Hotel Booking Demand

This dataset captures real-world operational records extracted directly from the Property Management Systems (PMS) of two Portuguese hotel properties (an urban property in Lisbon and a resort in the Algarve) between July 2015 and August 2017. In the hospitality industry, reservation management and cancellation forecasting are critical components of yield management, overbooking strategy, and operational staffing. Each row represents a single reservation entity, documenting its end-to-end lifecycle from initial booking ingestion through intermediate modifications to final resolution (check-out, cancellation, or no-show).

## Overview

- **Dataset name / source / license:** Hotel Booking Demand / [Kaggle](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand) (Original: Antonio, de Almeida and Nunes, 2019, *Data in Brief*) / CC0: Public Domain
- **Rows / features:** 119,390 rows / 32 features (including target and leakage columns)
- **Task:** classification (supervised binary) — target column: `is_canceled`
- **Class balance:** 37.04% canceled (`1`) vs. 62.96% completed (`0`)

## Features

| Column | Meaning | Unit | Plausible range | Untrustworthy? (missing/sentinel) | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `hotel` | Operating hotel type | Categorical | `Resort Hotel`, `City Hotel` | No | Resort located in Algarve, City Hotel located in Lisbon. |
| `is_canceled` | Reservation cancellation flag | Binary flag | `0`, `1` | No | **Target column**. `1` = Canceled or No-Show, `0` = Checked out. |
| `lead_time` | Days between booking creation and arrival | Days | 0 – 737 | No | Highly right-skewed. Longer lead times strongly correlate with cancellations. |
| `arrival_date_year` | Year of scheduled arrival | Year | 2015 – 2017 | No | Bounded historical window: July 2015 – August 2017. |
| `arrival_date_month` | Month of scheduled arrival | Month name | `January` – `December` | No | Captures annual tourism seasonality. |
| `arrival_date_week_number` | Week number of arrival date | Integer | 1 – 53 | No | Seasonality feature. |
| `arrival_date_day_of_month` | Day of arrival date | Day | 1 – 31 | No | Intra-month booking dynamics. |
| `stays_in_weekend_nights` | Number of weekend nights (Sat / Sun) | Nights | 0 – 50 | No | Length of stay decomposition. |
| `stays_in_week_nights` | Number of week nights (Mon – Fri) | Nights | 0 – 50 | No | Length of stay decomposition. |
| `adults` | Number of adult guests | Count | 0 – 55 | Yes (0, extreme) | 180 rows contain 0 adults, children, and babies. Max 55 is an anomaly. |
| `children` | Number of child guests | Count | 0 – 10 | Yes (4 NaNs) | Contains 4 missing values in raw CSV. Defaults to 0. |
| `babies` | Number of baby guests | Count | 0 – 10 | No | Typically accompanied by special requests. |
| `meal` | Catered meal package type | Categorical | `BB`, `HB`, `FB`, `SC`, `Undefined` | Yes (`Undefined`) | `Undefined` is logged interchangeably with `SC` (Self Catering) in PMS. |
| `country` | Guest nationality | ISO 3166-1 | 177 distinct country codes | Yes (488 missing) | Encoded as string `'NULL'` or NaN in 488 rows. High cardinality. |
| `market_segment` | Distribution market segment designation | Categorical | `Direct`, `Online TA`, `Offline TA/TO`, `Corporate`, `Groups`, `Aviation`, `Complementary`, `Undefined` | Yes (`Undefined`) | 2 rows contain sentinel `'Undefined'`. |
| `distribution_channel` | Booking distribution channel | Categorical | `Direct`, `Corporate`, `TA/TO`, `GDS`, `Undefined` | Yes (`Undefined`) | 5 rows contain sentinel `'Undefined'`. |
| `is_repeated_guest` | Whether guest previously booked this hotel | Binary flag | `0`, `1` | No | Returning guests show significantly lower cancellation rates. |
| `previous_cancellations` | Prior cancellations by this customer profile | Count | 0 – 26 | No | Indicator of systematic booking/cancellation behavior. |
| `previous_bookings_not_canceled` | Prior completed bookings by this customer | Count | 0 – 72 | No | Measure of customer loyalty and reliability. |
| `reserved_room_type` | Code of reserved room category | Code | `A` – `L` | No | Anonymized room category at booking time. |
| `assigned_room_type` | Code of actually allocated room category | Code | `A` – `P` | **Yes (Leakage risk)** | Often finalized at check-in. Mismatches indicate overbooking/upgrade. |
| `booking_changes` | Number of reservation modifications | Count | 0 – 21 | No | Amendments made between booking date and arrival / cancellation. |
| `deposit_type` | Upfront deposit terms guarantee | Categorical | `No Deposit`, `Non Refund`, `Refundable` | No | Paradox: `Non Refund` has high cancellation due to group booking policies. |
| `agent` | ID of travel agency making reservation | ID code | 1 – 535 | Yes (16,340 missing) | String `'NULL'` / NaN indicates individual direct booking without agency. |
| `company` | ID of corporate entity paying reservation | ID code | 6 – 543 | Yes (112,593 missing) | 94.3% missing. Missing indicates non-corporate / private reservations. |
| `days_in_waiting_list` | Days reservation spent in PMS waiting queue | Days | 0 – 391 | No | 0 for the vast majority of instant bookings. |
| `customer_type` | Reservation relationship grouping | Categorical | `Transient`, `Transient-Party`, `Contract`, `Group` | No | Behavioral classification of the customer. |
| `adr` | Average Daily Rate (total room revenue / nights) | EUR / night | -6.38 – 5400.00 | Yes (negative, extreme) | Negative value (`-6.38`) is erroneous; `5400.00` is an extreme single outlier. |
| `required_car_parking_spaces` | Number of parking spots requested | Count | 0 – 8 | No | Strongly protective against cancellation (almost 0% cancellation if > 0). |
| `total_of_special_requests` | Number of special requests made | Count | 0 – 5 | No | High request volume correlates with high commitment / completed stay. |
| `reservation_status` | Final PMS booking resolution status | Categorical | `Check-Out`, `Canceled`, `No-Show` | **CRITICAL LEAKAGE** | **Must be dropped.** Directly determines target (`Canceled`/`No-Show` = 1, `Check-Out` = 0). |
| `reservation_status_date` | Date when final status was recorded | Date (YYYY-MM-DD) | 2014-10-17 – 2017-09-14 | **CRITICAL LEAKAGE** | **Must be dropped.** Recorded after booking, leaks cancellation date. |

## Known quirks / caveats

- **Critical Target Leakage:**  
  `reservation_status` and `reservation_status_date` directly encode the outcome. Including either will yield a trivial 100% accurate model that cannot function during production inference. Both must be removed before feature engineering.
- **Inference Timing / Temporal Leakage Risk:**  
  `assigned_room_type` is typically assigned by front-desk staff during check-in or shortly before arrival. If predicting strictly at reservation time, this field must either be excluded or treated as an arrival-stage update.
- **Sentinel & String-encoded Missing Values:**  
  Missing values are not all standard `NaN`s in the raw export:
  - `country`: 488 entries encoded as string `'NULL'` or `NaN`.
  - `agent`: 16,340 entries are `'NULL'` (sentinel meaning "Direct booking without an agent", not missing information).
  - `company`: 112,593 entries are `'NULL'` (~94.3% of the dataset; represents non-corporate private bookings).
  - `meal`: Contains string `'Undefined'` (functionally identical to `SC` / Self Catering).
- **Phantom Guest Rows (Data Quality Violations):**  
  180 rows contain `adults == 0` AND `children == 0` AND `babies == 0`. These represent test bookings, system glitches, or unallocated reservations and should be flagged or dropped during quality gates.
- **Outliers in Numerical Targets:**  
  - `adr`: One negative value (`-6.38`, likely an accounting adjustment/refund error) and an extreme outlier (`5400.00 EUR`). Plausible values range between 0 and 500 EUR.
  - `adults`: Contains records with up to 55 adults for a single room allocation.
- **Class Imbalance & Cost Asymmetry:**  
  37% of reservations cancel. Missing a cancellation (false negative) causes inventory spoilage, while a false alarm (false positive) risks overbooking friction. Evaluation must prioritize PR-AUC and Macro F1 over standard accuracy.