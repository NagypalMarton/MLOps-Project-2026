# Project Proposal: Hotel Booking Cancellation Prediction

## 1. Dataset

**Hotel Booking Demand** - This dataset captures real-world hotel reservation records from two Portuguese hotels (a city hotel in Lisbon and a resort hotel in Algarve) between July 2015 and August 2017. Each row represents a single reservation entity documenting its lifecycle from booking to final resolution.

- **Source**: [Kaggle Hotel Booking Demand Dataset](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand) 
- **License**: CC0: Public Domain
- **Size**: 119,390 rows × 32 features (including target and leakage columns)
- **Target Variable**: `is_canceled` (binary: 0 = not canceled/checked out, 1 = canceled/no-show)

**Mini Data Dictionary** (selected features):
| Feature | Meaning | Unit/Type | Range | Notes |
|---------|---------|-----------|-------|-------|
| `hotel` | Hotel type | Categorical | Resort Hotel, City Hotel | Two distinct properties |
| `lead_time` | Days between booking and arrival | Integer (days) | 0-737 | Right-skewed; higher values correlate with cancellation |
| `stays_in_weekend_nights` | Weekend night stay length | Integer (nights) | 0-50 | Component of total stay |
| `stays_in_week_nights` | Weeknight stay length | Integer (nights) | 0-50 | Component of total stay |
| `adults` | Adult guest count | Integer | 0-55 | 180 rows have 0 adults/anomaly up to 55 |
| `children` | Child guest count | Integer | 0-10 | 4 missing values in raw data |
| `babies` | Baby guest count | Integer | 0-10 | Typically associated with special requests |
| `meal` | Meal package | Categorical | BB, HB, FB, SC, Undefined | Board basis selected |
| `country` | Guest nationality | ISO 3166-1 code | 177 countries | 488 missing (NULL/NaN) |
| `market_segment` | Booking segment | Categorical | Direct, Online TA, Corporate, etc. | Channel through which booking was made |
| `distribution_channel` | Booking channel | Categorical | Direct, Corporate, TA/TO, GDS, Undefined | How booking was made |
| `is_repeated_guest` | Returning guest flag | Binary | 0, 1 | History of previous stays |
| `previous_cancellations` | Past cancellations by guest | Integer | 0-26 | Behavioral indicator |
| `previous_bookings_not_canceled` | Past completed stays | Integer | 0-72 | Loyalty/reliability measure |
| `reserved_room_type` | Booked room category | Code | A-L | Anonymized room type at booking |
| `assigned_room_type` | Assigned room category | Code | A-P | **Leakage risk** - often final at check-in |
| `booking_changes` | Reservation modifications | Integer | 0-21 | Changes between booking and arrival |
| `deposit_type` | Deposit terms | Categorical | No Deposit, Non Refund, Refundable | Payment guarantee type |
| `agent` | Travel agency ID | Integer ID | 1-535 | 16,340 missing (direct bookings) |
| `company` | Corporate entity ID | Integer ID | 6-543 | 112,593 missing (mostly private) |
| `days_in_waiting_list` | Days on waiting list | Integer | 0-391 | 0 for most instant bookings |
| `customer_type` | Customer segment | Categorical | Transient, Transient-Party, Contract, Group | Behavioral grouping |
| `adr` | Average Daily Rate | EUR/night | -6.38-5400.00 | **Issue**: one negative value (-6.38), one extreme outlier (5400.00) |
| `required_car_parking_spaces` | Parking spots requested | Integer | 0-8 | Strong predictor - >0 spaces nearly eliminates cancellation |
| `total_of_special_requests` | Special request count | Integer | 0-5 | Higher commitment indicator |
| `reservation_status` | Final hotel status | Categorical | Check-Out, Canceled, No-Show | **CRITICAL LEAKAGE** - must be removed |
| `reservation_status_date` | Date status recorded | Date (YYYY-MM-DD) | 2014-10-17 to 2017-09-14 | **CRITICAL LEAKAGE** - post-booking date |

*Note: The leakage columns `reservation_status` and `reservation_status_date` will be excluded from modeling as they directly determine the target.*

## 2. Prediction Task

**Target**: Predicting hotel reservation cancellations (`is_canceled` column)

**Task Type**: Binary classification (supervised learning)

**Business Value**: Accurate cancellation prediction enables hotels to:
- Optimize overbooking strategies to maximize revenue
- Improve operational staffing and resource allocation  
- Enhance yield management through dynamic pricing
- Reduce losses from last-minute cancellations
- Better manage inventory and room assignments

This is a classic hospitality industry problem with direct financial impact. Hotels routinely sell more rooms than physically available (overbooking) based on predicted no-shows/cancellations. Improved prediction accuracy translates directly to increased revenue and reduced operational waste.

## 3. Suitability Check

**Memory & Training Performance**: 
- Dataset size: ~119K rows × 30 features (after removing leakage columns)
- Memory footprint: Well under 100MB when loaded as pandas DataFrame
- Training time: <10 seconds with scikit-learn LogisticRegression on mid-range laptop
- Verified: Loading and basic preprocessing completes instantly in Python

**Known Limitations/Qwirks**:
1. **Class Imbalance**: 37.04% cancellations vs 62.96% completed stays - moderate imbalance requiring careful metric selection
2. **Data Quality Issues**: 
   - Missing values in `children` (4 NaNs), `country` (488 NULL/NaN), `agent` (16,340 missing), `company` (112,593 missing)
   - Erroneous values: negative ADR (-6.38), extreme ADR outlier (5400.00)
   - Anomalies: 180 rows with 0 adults/children/babies, adult count up to 55
3. **Temporal Component**: Data spans July 2015-August 2017 - potential for concept drift over time
4. **Feature Leakage Risk**: Must carefully exclude `reservation_status` and `reservation_status_date` as they directly encode the target

## 4. Baseline Idea

**Initial Model**: Logistic Regression with L2 regularization
- **Why**: Interpretable coefficients, fast training, strong baseline for binary classification, handles mixed feature types after encoding
- **Preprocessing Plan**: 
  - Remove leakage columns (`reservation_status`, `reservation_status_date`)
  - Handle missing values appropriately (median/mode imputation or indicator variables)
  - Encode categorical variables (one-hot or ordinal based on cardinality)
  - Scale numeric features (StandardScaler)
  - Address outliers in `adr` and `lead_time` (winsorizing or robust scaling)

**Primary Metric**: **F1-score (weighted average)**
- **Why**: 
  - Accounts for the moderate class imbalance (37% cancellation rate)
  - Balances precision and recall - both important for business impact
  - False negatives (missed cancellations) lead to overbooking costs
  - False positives (incorrectly predicting cancellation) lead to lost revenue from unsold rooms
  - More informative than accuracy alone for imbalanced datasets
  - Will monitor precision-recall curve and ROC-AUC for secondary insights

**Alternative Models to Explore**: 
- Random Forest (handles non-linearities, feature interactions)
- XGBoost/LightGBM (gradient boosting strengths)
- Neural Networks (if feature engineering reveals complex patterns)

This proposal establishes a feasible, impactful ML project that satisfies all constraints: tabular supervised learning dataset, appropriate scale, single clear target, openly licensed, and readily reproducible via CSV commit. The hotel cancellation prediction task offers clear business relevance while providing ample opportunity to apply MLOps practices across the semester.