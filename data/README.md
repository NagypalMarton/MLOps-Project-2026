# data/

The project uses `hotel_bookings.csv`, tracked by DVC through
`hotel_bookings.csv.dvc`. The raw CSV is restored with `dvc pull`; it is not
committed to Git. The backup and archive files are retained as dataset
artifacts, but are not used by the pipeline.
