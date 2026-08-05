import polars as pl

# 1. Define Binance public trade data columns with explicit data types
# This acts as the schema parser for the headerless CSV file
CSV_SCHEMA = {
    "trade_id": pl.Int64,  # Unique transaction identifier
    "price": pl.Float64,  # Execution price in USDT
    "qty": pl.Float64,  # Amount of BTC traded
    "quote_qty": pl.Float64,  # Total value of the trade in USDT (price * qty)
    "time": pl.Int64,  # Unix timestamp in milliseconds or microseconds
    "is_buyer_maker": pl.Boolean,  # True = Sell pressure, False = Buy pressure
    "is_best_match": pl.Boolean,  # Legacy routing field (can be ignored)
}

FILE_PATH = "../data/BTCUSDT-trades-2026-05.csv"


def get_lazy_feeder(path: str):
    """
    Initializes a lazy execution graph using Polars out-of-core capabilities.
    """
    return (
        pl.scan_csv(path, has_header=False, schema=CSV_SCHEMA)
        # Cast the time integer column straight into a microsecond Datetime object
        .with_columns(pl.col("time").cast(pl.Datetime("us")))
    )


def stream_hourly_chunks(lazy_df, chunk_hours: int = 1):
    """
    Streams data out-of-core by processing it in time-based chunks.
    This guarantees your system RAM stays stable.
    """
    # 1. Collect only the min/max time first
    time_bounds = lazy_df.select(
        [pl.col("time").min().alias("min_time"), pl.col("time").max().alias("max_time")]
    ).collect()

    start_time = time_bounds["min_time"][0]
    end_time = time_bounds["max_time"][0]

    print(f"Data ranges from {start_time} to {end_time}")

    current_start = start_time
    # Advance window by our chunk size (e.g., 1 hour)
    step = pl.duration(hours=chunk_hours)

    while current_start < end_time:
        current_end = current_start + step

        # 2. Define the chunk via filter expressions (Lazy evaluation)
        chunk_lazy = lazy_df.filter(
            (pl.col("time") >= current_start) & (pl.col("time") < current_end)
        )

        # 3. Trigger execution only for this specific chunk
        chunk_df = chunk_lazy.collect()

        if len(chunk_df) > 0:
            yield chunk_df

        current_start = current_end


# --- Validation Validation Test ---
if __name__ == "__main__":
    print("Initializing lazy data connection...")
    lazy_pipeline = get_lazy_feeder(FILE_PATH)

    print("Starting out-of-core streaming test...")
    # Test stream with 1-hour chunks
    feeder = stream_hourly_chunks(lazy_pipeline, chunk_hours=1)

    # Grab just the first 1-hour chunk to validate
    try:
        first_hour_chunk = next(feeder)
        print("✅ Success! First 1-hour chunk collected.")
        print(f"Rows in this chunk: {len(first_hour_chunk):,}")
        print(first_hour_chunk.head(5))
    except StopIteration:
        print("❌ No data found or file path is incorrect.")
