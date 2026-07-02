import polars as pl
import numpy as np
import bipower_core # type: ignore
from data_feeder import get_lazy_feeder, stream_hourly_chunks, FILE_PATH

def build_training_matrix(df_raw: pl.DataFrame):
    # 1. Resample irregular ticks into uniform 1-second bars
    # This automatically provides our 1-second stride engine
    df_1s = (
        df_raw.sort("time")
        .group_by_dynamic("time", every="1s")
        .agg([
            pl.col("price").last().forward_fill(),
            pl.col("qty").sum(),
            # True if the majority of trades in this second were sell pressure
            (pl.col("is_buyer_maker").cast(pl.Int32).sum() > (pl.len() / 2)).alias("is_buyer_maker")
        ])
        .drop_nulls()
    )

    # 2. Create the 1-minute forward target y
    # 1 if the price goes up in the next 60 seconds, 0 otherwise
    df_1s = df_1s.with_columns(
        (pl.col("price").shift(-60) > pl.col("price")).cast(pl.Int8).alias("target_y")
    )

    # 3. Extract flat arrays for the C++ engine
    prices = df_1s["price"].to_numpy().astype(np.float64)
    qtys = df_1s["qty"].to_numpy().astype(np.float64)
    maker = df_1s["is_buyer_maker"].to_numpy().astype(bool)
    targets = df_1s["target_y"].to_numpy()

    # 4. Calculate 5-minute lookback (300 seconds)
    window_size = 300
    metrics = bipower_core.calculate_rolling_metrics(prices, qtys, maker, window_size)

    # 5. Align X features and y targets
    # The C++ engine returns arrays shifted by the window size
    valid_targets = targets[window_size - 1:]
    
    # Slice off the last 60 rows because the 1-minute forward targets are undefined there
    X_RV = np.array(metrics["realized_variance"])[:-60]
    X_BPV = np.array(metrics["bipower_variation"])[:-60]
    X_Jumps = np.array(metrics["jump_component"])[:-60]
    X_OFI = np.array(metrics["order_flow_imbalance"])[:-60]
    y_aligned = valid_targets[:-60]

    # Stack X features into a single 2D matrix
    X_matrix = np.column_stack((X_RV, X_BPV, X_Jumps, X_OFI))

    return X_matrix, y_aligned

if __name__ == "__main__":
    print("Streaming data chunk...")
    lazy_pipeline = get_lazy_feeder(FILE_PATH)
    feeder = stream_hourly_chunks(lazy_pipeline, chunk_hours=2) # Load 2 hours to ensure enough data
    
    chunk = next(feeder)
    
    print("Building X/y training matrix...")
    X, y = build_training_matrix(chunk)
    
    print("✅ Matrix Built Successfully!")
    print(f"X matrix shape: {X.shape}")
    print(f"y target shape: {y.shape}")