import polars as pl
import numpy as np
import bipower_core # type: ignore
from data_feeder import get_lazy_feeder, stream_hourly_chunks, FILE_PATH

# Minimum forward return required to call a move "profitable".
# 0.0005 == 5 basis points, roughly a taker round-trip on Binance spot.
FEE_THRESHOLD = 0.0005

# 5-minute lookback window (in 1-second bars) used by the C++ engine
WINDOW_SIZE = 300

# 1-minute forward prediction horizon (in 1-second bars)
HORIZON = 60

# Guards the volatility normalisation against completely flat windows
EPSILON = 1e-8

FEATURE_NAMES = [
    "Realized Variance",
    "Bipower Variation",
    "Jumps",
    "Order Flow Imbalance",
    "5m Return",
    "Vol-Adjusted OFI",
    "Signed Jumps",
]

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
    # 1 only if the forward move clears the fee threshold; every down move,
    # flat move, or unprofitable up move collapses into 0
    df_1s = df_1s.with_columns(
        ((pl.col("price").shift(-HORIZON) / pl.col("price")) - 1.0).alias("forward_return")
    ).with_columns(
        (pl.col("forward_return") > FEE_THRESHOLD).cast(pl.Int8).alias("target_y")
    )

    # 3. Extract flat arrays for the C++ engine
    prices = df_1s["price"].to_numpy().astype(np.float64)
    qtys = df_1s["qty"].to_numpy().astype(np.float64)
    maker = df_1s["is_buyer_maker"].to_numpy().astype(bool)
    targets = df_1s["target_y"].to_numpy()

    # 4. Calculate 5-minute lookback (300 seconds)
    metrics = bipower_core.calculate_rolling_metrics(prices, qtys, maker, WINDOW_SIZE)

    # 5. Align X features and y targets
    # The C++ engine returns arrays shifted by the window size
    valid_targets = targets[WINDOW_SIZE - 1:]

    # Slice off the last HORIZON rows because the 1-minute forward targets are undefined there
    X_RV = np.array(metrics["realized_variance"])[:-HORIZON]
    X_BPV = np.array(metrics["bipower_variation"])[:-HORIZON]
    X_Jumps = np.array(metrics["jump_component"])[:-HORIZON]
    X_OFI = np.array(metrics["order_flow_imbalance"])[:-HORIZON]
    y_aligned = valid_targets[:-HORIZON]

    # 6. Contextual features
    # Log return across the same 5-minute window the C++ engine consumed:
    # window k spans bars [k, k + WINDOW_SIZE - 1], so this array aligns 1:1 with the metrics
    log_prices = np.log(prices)
    X_Ret5m = (log_prices[WINDOW_SIZE - 1:] - log_prices[:len(log_prices) - WINDOW_SIZE + 1])[:-HORIZON]

    # Order flow normalised by the continuous volatility of the window.
    # BPV is a variance, so sqrt(BPV) puts OFI on a per-unit-of-risk scale.
    X_OFI_Vol = X_OFI / (np.sqrt(X_BPV) + EPSILON)

    # Jumps are magnitude-only by construction (max(RV - BPV, 0)).
    # Signing them with the lookback direction tells the trees whether the
    # shock happened into an up-trend or a down-trend.
    X_Signed_Jumps = X_Jumps * np.sign(X_Ret5m)

    # Stack X features into a single 2D matrix
    X_matrix = np.column_stack((
        X_RV,
        X_BPV,
        X_Jumps,
        X_OFI,
        X_Ret5m,
        X_OFI_Vol,
        X_Signed_Jumps,
    ))

    return X_matrix, y_aligned

if __name__ == "__main__":
    print("Streaming data chunk...")
    lazy_pipeline = get_lazy_feeder(FILE_PATH)
    feeder = stream_hourly_chunks(lazy_pipeline, chunk_hours=2) # Load 2 hours to ensure enough data

    chunk = next(feeder)

    print("Building X/y training matrix...")
    X, y = build_training_matrix(chunk)

    positives = int((y == 1).sum())
    print("✅ Matrix Built Successfully!")
    print(f"X matrix shape: {X.shape}")
    print(f"y target shape: {y.shape}")
    print(f"Positive class (fwd return > {FEE_THRESHOLD:.2%}): {positives:,} / {len(y):,} ({positives / len(y):.2%})")
