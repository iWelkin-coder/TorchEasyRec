# Copyright (c) 2025, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pandas as pd

"""
This script is used to flatten the sequence for only use sequence to train.
"""

train_path = "ml-20m_train.csv"
df = pd.read_csv(train_path, header=None)
df.columns = [
    "user_id",
    "history_length",
    "historical_ids",
    "historical_ratings",
    "historical_timestamps",
    "target_id",
    "target_rating",
    "target_timestamp",
]
rows = []
for _, row in df.iterrows():
    hist_ids = str(row["historical_ids"]).split(",")
    hist_ids = hist_ids + [str(row["target_id"])]
    hist_ratings = str(row["historical_ratings"]).split(",")
    hist_timestamps = str(row["historical_timestamps"]).split(",")
    new_row = {
        "user_id": row["user_id"],
        "history_length": row["history_length"],
        "historical_ids": ";".join(hist_ids),
        "historical_ratings": ";".join(hist_ratings),
        "historical_timestamps": ";".join(hist_timestamps),
        "item_id": f"[{row['target_id']}]",
        "target_id": row["target_id"],
        "target_rating": row["target_rating"],
        "target_timestamp": row["target_timestamp"],
        "clk": 1,
    }
    rows.append(new_row)
new_df = pd.DataFrame(rows)
new_df = new_df.iloc[1:]
new_df.to_csv(
    train_path.replace(".csv", "_seq_with_target.csv"), index=False, header=False
)
