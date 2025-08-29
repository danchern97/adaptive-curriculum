# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import logging
import os
from typing import Any, Iterable, Iterator, List, Tuple, Union

import pandas as pd
import torch

from .local_storage import LocalClient

# Try to import MongoDB, fall back to local storage if not available
try:
    from pymongo import MongoClient
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False
    MongoClient = type(None)  # Placeholder type


def mongodb_client(mongdb_uri: str = "mongodb://localhost:27017/mongo"):
    """Constructs local storage client by default, or MongoDB client if explicitly requested.

    By default, uses local file storage. To use MongoDB, set environment variable USE_MONGODB=true.

    Args:
        mongdb_uri (str, optional): MongoDB URI if using MongoDB. Defaults to "mongodb://localhost:27017/mongo".

    Returns:
        Client used to connect to local storage or MongoDB.
    """
    # Check if we should use MongoDB (opt-in instead of opt-out)
    use_mongodb = os.environ.get("USE_MONGODB", "false").lower() == "true"
    
    if use_mongodb and MONGODB_AVAILABLE:
        db_uri = os.environ.get("MONGODB_URI", mongdb_uri)
        logging.info(f"Connecting to {db_uri}")
        return MongoClient(
            host=db_uri,
            socketTimeoutMS=1800000,
            connectTimeoutMS=1800000,
        )
    else:
        logging.info("Using local file storage")
        data_path = os.environ.get("LOCAL_DATA_PATH", "data")
        return LocalClient(data_path)


def repeat_iterator(it: Iterable[Any], n: int) -> Iterator[Any]:
    """Repeats the provided iterator n times.

    Args:
        it (Iterable[Any]): Iterator that is to be repeated.
        n (int): Number of repeats.

    Yields:
        Iterator[Any]: The repeated iterator.
    """
    step = 0
    while step < n:
        for element in it:
            yield element
            step += 1
            if step >= n:
                break


def split_df_columns_to_value(
    df: pd.DataFrame,
    index: List[str],
    split_columns: Tuple[str, ...],
    split_char: str = ".",
) -> pd.DataFrame:
    """Transforms logging dataframe.

    For example, a DataFrame with this format

    | _id | step | loss.trace | accuracy.trace |
    | --- | ---  | ---        | ---            |
    | abc | 0    | x          | y              |

    can be transformed into this format

    | _id | step | value | type     | portion |
    | --- | ---  | ---   | ---      | ---     |
    | abc | 0    | x     | loss     | trace   |
    | abc | 0    | y     | accuracy | trace   |

    This function splits all non-indexing columns at the `.` character
    and then sets the corresponding portions as values for the specified
    `split_columns`. The example above uses the settings

    ```
        index=["_id", "step"],
        split_columns=("type", "portion"),
        split_char=".",
    ```

    This results in the non-indexing columns `loss.trace` and `accuracy.trace`
    to be split and assigned the columns `type` and `portion`.

    Args:
        df (pd.DataFrame): Data frame that is transformed.
        index (List[str]): Columns that index an individual run. the example
            this would be `["_id", "step"]`.
        split_columns (Tuple[str, ...]): Column titles into which all non-index
            columns are split. This would be `("type", "portion")` in the
            example above.
        split_char (str, optional): Character that is used to split all
            non-index column names. Defaults to ".".

    Returns:
        pd.DataFrame: Transformed data frame.
    """
    columns = df.columns.difference(index)
    df_list = []
    for column in columns:
        df_col = df[[*index, column]].copy()
        for k, v in zip(split_columns, column.split(split_char)):
            df_col[k] = v
        df_col.rename(columns={column: "value"}, inplace=True)  # type: ignore
        df_list.append(df_col)
    return pd.concat(df_list)


def setup_logging_ddp():
    """Logging setup for DDP runs."""
    from torch.distributed import get_rank, get_world_size

    rank = get_rank()
    world_size = get_world_size()
    format_comp = [
        "%(levelname)s",
        "%(asctime)s",
        "%(name)s",
        f"{rank}/{world_size}",
        "%(message)s",
    ]
    logging.basicConfig(
        level=logging.DEBUG,
        format=" - ".join(format_comp),
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.info(f"rank={rank}, world_size={world_size}")
    logging.info(f"Number of cuda devices: {torch.cuda.device_count()}")
