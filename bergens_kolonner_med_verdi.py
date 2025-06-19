import polars as pl

def bergens_kolonner_med_verdi(df: pl.DataFrame, column_name: str) -> pl.DataFrame:
    """
    Concatenates specified columns into a single descriptive text column.

    The new column is only populated for rows where at least one of the
    source columns has a valid value (not null, or not False for booleans).

    Args:

        df: The input Polars DataFrame.
        column_name: The name for the new descriptive column (e.g., "Beskrivelse").

    Returns:
        A new Polars DataFrame with the added descriptive column.
    """

    # 1. Define the list of columns to be included in the description
    columns_for_description = [
        "Kategori","Anmerkning", "Menighet", "Ant barn under 15 år", "Dåpskirke/sted",
        "Og", "SAM_FOR", "SAM_EFTER", "Utsendelse", "Utsendelses kode",
        "Trossamfunn", "Gift?", "Utsendelsesformat", "Ektemakens trossamfunn",
        "Svart?", "Er mor og far gift?", "Prest", "Dåpsattest", "Søk"
    ]

    # 2. Define the static header text
    HEADER_TEXT = "\nFelter i fra det gammle Bergens registeret:\n---\n"

    # 3. Build the list of expressions for each part of the description
    formatted_parts = [
        pl.when(
            # Use a Python ternary operator to choose the correct inclusion condition
            # based on the column's actual data type from the schema.
            (pl.col(c) == True) if (df.schema[c] == pl.Boolean) else pl.col(c).is_not_null()
        )
        .then(
            # Format the output string: "==> ColumnName: [Value]\n"
            pl.concat_str([
                pl.lit(f"==> {c}: "),
                pl.col(c).cast(pl.String),
                pl.lit("\n")
            ])
        )
        .otherwise(None)
        for c in columns_for_description
    ]

    # 4. Create the final descriptive column using a single conditional expression
    df_with_description = df.with_columns(
        # This is the "all-or-nothing" condition.
        # IF there is any data to show...

        pl.when(
            pl.any_horizontal(expr.is_not_null() for expr in formatted_parts)
        )
        .then(
            # THEN, perform the full concatenation with the header.
            pl.concat_str(
                [
                    pl.lit(HEADER_TEXT),
                    *formatted_parts
                ],
                ignore_nulls=True
            )
        )
        # OTHERWISE (if there was no data), the result for this row is null.
        .otherwise(None)
        .alias(column_name)
    )



    # 4. C
    # Note: The print statements below are for debugging during development.
    # You might want to remove them for production code.
    # ---
    # print("First row's description:")
    # print(df_with_description.get_column(column_name)[0])
    # print("\n" + "="*40 + "\n")
    # print("Second row's description:")
    # print(df_with_description.get_column(column_name)[1])
    #
    # with pl.Config(fmt_str_lengths=-1, tbl_rows=10):
    #     print(df_with_description.select("FORNAVN", "EFTERNAVN", column_name))
    # ---

    # The function must return the DataFrame that includes the new column.
    return df_with_description
