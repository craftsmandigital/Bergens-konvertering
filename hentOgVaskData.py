import   polars as pl
from polars import selectors as cs

def hent_og_vask_data() -> pl.DataFrame:
    """
    Henter data fra fil
    leser inn Data exportert i fra Bergens Access database
    i fra den tabellen som heter "tblInnmeldte"
    Gjør filtrering på data, sån at det blir likt som det som presenteres i Access database query
    Oppdaterer Medlemsnummere til et Ledig intervall i Cornerstone
    Automatisk vask av "Personnummer", der det bare er skervet in fødselsdato delen. Har sjekket at "Fødselsdato" kolonnen er satt på alle det gjelder
    """
    '''
    Medlemsnummer intervall fra 11000 til 16000.
    Medlemsnummer i bergens registeret strekker seg mellom 1 og 6000
    I Cornerstone er det ledige nummer intervall mellom 11000 og 16000
    For å få de gamle medlemsnummer innenfor intervallet så plusser vi 11000 til gammelt medlemsnummer
    Bergen hadde 282 medlemme ved årskiftet til 2025
    '''
    RANGE_START: int = 11000

    # leser inn Data exportert i fra Bergens Access database
    # I fra den tabellen som heter "tblInnmeldte"
    df = (pl.read_excel("tblInnmeldte 2025-06-12.xlsx")
          # Vet ikke hvorfor dette blir gjort, men det er det som skjer i access databasen "qryMedlemmer" -
          # # som er utgangspunkt for alle data som vises i Access
          # # uten filter finnes 333 medlemmer, mens med filter finnes 284 medlemmer.
          # # det er disse som har vært utgangspunkt for medlems opptellig
          .filter(
            # pl.col("POSTNR").is_between(pl.lit("0100"), pl.lit("9999"), closed="both") &
            pl.col("POSTNR").is_not_null() &
            (pl.col("POSTNR") != "") &
            # Det er en god del med disse verdiene. dette er utdaterte greier
            ~pl.col("POSTNR").is_in(["UL-100", "U-100"]) &
            # Organisasjoner skal ikke bli konvertert. Disse blir konvertert manuelt.
            ~(pl.col("Kategori") == "O")
          )
        )
    '''
    # --- Clean all string columns ---
    # Convert empty strings and whitespace-only strings to null
    df = df.with_columns(
        pl.when(cs.string().str.strip_chars().str.len_bytes() == 0)
        .then(None) # 'None' is used to represent a null value
        .otherwise(cs.string())
    )
    '''
    # --- The Single, Robust Pattern to Clean All String Columns ---
    df = df.with_columns(
        [
            pl.when(pl.col(c).str.strip_chars().str.len_bytes() == 0)
            .then(None)
            .otherwise(pl.col(c))
            .alias(c)
            for c in df.select(cs.string()).columns
        ]
    )

            # Some heawy hardcoding. Dubletter mellom medlemmer i Oslo/.
    
    df = (df.with_columns(
        pl.when(pl.col("MedlemID").is_in([1577, 4598, 4593, 1181, 3484])) # IF this condition is true...
        .then(pl.lit("V"))                                # THEN set the value to "V"
        .otherwise(pl.col("Kategori"))                    # OTHERWISE, keep the original value
        .alias("Kategori")                                # Assign the result back to the "Kategori" column
    )
          )


    # Oppdatering av medlemsID til Cornerstone standard og litt vasking også
    df = df.with_columns(
        # Lager nye medlemsnummere som er tilpasset Cornerstone
        (pl.col("MedlemID") + RANGE_START).alias("MedlemID"),
        (pl.col("SamID") + RANGE_START).alias("SamID"),
        (pl.col("MorID") + RANGE_START).alias("MorID"),
        (pl.col("FarID") + RANGE_START).alias("FarID"),
        # Tar bort Personnummer der det bare er skervet in fødselsdato delen
        # Har sjekket at "Fødselsdato" er satt på alle det gjelder
        pl.when(pl.col("Personnummer").str.len_chars() == 6)
        .then(None)
        .otherwise(pl.col("Personnummer"))
        .alias("Personnummer")
    )

    return df 
