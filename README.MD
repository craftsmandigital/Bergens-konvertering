## Konverterings fremgangsmåte
Det benyttes 3 python Jupiter notebooks til konverteringen, Disse lager 3 forskjellige tabeller (dataframes) som kan benyttes for direkte import til Cornerstone. Det lages også et Excel regneark for dubletter som skal gås igjennom manuelt.

Her er de 3 Jupiter notebooks og rekkefølgen på hvordan det skal jobbes med konverteringen
1. Notebook: `dubletter.ipynb`
   Lager et Excel regneark `DubletterOsloBergen.xlsx` som skal behandles manuelt, etter instruksjoner som er beskrevet i notebook. dette regnearket vil bli benyttet i steg 2.
2. Notebook: `konvertering.ipynb`
   Lager tabeller for konverteringen
	1. Oppdatering av personer fra Bergens registeret som allerede eksisterer i Cornerstone(dubletter)
	   tabellen `df_konverterings_tabell_bare_dubletter` benyttes. Les instruksjoner under punkt 7. i notebook
	2. Nye personer som ikke eksisterer i Cornerstone.
	   Tabellen `df_konverterings_tabell_uten_dubletter` benyttes
3. Notebook: `personSamenkoplinger.ipynb`
   Oppdaterer Cornerstone med person relasjoner på Bergens personer som nå alle som en ligger klare i Cornerstone.
   Tabellen som benyttes for dette er `df_connectedMedlemer`

