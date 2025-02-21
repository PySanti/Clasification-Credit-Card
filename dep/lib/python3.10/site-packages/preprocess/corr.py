def corr(df):
    """
            Recibe un dataframe y retorna otro dataframe con
            las correlaciones mas destacadas entre datos

            Se espera recibir el dataframe completo, esto
            para retornar todas las correlaciones, tanto feature-feature
            como feature-target.

            Esta funcion toma dataframe.corr(), convierte la matriz
            obtenida en un conjunto de pares clave-clave-valor (metodo unstack)
            y elimina duplicados y relaciones feature x con feature x.
    """
    corr_matrix = df.corr()

    # Obtener los pares de características con las correlaciones más altas
    corr_pairs = corr_matrix.unstack()
    sorted_pairs = corr_pairs.sort_values(kind="quicksort", ascending=False)

    # Filtrar los pares duplicados y la diagonal (correlación de una columna consigo misma)
    filtered_pairs = sorted_pairs[sorted_pairs < 1].drop_duplicates()

    return filtered_pairs

