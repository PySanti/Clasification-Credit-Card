def sampling_bias_analysis(df):
    """
        Funcion creada para realizar un estudio rapido de posibles
        desequilibrios de datos en dataframes.
    """
    num_features = df.select_dtypes(exclude="object").columns.tolist()
    cat_features = df.select_dtypes(include="object").columns.tolist()
    print("Mostrando features categoricas")
    for a in cat_features:
        print(f"~ {a}")
        for k,v in df[a].value_counts().items():
            print(f'{k} : {v*100/len(df)}')
    print(df[num_features].describe())

