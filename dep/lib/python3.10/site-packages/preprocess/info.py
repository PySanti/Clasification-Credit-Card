
def show_columns_dist(df):
    """
        Muesta la distribucion de datos de cada columna
    """
    print("~~~~~ Distribucion de datos")
    for a in df:
        s_data = df[a].value_counts()
        print(f"~~~~~~~~~~~~~~ Columna : {a}")
        for k,v in s_data.items():
            print(f"{k} -> {v*100/len(df)}")


