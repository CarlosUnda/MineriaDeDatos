from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd

# Datos de transacciones
transacciones = [
    ['Pan', 'Leche'],
    ['Pan', 'Pañales', 'Cerveza'],
    ['Leche', 'Pañales'],
    ['Pan', 'Leche', 'Pañales', 'Huevos'],
    ['Huevos', 'Leche'],
    ['Pan', 'Huevos'],
    ['Pañales', 'Cerveza'],
    ['Pan', 'Leche', 'Huevos']
]

# 1. Preprocesamiento
te = TransactionEncoder()
te_ary = te.fit(transacciones).transform(transacciones)
df = pd.DataFrame(te_ary, columns=te.columns_)

# 2. Itemsets frecuentes
frequent_itemsets = apriori(df, min_support=0.4, use_colnames=True)

# 3. Reglas de asociación
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)
rules = rules[rules['lift'] > 1]

# Filtrar reglas donde "Pan" está en el antecedente o consecuente
pan_rules = rules[(rules['antecedents'].apply(lambda x: 'Pan' in x)) |
                  (rules['consequents'].apply(lambda x: 'Pan' in x))]

# Ordenar por lift descendente
pan_rules_sorted = pan_rules.sort_values(by='lift', ascending=False)

# Mostrar reglas asociadas a Pan ordenadas por lift
print(pan_rules_sorted[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

# Producto más asociado a Pan por lift
if not pan_rules_sorted.empty:
    top_rule = pan_rules_sorted.iloc[0]
    if 'Pan' in top_rule['antecedents']:
        asociado = list(top_rule['consequents'])[0]
    else:
        asociado = list(top_rule['antecedents'])[0]
    print(f"\nProducto más asociado a 'Pan': {asociado} (Lift: {top_rule['lift']:.2f})")
else:
    print("No se encontraron reglas con Pan y lift > 1.")
