from wineteller.modeling.import_data import clean_wine_data, get_test_data

# from a trained knn model, return country, region_1, province, variety, price of a wine

# load distance, i from trained knn model
distances = []
indices = []
df = get_test_data("winemag-data_first150k")
cleaned = clean_wine_data(df)

for i in indices :
    wine_variety = cleaned['variety'][i]
    wine_country = cleaned['country'][i]
    wine_province = cleaned['province'][i]
    wine_province = cleaned['region_1'][i]
    wine_price = cleaned['price'][i]
