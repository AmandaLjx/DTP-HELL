import pandas as pd
import os
from enum import Enum
# import logging
# import datetime

# formatter = logging.Formatter(
#     '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# handler = logging.FileHandler(f'logs\\{datetime.date.today()}.log', mode='w')
# handler.setFormatter(formatter)
# logger = logging.getLogger(__name__)
# logger.addHandler(handler)
# logger.setLevel(logging.DEBUG)


no_alpha_3_country_string = "ZZZZZ"


class FileOrigin(Enum):
    DATA_WORLD_BANK = "https://data.worldbank.org/"
    CLIMATE_KNOWLEDGE_PORTAL = "https://climateknowledgeportal.worldbank.org/"
    FAO = "https://www.fao.org/"
    DATABANK_WORLDBANK = "https://databank.worldbank.org/"
    SANITIZED = "sanitized"


class FeaturesCompiler:
    def __init__(self, features_path=os.path.join(".", "features")):
        # Path to the folder containing all feature folders like 1. Mean air temperature
        self.features_path = features_path
        self.feature_folder_paths = self.get_all_feature_folder_paths()
        self.output_folder_path = os.path.join(".", "output")

        # Logging purposes
        self.cannot_convert_to_alpha_3 = {}

        # Narrowing what data to save
        self.country_codes_to_save = [
            "DZA", "AGO", "BEN", "BWA", "BFA", "BDI",
            "CPV", "CMR", "CAF", "TCD", "COM",
            "COG", "DJI", "EGY", "GNQ", "ERI",
            "SWZ", "ETH", "GAB", "GMB", "GHA", "GIN",
            "GNB", "KEN", "LSO", "LBR", "LBY", "MDG",
            "MWI", "MLI", "MRT", "MUS", "MAR", "MOZ",
            "NAM", "NER", "NGA", "RWA", "STP",
            "SEN", "SYC", "SLE", "SOM", "ZAF",
            "SSD", "SDN", "TZA", "TGO", "TUN", "UGA",
            "ZMB", "ZWE"
        ]
        self.features_to_save_dict = {
            "0. Target": True,
            "1. Mean air temperature": False,
            "2. Energy use in agriculture": True,
            "3. Land area (sq. km)": True,
            "4. Agriculture land area (% of land area)": True,
            "5. Average precipitation in depth (mm per year)": True,
            "6. Permanent cropland (% of land area)": False,
            "7. Fertilizer consumption (kilograms per hectare of arable land)": True,
            "8. Annual freshwater withdrawals, total (billion cubic meters)": False,
            "9. Fertilizers by Nutrient (potash K2O)": False,
            "10. PM2.5 air pollution, mean annual exposure (micrograms per cubic meter)": False,
            "11. Arable land (hectares)": True,
            "13. Population": True,
            "14. Fertilizers by Nutrient (phosphate P2O5)": False,
            "15. GDP per capita, PPP (current international $)": True,
            "16. Population living in slums (% of urban population)": False,
            "17. Employment in agriculture (% of total employment) (modeled ILO estimate)": False,
            "18. Temperature change on land": False,
            "19. Fertilizers by Nutrient (nitrogen N)": False,
            "20. Agriculture land area (sq. km)": False,
            "22. Fertilizer consumption (kilograms)": False
        }

        # Country name to alpha 3 code mapping
        self.country_name_to_alpha_3_dict = {
            "Ethiopia PDR": "ETH", "China, mainland": "CHN",
            "China, Taiwan Province of": "TWN", "Democratic Republic of the Congo": "COD", "Sudan (former)": "SDN",
            "Algeria": "DZA",
            "Angola": "AGO",
            "Botswana": "BWA",
            "Burundi": "BDI",
            "Cameroon": "CMR",
            "Chad": "TCD",
            "Egypt": "EGY",
            "Eritrea": "ERI",
            "Eswatini": "SWZ",
            "Ethiopia": "ETH",
            "Kenya": "KEN",
            "Lesotho": "LSO",
            "Libya": "LBY",
            "Madagascar": "MDG",
            "Malawi": "MWI",
            "Mali": "MLI",
            "Mauritania": "MRT",
            "Morocco": "MAR",
            "Mozambique": "MOZ",
            "Namibia": "NAM",
            "Niger": "NER",
            "Nigeria": "NGA",
            "Rwanda": "RWA",
            "Somalia": "SOM",
            "South Africa": "ZAF",
            "South Sudan": "SSD",
            "Sudan": "SDN",
            "Tunisia": "TUN",
            "Uganda": "UGA",
            "United Republic of Tanzania": "TZA",
            "Zambia": "ZMB",
            "Zimbabwe": "ZWE",
            "Benin": "BEN",
            "Burkina Faso": "BFA",
            "Cabo Verde": "CPV",
            "Central African Republic": "CAF",
            "Congo": "COG",
            "Equatorial Guinea": "GNQ",
            "Gabon": "GAB",
            "Gambia": "GMB",
            "Ghana": "GHA",
            "Guinea": "GIN",
            "Guinea-Bissau": "GNB",
            "Mauritius": "MUS",
            "Senegal": "SEN",
            "Sierra Leone": "SLE",
            "Togo": "TGO"
        }

        # Caching purposes to avoid re-reading the same files
        self.feature_to_origin_url_dict = self.map_origin_urls()
        self.feature_to_df_dict = {}

        # Loads all features int self.feature_to_df_dict, accessible by feature name like 1. Mean air temperature except for 0. Target
        self.load_features_into_memory()

    def get_feature_file_path(self, feature_folder_path: str):
        for feature_file in os.listdir(feature_folder_path):
            if feature_file == "url.txt":
                continue
            return os.path.join(feature_folder_path, feature_file)

    def load_features_into_memory(self):
        for feature_folder_path in self.feature_folder_paths:
            feature_name = os.path.basename(feature_folder_path)
            if self.features_to_save_dict.get(feature_name, False) == False:
                continue
            feature_path = self.get_feature_file_path(feature_folder_path)
            df = self.feature_file_to_df(
                feature_name, feature_path, self.feature_to_origin_url_dict[feature_name])
            self.feature_to_df_dict[feature_name] = df

    def country_name_to_alpha_3(self, country_name: str):
        alpha_3 = self.country_name_to_alpha_3_dict.get(country_name)
        if alpha_3 is None:
            self.cannot_convert_to_alpha_3[country_name] = True
            return no_alpha_3_country_string
        return alpha_3

    def feature_file_to_df(self, feature_name: str, file_path: str, file_origin: FileOrigin):
        try:
            df = None
            feature_name = os.path.basename(os.path.dirname(file_path))
            file_ext = os.path.splitext(file_path)[1]
            if file_ext == ".xlsx":
                df = pd.read_excel(file_path)
            elif file_ext == ".csv":
                df = pd.read_csv(file_path)
            else:
                raise Exception("Unknown file extension", file_path)
            df = df.loc[:, ~df.columns.str.contains('Unnamed: 68')]
            if file_origin == FileOrigin.DATA_WORLD_BANK:
                columns_to_drop = ["Country Name", "Indicator Name",
                                   "Indicator Code"]
                df.drop(columns=columns_to_drop, inplace=True)
                df = pd.melt(df, id_vars=[
                             "Country Code"], value_vars=df.columns[1:], var_name="year", value_name=feature_name)
                df.rename(columns={"Country Code": "code"}, inplace=True)
                df["year"] = df["year"].apply(lambda x: int(x))
            elif file_origin == FileOrigin.CLIMATE_KNOWLEDGE_PORTAL:
                df.drop(columns=["name"], inplace=True)
                df = pd.melt(df, id_vars=[
                             "code"], value_vars=df.columns[1:], var_name="year", value_name=feature_name)
                df["year"] = df["year"].apply(lambda x: int(x.split("-")[0]))
            elif file_origin == FileOrigin.FAO:
                df["Area"] = df["Area"].apply(
                    self.country_name_to_alpha_3)
                df.rename(
                    columns={"Area": "code", "Year": "year", "Value": feature_name}, inplace=True)
                columns_to_drop = ["Item Code (CPC)", "Area Code (M49)", "Domain", "Months Code", "Months", "Note",
                                   "Domain Code", "Element Code", "Element", "Item Code", "Item", "Year Code", "Unit", "Flag", "Flag Description"]
                df.drop(columns=columns_to_drop,
                        inplace=True, errors='ignore')
                df["year"] = df["year"].apply(lambda x: int(x))
            elif file_origin == FileOrigin.DATABANK_WORLDBANK:
                df.rename(columns={"Country Code": "code"}, inplace=True)
                df.drop(columns=["Country Name", "Series Code",
                        "Series Name"], inplace=True)
                df = pd.melt(df, id_vars=[
                             "code"], value_vars=df.columns[1:], var_name="year", value_name=feature_name)
                df["year"] = df["year"].apply(lambda x: int(x.split(" [")[0]))
            elif file_origin == FileOrigin.SANITIZED:
                df.rename(columns={"value": feature_name}, inplace=True)
                df["year"] = df["year"].apply(lambda x: int(x))
            else:
                raise Exception("Unknown file origin", file_path)
            df.dropna(inplace=True)
            df = self.clean_and_sort_dataframe(df)
            # logger.debug(f"Feature: {feature_name}, shape: {
            #              df.shape}, years: {df['year'].unique()}")
            print("Converted to feature to df:",
                  feature_name, file_path, file_origin)
            return df
        except Exception as e:
            print("Error reading feature file", file_path, file_origin)
            print("DataFrame shape:", df.shape)
            print("Columns:", df.columns)
            raise e

    def find_origin_url(self, feature_folder_path: str):
        entries = os.listdir(feature_folder_path)
        if "url.txt" not in entries:
            raise Exception("URL file not found in folder",
                            feature_folder_path)
        with open(os.path.join(feature_folder_path, "url.txt"), 'r') as f:
            url = f.read().strip()
        if url.startswith(FileOrigin.DATA_WORLD_BANK.value):
            return FileOrigin.DATA_WORLD_BANK
        elif url.startswith(FileOrigin.CLIMATE_KNOWLEDGE_PORTAL.value):
            return FileOrigin.CLIMATE_KNOWLEDGE_PORTAL
        elif url.startswith(FileOrigin.FAO.value):
            return FileOrigin.FAO
        elif url.startswith(FileOrigin.DATABANK_WORLDBANK.value):
            return FileOrigin.DATABANK_WORLDBANK
        elif url.startswith(FileOrigin.SANITIZED.value):
            return FileOrigin.SANITIZED
        else:
            raise Exception("Unsupported origin URL", url)

    def map_origin_urls(self):
        output = {}
        for feature_folder in os.listdir(self.features_path):
            feature_folder_path = os.path.join(
                self.features_path, feature_folder)
            if os.path.isdir(feature_folder_path):
                origin_url = self.find_origin_url(feature_folder_path)
                output[feature_folder] = origin_url
        return output

    def get_all_feature_folder_paths(self):
        return [os.path.join(self.features_path, f) for f in os.listdir(self.features_path) if os.path.isdir(os.path.join(self.features_path, f))]

    def clean_and_sort_dataframe(self, df: pd.DataFrame):
        # Only keep the countries we want
        df = df.drop(df[~df["code"].isin(
            self.country_codes_to_save)].index)

        # Numeric coercion (some values are string for some reason?)
        numeric_mask = df.iloc[:, 2:].apply(
            lambda x: pd.to_numeric(x, errors='coerce')).notna().all(axis=1)
        df = df[numeric_mask]
        df.iloc[:, 2:] = df.iloc[:, 2:].apply(
            pd.to_numeric, errors='coerce')

        # Drop rows with NaN values
        df = df.dropna()

        # Only keep years 2014-2021
        df = df[df["year"] >= 2014]
        df = df[df["year"] <= 2021]

        # Sort the columns
        sorted_columns = sorted(df.columns, key=lambda x: (0, x) if x in [
            "code", "year"] else (1, int(x.split(".")[0])))
        df = df[sorted_columns]
        return df

    def compile(self):
        # Iterate over loaded features in self.feature_to_df_dict and compile them
        df = pd.DataFrame(columns=["code", "year"])
        for feature_folder_path in self.feature_folder_paths:
            feature_name = os.path.basename(feature_folder_path)
            if self.features_to_save_dict.get(feature_name, False) == False:
                continue
            df = pd.merge(
                df, self.feature_to_df_dict[feature_name], on=['code', 'year'], how='outer')
            df.drop(df[df["code"] ==
                       no_alpha_3_country_string].index, inplace=True)
            df = self.clean_and_sort_dataframe(df)

        # Obtain derived columns
        if "3. Land area (sq. km)" in df.columns and "4. Agriculture land area (% of land area)" in df.columns:
            df["20. Agriculture land area (sq. km)"] = df["3. Land area (sq. km)"] * \
                df["4. Agriculture land area (% of land area)"] / 100
            df.drop(columns=["3. Land area (sq. km)",
                             "4. Agriculture land area (% of land area)"], inplace=True)
        if "11. Arable land (hectares)" in df.columns and "7. Fertilizer consumption (kilograms per hectare of arable land)" in df.columns:
            df["22. Fertilizer consumption (kilograms)"] = df["11. Arable land (hectares)"] * \
                df["7. Fertilizer consumption (kilograms per hectare of arable land)"]
            df.drop(columns=["11. Arable land (hectares)",
                             "7. Fertilizer consumption (kilograms per hectare of arable land)"], inplace=True)

        # Final cleaning
        df = self.clean_and_sort_dataframe(df)

        # Logging purposes
        print('Aggregrated DataFrame shape:', df.shape)
        print('Features:', df.columns)
        # logger.debug(f"Cannot convert to alpha 3: {
        #     list(self.cannot_convert_to_alpha_3.keys())}")

        # Save the aggregated features
        df.to_excel(os.path.join(self.output_folder_path,
                                 "aggregrated.xlsx"), index=False)
        df.to_csv(os.path.join(self.output_folder_path,
                               "aggregrated.csv"), index=False)

        return df


features_compiler = FeaturesCompiler()
features_compiler.compile()
