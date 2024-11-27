import pandas as pd
import os
from enum import Enum
import pycountry as pc
import numpy as np
import logging
import datetime
import textwrap
import itertools
import statsmodels.api as sm
import random

formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler = logging.FileHandler(f'logs\\{datetime.date.today()}.log', mode='w')
handler.setFormatter(formatter)
logger = logging.getLogger(__name__)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)


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
        # All features will be merged into this DataFrame
        self.df = pd.DataFrame(columns=["code", "year"])
        self.cannot_convert_to_alpha_3 = {}
        self.additional_country_name_to_alpha_3_map = {
            "Ethiopia PDR": "ETH", "China, mainland": "CHN", "China, Taiwan Province of": "TWN", "Democratic Republic of the Congo": "COD", "Sudan (former)": "SDN"}
        self.empty_country_count = 0
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
            "1. Mean air temperature": False,
            "2. Energy use in agriculture": True,
            "3. Land area (sq. km)": False,
            "4. Agriculture land area (% of land area)": False,
            "5. Average precipitation in depth (mm per year)": True,
            "6. Permanent cropland (% of land area)": False,
            "7. Fertilizer consumption (kilograms per hectare of arable land)": True,
            "8. Annual freshwater withdrawals, total (billion cubic meters)": False,
            "9. Fertilizers by Nutrient (potash K2O)": False,
            "10. PM2.5 air pollution, mean annual exposure (micrograms per cubic meter)": False,
            "11. Arable land (hectares)": False,
            "13. Population": True,
            "14. Fertilizers by Nutrient (phosphate P2O5)": False,
            "15. GDP per capita, PPP (current international $)": True,
            "16. Population living in slums (% of urban population)": False,
            "17. Employment in agriculture (% of total employment) (modeled ILO estimate)": False,
            "18. Temperature change on land": False,
            "19. Fertilizers by Nutrient (nitrogen N)": False,
            "20. Agriculture land area (sq. km)": True,
            "22. Fertilizer consumption (kilograms)": False
        }
        self.output_folder_path = os.path.join(".", "output")
        self.saved_models = []
        self.feature_to_df_dict = {}
        self.feature_to_origin_url_dict = {}
        self.feature_folder_paths = self.get_all_feature_folder_paths()
        self.load_features_into_memory()

    def load_features_into_memory(self):
        feature_to_origin_url_dict = self.map_origin_urls()
        feature_to_df_dict = {}
        for feature_folder_path in self.feature_folder_paths:
            feature_name = os.path.basename(feature_folder_path)
            if feature_name != "0. Target":
                if self.features_to_save_dict.get(feature_name, False) == False:
                    continue
            feature_path = self.get_feature_file_path(feature_folder_path)
            df = self.file_to_df(
                feature_name, feature_path, feature_to_origin_url_dict[feature_name])
            feature_to_df_dict[feature_name] = df
        self.feature_to_df_dict = feature_to_df_dict
        self.feature_to_origin_url_dict = feature_to_origin_url_dict

    def country_name_to_alpha_3(self, country_name: str):
        try:
            alpha_3 = pc.countries.lookup(country_name).alpha_3
            return alpha_3
        except LookupError:
            alpha_3 = self.additional_country_name_to_alpha_3_map.get(
                country_name, "ZZZZZ")
            if alpha_3 == "ZZZZZ":
                self.cannot_convert_to_alpha_3[country_name] = True
            return alpha_3

    def alpha_3_to_country_name(self, alpha_3: str):
        try:
            country_name = pc.countries.get(alpha_3=alpha_3).name
            return country_name
        except Exception as e:
            return alpha_3

    def file_to_df(self, feature_name: str, file_path: str, file_origin: FileOrigin):
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
            logger.debug(f"Feature: {feature_name}, shape: {
                         df.shape}, years: {df['year'].unique()}")
            print("Converted to feature to df:",
                  feature_name, file_path, file_origin)
            return df
        except Exception as e:
            print("Error reading feature file", file_path, file_origin)
            print("DataFrame shape:", df.shape)
            print("Columns:", df.columns)
            raise e

    def find_origin_url(self, folder: str):
        entries = os.listdir(folder)
        if "url.txt" not in entries:
            raise Exception("URL file not found in folder", folder)
        with open(os.path.join(folder, "url.txt"), 'r') as f:
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
        # Clean the DataFrame
        df.drop(df[~df["code"].isin(
            self.country_codes_to_save)].index, inplace=True)
        numeric_mask = df.iloc[:, 2:].apply(
            lambda x: pd.to_numeric(x, errors='coerce')).notna().all(axis=1)
        df = df[numeric_mask]
        df.iloc[:, 2:] = df.iloc[:, 2:].apply(
            pd.to_numeric, errors='coerce')
        df = df.dropna()
        # df.drop(df[df["code"] == "CPV"].index, inplace=True)
        # df = df[df["year"] >= 2014]
        # df = df[df["year"] <= 2021]
        # df["3. Agricultural land area (sq. km)"] = df["3. Land area (sq. km)"] * \
        #     df["4. Agriculture land area (% of land area)"] / 100
        # df.drop(columns=["3. Land area (sq. km)",
        #              "4. Agriculture land area (% of land area)"], inplace=True)

        # Sort the columns
        sorted_columns = sorted(df.columns, key=lambda x: (0, x) if x in [
            "code", "year"] else (1, int(x.split(".")[0])))
        df = df[sorted_columns]
        return df

    def get_feature_file_path(self, feature_folder_path: str):
        for feature_file in os.listdir(feature_folder_path):
            if feature_file == "url.txt":
                continue
            return os.path.join(feature_folder_path, feature_file)

    def save_countries(self, df: pd.DataFrame):
        for country in self.country_codes_to_save:
            df = df.drop(df[df["code"] != country].index)
            if df.empty:
                logger.debug(f"Empty DataFrame for country: {
                             self.alpha_3_to_country_name(country)}")
                self.empty_country_count += 1
                continue
            df.to_excel(os.path.join(self.output_folder_path,
                                     f"{country}.xlsx"), index=False)
            df.to_csv(os.path.join(self.output_folder_path,
                                   f"{country}.csv"), index=False)
        logger.debug(f"Empty country count: {self.empty_country_count}")

    def log_satisfactory_columns(self):
        # Get all feature files and convert them into DataFrames to store into memory
        features_to_save_list = []
        for feature in self.features_to_save_dict.keys():
            if self.features_to_save_dict[feature]:
                features_to_save_list.append(feature)
        all_combinations = []
        for r in range(1, len(features_to_save_list) + 1):
            combinations = itertools.combinations(features_to_save_list, r)
            all_combinations.extend(combinations)
        random.shuffle(all_combinations)

        # Get dependent variable
        target = "0. Target"
        target_df = self.file_to_df(target, os.path.join(self.features_path, target, "FAOSTAT_data_en_11-27-2024.csv"),
                                    FileOrigin.FAO)
        saved_count = 0
        for comb in all_combinations:
            print(comb)
            df = None
            for feature in comb:
                df = pd.merge(target_df, self.feature_to_df_dict[feature], on=[
                              "code", "year"], how="outer")
            df = self.clean_and_sort_dataframe(df)
            Y = df.iloc[:, [2]]
            X = df.iloc[:, 3:]
            X = sm.add_constant(X)
            model = sm.OLS(Y.astype(float), X.astype(float)).fit()
            if model.rsquared > 0.4 and model.rsquared < 0.6:
                self.saved_models.append((model.rsquared, comb, model))
                saved_count += 1
                print(saved_count)
                if saved_count >= 10:
                    break
        self.saved_models.sort(key=lambda x: x[0], reverse=True)
        for saved_model in self.saved_models:
            logger.debug(f"Model: {saved_model[1]}, R^2: {saved_model[0]}")

    def compile(self):
        # Iterate over all feature folders to merge them into a single DataFrame
        df = pd.DataFrame(columns=["code", "year"])
        for feature_folder_path in self.feature_folder_paths:
            for feature_file in os.listdir(feature_folder_path):
                if feature_file == "url.txt":
                    continue
                feature_name = os.path.basename(feature_folder_path)
                if self.features_to_save_dict.get(feature_name, False) == False:
                    if feature_name != "0. Target":
                        continue
                df = pd.merge(
                    df, self.feature_to_df_dict[feature_name], on=['code', 'year'], how='outer')
                df.drop(df[df["code"] ==
                           "ZZZZZ"].index, inplace=True)
                df = self.clean_and_sort_dataframe(df)
        # Save the aggregated features
        print('Aggregrated DataFrame shape:', df.shape)
        print('Features:', df.columns)
        logger.debug(f"Cannot convert to alpha 3: {
            list(self.cannot_convert_to_alpha_3.keys())}")
        df.to_excel(os.path.join(self.output_folder_path,
                                 "aggregrated.xlsx"), index=False)
        df.to_csv(os.path.join(self.output_folder_path,
                               "aggregrated.csv"), index=False)


features_compiler = FeaturesCompiler()
# features_compiler.log_satisfactory_columns()
features_compiler.compile()
