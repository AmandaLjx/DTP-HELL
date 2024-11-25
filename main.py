import pandas as pd
import os
from enum import Enum
import pycountry as pc
import numpy as np
import logging
import datetime
import textwrap

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


class FeaturesCompiler:
    def __init__(self, features_path=os.path.join(".", "features")):
        # Path to the folder containing all feature folders like 1. Mean air temperature
        self.features_path = features_path
        # All features will be merged into this DataFrame
        self.df = pd.DataFrame(columns=["code", "year"])
        self.cannot_convert_to_alpha_3 = {}
        self.country_name_to_alpha_3_map = {
            "Ethiopia PDR": "ETH", "China, mainland": "CHN", "China, Taiwan Province of": "TWN"}

    def country_name_to_alpha_3(self, country_name: str):
        try:
            alpha_3 = pc.countries.lookup(country_name).alpha_3
            return alpha_3
        except LookupError:
            alpha_3 = self.country_name_to_alpha_3_map.get(
                country_name, "ZZZZZ")
            if alpha_3 == "ZZZZZ":
                self.cannot_convert_to_alpha_3[country_name] = 1
            return alpha_3

    def alpha_3_to_country_name(self, alpha_3: str):
        try:
            country_name = pc.countries.get(alpha_3=alpha_3).name
            return country_name
        except Exception as e:
            return alpha_3

    def merge_feature(self, feature_name: str, file_path: str, file_origin: FileOrigin):
        try:
            df = None
            feature_name = os.path.basename(os.path.dirname(file_path))
            if feature_name in self.df.columns:
                raise Exception(
                    "Feature already exists in DataFrame:", feature_name)
            # Drop columns that start with 'Unnamed:'
            if file_origin == FileOrigin.DATA_WORLD_BANK:
                df = pd.read_csv(file_path)
                df = df.loc[:, ~df.columns.str.contains('Unnamed: 68')]
                columns_to_drop = ["Country Name", "Indicator Name",
                                   "Indicator Code"]
                df.drop(columns_to_drop, inplace=True, axis=1)
                df = pd.melt(df, id_vars=[
                             "Country Code"], value_vars=df.columns[1:], var_name="year", value_name=feature_name)
                df.rename(columns={"Country Code": "code"}, inplace=True)
                df["year"] = df["year"].apply(lambda x: int(x))
                self.df = pd.merge(
                    self.df, df, on=['code', 'year'], how='outer')
            elif file_origin == FileOrigin.CLIMATE_KNOWLEDGE_PORTAL:
                df = pd.read_excel(file_path)
                df = df.loc[:, ~df.columns.str.contains('Unnamed: 68')]
                df.drop(columns=["name"], inplace=True)
                df = pd.melt(df, id_vars=[
                             "code"], value_vars=df.columns[1:], var_name="year", value_name=feature_name)
                df["year"] = df["year"].apply(lambda x: int(x.split("-")[0]))
                self.df = pd.merge(
                    self.df, df, on=['code', 'year'], how='outer')
            elif file_origin == FileOrigin.FAO:
                df = pd.read_csv(file_path)
                df = df.loc[:, ~df.columns.str.contains('Unnamed: 68')]
                df["Area"] = df["Area"].apply(
                    self.country_name_to_alpha_3)
                df.rename(
                    columns={"Area": "code", "Year": "year", "Value": feature_name}, inplace=True)
                columns_to_drop = ["Item Code (CPC)", "Area Code (M49)", "Domain", "Months Code", "Months", "Note",
                                   "Domain Code", "Element Code", "Element", "Item Code", "Item", "Year Code", "Unit", "Flag", "Flag Description"]
                df.drop(columns=columns_to_drop,
                        inplace=True, errors='ignore', axis=1)
                df["year"] = df["year"].apply(lambda x: int(x))
                self.df = pd.merge(
                    self.df, df, on=["code", "year"], how='outer')
            else:
                raise Exception("Unknown file origin", file_path)
            print("Merge successful:", feature_name, file_path, file_origin)
        except Exception as e:
            print("Error reading feature file", file_path, file_origin)
            print("DataFrame shape:", self.df.shape)
            print("Columns:", self.df.columns)
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
        else:
            raise Exception("Unsupported origin URL", url)

    def get_all_feature_folder_paths(self):
        return [os.path.join(self.features_path, f) for f in os.listdir(self.features_path) if os.path.isdir(os.path.join(self.features_path, f))]

    def compile(self):
        countries_to_save = ["SOM", "ETH", "KEN", "SSD", "SDN"]
        features_to_save = {"0. Crop yield (Wheat, tons)": True, "1. Mean air temperature": True,
                            "5. Average precipitation (mm)": True,
                            "7. Fertilizer consumption (kilograms per hectare of arable land)": True,
                            "13. Population": True, "4. Agriculture land area (% of land area)": True}

        # Iterate over all feature folders to merge them into a single DataFrame
        feature_folder_paths = self.get_all_feature_folder_paths()
        for feature_folder_path in feature_folder_paths:
            origin_url = self.find_origin_url(feature_folder_path)
            for feature_file in os.listdir(feature_folder_path):
                if feature_file == "url.txt":
                    continue
                feature_name = os.path.basename(feature_folder_path)
                if features_to_save.get(feature_name, False) == False:
                    continue
                feature_path = os.path.join(feature_folder_path, feature_file)
                self.merge_feature(feature_name, feature_path, origin_url)
                self.df.drop(self.df[self.df["code"] ==
                             "ZZZZZ"].index, inplace=True)

        # Clean the DataFrame
        self.df.drop(self.df[~self.df["code"].isin(
            countries_to_save)].index, inplace=True)
        self.df.dropna(inplace=True)

        # Sort the columns
        sorted_columns = sorted(self.df.columns, key=lambda x: (0, x) if x in [
            "code", "year"] else (1, int(x.split(".")[0])))
        self.df = self.df[sorted_columns]

        # Save the individual country features
        output_folder_path = os.path.join(".", "output")
        for country in countries_to_save:
            df = self.df.drop(self.df[self.df["code"] != country].index)
            df.to_excel(os.path.join(output_folder_path,
                                     f"{country}.xlsx"), index=False)
            df.to_csv(os.path.join(output_folder_path,
                                   f"{country}.csv"), index=False)

        # Save the aggregated features
        print('Aggregrated DataFrame shape:', self.df.shape)
        print('Features:', self.df.columns)
        wrapped_message = textwrap.fill(f"Cannot convert to alpha 3: {
                                        list(self.cannot_convert_to_alpha_3.keys())}", width=50)
        logger.debug(wrapped_message)

        self.df.to_excel(os.path.join(output_folder_path,
                                      "aggregrated.xlsx"), index=False)
        self.df.to_csv(os.path.join(output_folder_path,
                                    "aggregrated.csv"), index=False)


features_compiler = FeaturesCompiler()
features_compiler.compile()
