import pandas as pd
import os
from enum import Enum
import pycountry as pc


class FileOrigin(Enum):
    DATA_WORLD_BANK = "https://data.worldbank.org/"
    CLIMATE_KNOWLEDGE_PORTAL = "https://climateknowledgeportal.worldbank.org/"
    FAO = "https://www.fao.org/"


class FeaturesCompiler:
    def __init__(self, feature_folders_path='.'):
        # Path to the folder containing all feature folders like 1. Mean air temperature
        self.feature_folders_path = feature_folders_path
        # All features will be merged into this DataFrame
        self.df = pd.DataFrame(columns=["code", "year"])
        self.cannot_convert_to_alpha_3 = {}

    def country_name_to_alpha_3(self, country_name: str):
        try:
            alpha_3 = pc.countries.lookup(country_name).alpha_3
            return alpha_3
        except LookupError:
            self.cannot_convert_to_alpha_3[country_name] = True
            return "ZZZZZ"

    def read_feature(self, feature_name: str, file_path: str, file_origin: FileOrigin):
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
                df.drop(columns=["Country Name", "Indicator Name",
                        "Indicator Code"], inplace=True)
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
                df.drop(columns=["Area Code (M49)", "Domain", "Months Code", "Months",
                                 "Domain Code", "Element Code", "Element", "Item Code", "Item", "Year Code", "Unit", "Flag", "Flag Description"], inplace=True, errors="ignore")
                df["year"] = df["year"].apply(lambda x: int(x))
                self.df = pd.merge(
                    self.df, df, on=["code", "year"], how='outer')
            else:
                raise Exception("Unknown file origin", file_path)
            print(df)
        except Exception as e:
            print("Error reading feature file", file_path, file_origin)
            raise e

    def find_origin_url(self, folder: str):
        entries = os.listdir(folder)
        if "url.txt" not in entries:
            raise Exception("URL file not found in folder", folder)
        with open(os.path.join(folder, "url.txt"), 'r') as f:
            url = f.read().strip()
        print("URL:", url)
        if url.startswith(FileOrigin.DATA_WORLD_BANK.value):
            return FileOrigin.DATA_WORLD_BANK
        elif url.startswith(FileOrigin.CLIMATE_KNOWLEDGE_PORTAL.value):
            return FileOrigin.CLIMATE_KNOWLEDGE_PORTAL
        elif url.startswith(FileOrigin.FAO.value):
            return FileOrigin.FAO
        else:
            raise Exception("Unsupported origin URL", url)

    def compile(self):
        print("Starting compilation at:", f'"{self.feature_folders_path}"')
        for feature_folder in os.listdir(self.feature_folders_path):
            feature_folder_path = os.path.join(
                self.feature_folders_path, feature_folder)
            if feature_folder.startswith('.') or not os.path.isdir(feature_folder_path):
                continue
            print("Feature folder located:", feature_folder_path)
            origin_url = self.find_origin_url(feature_folder_path)
            for feature in os.listdir(feature_folder_path):
                if feature == "url.txt":
                    continue
                feature_name = os.path.basename(feature_folder_path)
                feature_path = os.path.join(feature_folder_path, feature)
                self.read_feature(feature_name, feature_path, origin_url)
        self.df.dropna(inplace=True)
        print('Final DataFrame:', self.df)
        print('DataFrame columns:', self.df.columns)
        print('Countries that could not be converted to alpha-3:',
              self.cannot_convert_to_alpha_3)
        self.df.to_csv('features.csv', index=False)


features_compiler = FeaturesCompiler(os.path.join(".", "features"))
features_compiler.compile()
