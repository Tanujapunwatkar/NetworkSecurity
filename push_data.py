import os
import sys
import json
import certifi
import pandas as pd
import numpy as np
import pymongo
from dotenv import load_dotenv

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

# Load environment variables
load_dotenv()
MONGO_DB_URL = os.getenv("MONGO_DB_URL")
print("🔗 MongoDB URL Loaded:", MONGO_DB_URL)

# SSL certificate path
ca = certifi.where()


class NetworkDataExtract():
    def __init__(self):
        try:
            logging.info("NetworkDataExtract object created")
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def cv_to_json_convertor(self, file_path):
        try:
            logging.info(f"Reading CSV file from: {file_path}")
            data = pd.read_csv(file_path)
            data.reset_index(drop=True, inplace=True)

            records = list(json.loads(data.T.to_json()).values())
            logging.info(f"Converted {len(records)} records from CSV to JSON")
            return records
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def insert_data_mongodb(self, records, database, collection):
        try:
            logging.info(f"Connecting to MongoDB database: {database}, collection: {collection}")

            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL, tlsCAFile=ca)
            self.database = self.mongo_client[database]
            self.collection = self.database[collection]

            result = self.collection.insert_many(records)
            logging.info(f"✅ Successfully inserted {len(result.inserted_ids)} records into {database}.{collection}")
            return len(result.inserted_ids)

        except Exception as e:
            raise NetworkSecurityException(e, sys)


if __name__ == '__main__':
    try:
        FILE_PATH = "Network_Data\\phisingData.csv"
        DATABASE = "Tanuja"
        COLLECTION = "NetworkData"

        networkobj = NetworkDataExtract()

        # Convert CSV -> JSON
        records = networkobj.cv_to_json_convertor(file_path=FILE_PATH)
        print(f"📄 Sample Record: {records[0]}")  # Show first record for verification

        # Insert into MongoDB
        no_of_records = networkobj.insert_data_mongodb(records, DATABASE, COLLECTION)
        print(f"✅ {no_of_records} records inserted into MongoDB successfully!")

    except Exception as e:
        print("❌ Error:", e)
