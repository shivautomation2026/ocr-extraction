from backend.core.config import settings 
import requests
import urllib3
import pandas as pd
# from ..core.config import settings
import logging
from pathlib import Path
import os
from dotenv import load_dotenv
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Load environment variables from a specific .env file
# Priority: ENV_FILE env var > project-root/.env (two levels up from this file)
# DEFAULT_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
# ENV_FILE = os.getenv("ENV_FILE", str(DEFAULT_ENV_PATH))
# load_dotenv(dotenv_path=ENV_FILE)

# if Path(ENV_FILE).exists():
#     logger.info(f"Loaded environment from {ENV_FILE}")
# else:
#     logger.warning(f".env file not found at {ENV_FILE}; relying on process environment")

class SAPClient:
    def __init__(self):
        self.session = requests.Session()
        self.creds = {
            "CompanyDB": settings.COMPANY_DB,
            "UserName": settings.USERNAME,
            "Password": settings.PASSWORD
        }
        self.base_url = settings.BASE_URL

        try:
            login_req = self.session.post(f"{self.base_url}Login", json=self.creds, verify=False)

            if login_req.status_code == 200:
                logger.info("Successfully logged in to SAP.")
                self.save_items_to_csv()
                self.save_business_partners()

            else:
                logger.error(f"Login failed with status code: {login_req.status_code}")
                exit(1)
        except requests.exceptions.RequestException as e:
                print(f"Login request failed: {e}")
                exit(1)

    def save_items_to_csv(self):
        all_items = []
        item_url_req = f'{self.base_url}Items/?$select=ItemCode,ItemName,UoMGroupEntry,InventoryUoMEntry&$orderby=ItemName'
        
        while item_url_req:
            try:
                response = self.session.get(item_url_req, verify=False)
                items = response.json()['value']
                for item in items:
                    all_items.append({'ItemCode': item['ItemCode'],
                                    'ItemName': item['ItemName'],
                                    'UoMGroupEntry': item['UoMGroupEntry'],
                                    'InventoryUoMEntry': item['InventoryUoMEntry']
                                })
                if 'odata.nextLink' in response.json():
                    next_link = response.json()['odata.nextLink']
                    item_url_req = f'{self.base_url}{next_link}'
                else:
                    break
            except Exception as e:
                logger.error(f"Error fetching items: {e}")
                raise
            
        logger.info("Successfully fetched all pages of Items.")
        items_df = pd.DataFrame(all_items)
        items_df.to_csv("backend/assets/item_list.csv", index=False)
        logger.info("Items saved as csv")

        return items_df

    def save_item_groups_to_csv(self):
        item_group_list = []
        item_group_res = f"{self.base_url}ItemGroups?$select=GroupName,Number"

        while item_group_res:
            try:
                response = self.session.get(item_group_res, verify=False)
                group_values = response.json()['value']
                for num in group_values:
                    item_group_list.append({
                        "Series": num["Number"],
                        "GroupName": num["GroupName"]
                    })
                if 'odata.nextLink' in response.json():
                    next_link = response.json()['odata.nextLink']
                    item_group_res = f"{self.base_url}{next_link}"
                else:
                    break
            except Exception as e:
                logger.error(f"Error fetching Item Groups: {e}")
                raise
        logger.info("Successfully fetched all pages of ItemGroups.")
        
        item_group_df = pd.DataFrame(item_group_list)
        item_group_df.to_csv('backend/assets/item_groups.csv', index=False)
        logger.info("Item groups saved as csv")

        return item_group_df

    def save_uom_groups_to_csv(self):
        uom_group_list = []
        uom_group_url = f"{self.base_url}UnitOfMeasurementGroups?$select=AbsEntry,Code,BaseUoM&$orderby=AbsEntry"

        while uom_group_url:
            try:
                response = self.session.get(uom_group_url, verify=False)
                if 'value' in response.json():
                    uom_group_values = response.json()['value']
                    for num in uom_group_values:
                        uom_group_list.append({
                            "UoMGroupEntry": num["AbsEntry"],
                            "Code": num["Code"],
                        "BaseUoM": num["BaseUoM"]
                    })
                    
                    if 'odata.nextLink' in response.json():
                        next_link = response.json()['odata.nextLink']
                        uom_group_url = f"{self.base_url}{next_link}"
                    else:
                        break
                else:
                    break
                
            except Exception as e:
                logger.error(f"Error fetching UoM groups: {e}")
                raise
        logger.info("Successfully fetched all pages of UoM Groups.") 

        uom_group_df = pd.DataFrame(uom_group_list)
        uom_group_df.to_csv('backend/assets/uom_groups.csv', index=False)
        logger.info("UoM Groups saved as csv")
        return uom_group_df
    
    def save_account_codes(self):
        account_code_list = []
        account_code_url = f"{self.base_url}ChartOfAccounts?$select=Code,Name&$orderby=Name"

        while account_code_url:
            try:
                response = self.session.get(account_code_url, verify=False)
                logger.info("Fetching account codes from SAP...")

                if 'value' in response.json():
                    account_code_values = response.json()['value']
                    for num in account_code_values:
                        account_code_list.append({
                            "AccountCode": num["Code"],
                            "Name": num["Name"]
                        })
                    
                    if 'odata.nextLink' in response.json():
                        next_link = response.json()['odata.nextLink']
                        account_code_url = f"{self.base_url}{next_link}"
                    else:
                        break
                else:
                    break
                
            except Exception as e:
                logger.error(f"Error fetching Account Codes: {e}")
                raise
        logger.info("Successfully fetched all pages of Account Codes.")

        account_code_df = pd.DataFrame(account_code_list)
        account_code_df.to_csv('backend/assets/account_codes.csv', index=False)
        logger.info("Account Codes saved as csv")
        return account_code_df
    
    def save_cost_codes(self):
        cost_code_list = []
        cost_code_url = f"{self.base_url}DistributionRules?$select=FactorCode,FactorDescription&$orderby=FactorCode"

        while cost_code_url:
            try:
                response = self.session.get(cost_code_url, verify=False)

                if 'value' in response.json():
                    cost_code_values = response.json()['value']
                    for num in cost_code_values:
                        cost_code_list.append({
                            "FactorCode": num["FactorCode"],
                            "FactorDescription": num["FactorDescription"]
                        })

                    if 'odata.nextLink' in response.json():
                        next_link = response.json()['odata.nextLink']
                        cost_code_url = f"{self.base_url}{next_link}"
                    else:
                        break
                else:
                    break

            except Exception as e:
                logger.error(f"Error fetching Cost Codes: {e}")
                raise
        logger.info("Successfully fetched all pages of Cost Codes.")

        cost_code_df = pd.DataFrame(cost_code_list)
        cost_code_df.to_csv('backend/assets/cost_codes.csv', index=False)
        logger.info("Cost Codes saved as csv")

        return cost_code_df
    
    def save_business_partners(self):
        bp_list = []
        bp_url = f"{self.base_url}BusinessPartners?$select=CardCode,CardName,CardType&$orderby=CardName&$filter=startswith(CardCode, 'V')"

        while bp_url:
            try:
                response = self.session.get(bp_url, verify=False)
                if 'value' in response.json():
                    bp_values = response.json()['value']
                    for num in bp_values:
                        bp_list.append({
                            "CardCode": num["CardCode"],
                            "CardName": num["CardName"]
                        })

                    if 'odata.nextLink' in response.json():
                        next_link = response.json()['odata.nextLink']
                        bp_url = f"{self.base_url}{next_link}"
                    else:
                        break
                else:
                    break

            except Exception as e:
                logger.error(f"Error fetching business partners: {e}")
                raise
        logger.info("Successfully fetched all pages of Business Partners. (Vendors only)")

        bp_df = pd.DataFrame(bp_list)
        bp_df.dropna(inplace=True)
        bp_df.to_csv('backend/assets/vendor_list.csv', index=False)
        logger.info("Business Partners saved as csv")

        return bp_df
    
    def post_items_to_sap(self, item: dict):
        try:
            logger.info(f"Posting item to SAP: {item}")
            response = self.session.post(f"{self.base_url}Items", json=item, verify=False)
            if response.status_code == 201:
                logger.info("Successfully posted item to SAP.")
                code_given_by_sap = {
                    'ItemCode': response.json()['ItemCode'],
                    'InventoryUoMEntry': response.json().get('InventoryUoMEntry')
                }
                return code_given_by_sap
            else:
                logger.error(f"Failed to post item. Status: {response.status_code}, Response: {response.text}")
                return None
        except Exception as e:
            logger.error(f"Error posting item: {e}")
        finally:
            self.save_items_to_csv()

    def post_purchase_invoice(self, invoice: dict):
        try:
            logger.info(f"Posting purchase invoice to SAP: {invoice}")
            response = self.session.post(f"{self.base_url}PurchaseInvoices", json=invoice, verify=False)
            if response.status_code == 201:
                logger.info("Successfully posted purchase invoice to SAP.")
                return response.json()
            else:
                error_detail = response.text
                try:
                    error_json = response.json()
                    error_detail = error_json.get("error", {}).get("message", {}).get("value", response.text)
                except:
                    pass
                logger.error(f"Failed to post purchase invoice. Status: {response.status_code}, Error: {error_detail}")
                return {"error": True, "status_code": response.status_code, "detail": error_detail}
        except Exception as e:
            logger.error(f"Error posting purchase invoice: {e}")
            return {"error": True, "detail": str(e)}


# Shared singleton — import this instead of instantiating SAPClient() directly
sap_client = SAPClient()

