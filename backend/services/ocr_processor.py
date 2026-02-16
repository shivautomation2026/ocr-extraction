import os
import datetime
import json
from mistralai import Mistral
from dotenv import load_dotenv
from ..models import OCRResponse
from ..database import  add_default_prompt
from ..core.config import settings
from ..utils.cost_tracker import LLMCostTracker, extract_langchain_usage
import logging
# from google import genai 
from langchain_google_genai import ChatGoogleGenerativeAI


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# try:
#     api_key = settings.MISTRAL_API_KEY
#     if not api_key:
#         raise ValueError("MISTRAL_API_KEY is not set or is empty in environment variables")
#     client = Mistral(api_key=api_key)
#     model = "pixtral-large-latest"
#     print("✅ Successfully initialized Mistral client")
# except KeyError:
#     print("❌ Error: MISTRAL_API_KEY not found in environment variables")
#     print("Please make sure you have a .env file with MISTRAL_API_KEY set")
#     sys.exit(1)
# except Exception as e:
#     print(f"❌ Error initializing Mistral client: {e}")
#     sys.exit(1)

class OCR_Processor:
    def __init__(self):
        api_key = settings.MISTRAL_API_KEY
        if not api_key:
            logger.critical("MISTRAL_API_KEY is not set or is empty in environment variables")
            raise ValueError("MISTRAL_API_KEY is not set or is empty in environment variables")
        self.client = Mistral(api_key=api_key)
        self.ocr_model = "mistral-ocr-latest"
        # self.gemini_client = genai.Client(api_key=settings.GEMINI_API_KEY)
        self.llm = ChatGoogleGenerativeAI(model = settings.GEMINI_MODEL_NAME, temperature=0,project=settings.GOOGLE_CLOUD_PROJECT, location=settings.GOOGLE_CLOUD_LOCATION)
        self.model = "mistral-small-latest"
        self.cost_tracker = LLMCostTracker(model_name=settings.GEMINI_MODEL_NAME)
        logger.info(f"OCR_Processor initialized with model: {self.ocr_model} {self.llm.model}") 

    def extract_raw_text_from_pdf(self, file_path):
        """Uploads the PDF and extracts raw text using Mistral."""
        if not os.path.exists(file_path):
            logger.warning(f"Filepath not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            with open(file_path, "rb") as f:
                uploaded_pdf = self.client.files.upload(

                    file={
                        "file_name": os.path.basename(file_path),
                        "content": f,
                    },
                    purpose="ocr"
                )

            logger.info(f"Successfully uploaded {uploaded_pdf.filename} for OCR processing")

            signed_url = self.client.files.get_signed_url(file_id=uploaded_pdf.id)
            
            ocr_response = self.client.ocr.process(
                model = self.ocr_model,
                document = {
                    "type": "document_url",
                    "document_url": signed_url.url
                },
                include_image_base64 = True
            )

            self.cost_tracker.track_fixed_cost(
                cost=0.002,
                operation='ocr_extraction',
                model_name=self.ocr_model
            )

            # messages = [
            #     {
            #         "role": "user",
            #         "content": [
            #             {
            #                 "type": "text",
            #                 "text": "You are an intelligent document parser, and your role is to extract the text from the PDF below as you read naturally. Do not hallucinate."
            #             },
            #             {
            #                 "type": "document_url",
            #                 "document_url": signed_url.url
            #             }
            #         ]
            #     }
            # ]

            # chat_response = self.client.chat.complete(
            #     model= self.model,
            #     messages=messages
            # )

            logger.info("Extracted text from PDF using OCR model")
            
            return ocr_response.pages[0].markdown

        except Exception as e:
            logger.error(f"Error during PDF text extraction: {e}")
            raise

    def extract_vendor_details(self, raw_text, user_prompt=""):
        try:
            if user_prompt.strip():
                logger.info("Using custom user prompt for extraction")
                prompt_template = f"""
                {user_prompt.strip()}
        Text:
        {raw_text}

        Important: Return ONLY the raw JSON object without any markdown formatting or code blocks. NO explanations, no headings, no extra text.
        """
            else:
                logger.info("Using default prompt for extracting in json format")
                prompt_template = f"""
                You are an expert document parser specializing in commercial documents like invoices, bills, etc. Extract the following structured data from the document text and return it as pure JSON without any markdown formatting or code blocks:
                        - vendor_details: name, address, phone, email, website, PAN
                        - customer_details: name, address, contact, PAN (usually below vendor_details)
                        - invoice_details: bill_number, bill_date, transaction_date, mode_of_payment, finance_manager, authorized_signatory
                        - payment_details: total, in_words, discount, taxable_amount, vat, net_amount
                        - line_items (list): hs_code, description, qty, rate, amount
                            Rules:
                                1. Extract only the fields listed; do not guess or add extra fields.
                                2. If a field is missing, set its value as "".
                                3. Use context ('Vendor', 'Supplier', 'Bill To', 'Customer', etc.) to distinguish parties. If unclear, the first business is Vendor,                        the second is Customer.
                                4. Each line_item must include hs_code and description; qty, rate, and amount are optional.
                                5. Always return the result strictly in the following JSON structure.
                                6. PAN numbers are typically boxed or near labels like 'PAN No.', and follow a 9-digit (Nepal) format.
                                7. For Dates in bill_date put - between year, month and day like YYYY-MM-DD, if the date exceeds 2080 then convert the following Bikram Sambat (BS) date to the Gregorian (AD) calendar.
                                8. Return  JSON without any markdown formatting or code blocks.

                                Return the standard structured JSON format shown below:
                                {{
                                    "vendor_details": {{
                                    "name": "",
                                    "address": "", 
                                    "contact_number": "", 
                                    "email": "",
                                    "website": "",
                                    "pan_number": "" // This is the pan number/ VAT number of a company
                                    }},
                                    "customer_details": {{
                                        "name": "",
                                        "address": "",
                                        "contact_number": "",
                                        "pan_number": ""// This is the pan number/ VAT number of a company
                                    }},
                                    "invoice_details": {{
                                        "bill_number": "",
                                        "posting_date": "", // Current date is {datetime.date.today().strftime("%Y-%m-%d")}
                                        "bill_date": "",
                                        "nepali_miti": "", // This is the date in nepali calendar if available
                                        "mode_of_payment": "",
                                        "finance_manager": "",
                                        "authorized_signatory": "",
                                        "lc_no": "" // This is the letter of credit number of the company 
                                    }},
                                    "payment_details": {{
                                        "net_amount": "",  // It can also be taxable total amount or this is the before taxes discount and vat
                                        "discount_amount": "" , // this should be in amount not percentage
                                        "taxable_amount": "" , // this should be in amount not percentage and it is after taxes
                                        "vat_percentage": "", // this should be in percentage
                                        "vat_amount": "", // this should be in amount not percentage it is only the vat percentage of the net amount
                                        "grand_total": "",  // This is the last amount after all calculations such as after adding vat_amount taxable_amount and decreasing discount_amount and the amount should be equal to total amount in words
                                        "grand_total_in_words": "",
                                    }},
                                    "line_items": [
                                        {{
                                        "hs_code": "",
                                        "description": "", // This is the line items of a bill in a tabular for which can be a product or a service, if there is no description do not include this item in the list
                                        "quantity": "", // should be in integer
                                        "rate": "", // should be in float
                                        "amount": "" // should be in float
                                        }}
                                    ]
                                    }}
                                    Text:
                                    {raw_text}

                                    Important: Return ONLY the standard JSON schema object without any markdown formatting or code blocks. No explanations, no headings, no extra text.
                                    """
                add_default_prompt(prompt_template)

            messages = [
                {
                    "role": "user",
                    "content": prompt_template
                }
            ]
            
            output = self.llm.invoke(prompt_template)

            usage = extract_langchain_usage(output)
            self.cost_tracker.track_llm_usage(
                input_tokens=usage["input_tokens"],
                output_tokens=usage["output_tokens"],
                operation="extract_vendor_details",
                model_name=self.llm.model
            )

            # chat_response = self.gemini_client.models.generate_content(
            #     model="gemini-2.5-flash",
            #     contents=prompt_template,
            #     config = {
            #         "response_mime_type": "application/json",
            #     }
            # )
            # # output = chat_response.choices[0].message.content
            # output = chat_response.text

            logger.info("Extracted vendor details using OCR model")

            # return json.loads(output)
            return output.content
            
        except Exception as e:
            logger.error(f"Error during vendor details extraction: {e}")
            raise

    def process_file(self, file_path, user_prompt="") -> OCRResponse:
        try:
            # Validate file path
            if not file_path or not file_path.strip():
                logger.error("Empty file path provided")
                return OCRResponse(
                    status="error",
                    message="Empty file path provided",
                    content={},
                    extracted_text=""
                )

            if file_path.endswith(('.pdf', '.PDF')):
                text = self.extract_raw_text_from_pdf(file_path)
                
                # Validate extracted text
                if not text or not text.strip():
                    logger.error("No text extracted from PDF")
                    return OCRResponse(
                        status="error",
                        message="No text could be extracted from the PDF",
                        content={},
                        extracted_text=""
                    )
            else:
                logger.error("Unsupported file type. Only PDF files are supported.")
                return OCRResponse(
                    status="error",
                    message="Unsupported file type. Only PDF files are supported.",
                    content={},
                    extracted_text=""
                )
            
            result = self.extract_vendor_details(text, user_prompt)
            
            # Validate result before processing
            if not result or not result.strip():
                logger.error("Empty result from vendor details extraction")
                return OCRResponse(
                    status="error",
                    message="Model returned empty response for data extraction",
                    content={},
                    extracted_text=text
                )

            if isinstance(result, str):
                try:
                    # Clean the result before parsing
                    cleaned_result = result.strip()
                    
                    # Remove markdown formatting if present
                    if cleaned_result.startswith("```json"):
                        cleaned_result = cleaned_result[7:]
                    elif cleaned_result.startswith("```"):
                        cleaned_result = cleaned_result[3:]
                    
                    if cleaned_result.endswith("```"):
                        cleaned_result = cleaned_result[:-3]
                    
                    cleaned_result = cleaned_result.strip()
                    
                    # Final check for empty result
                    if not cleaned_result:
                        logger.error("Result is empty after cleaning")
                        return OCRResponse(
                            status="error",
                            message="Empty response from model after cleaning",
                            content={"raw_response": result},
                            extracted_text=text
                        )
                    
                    result = json.loads(cleaned_result)
                    logger.info("Successfully parsed JSON response from model")
                except json.JSONDecodeError as e:
                    logger.error(f"JSON parsing error: {e}")
                    logger.error(f"Raw result: {repr(result)}")
                    return OCRResponse(
                        status="error",
                        message=f"Failed to parse JSON from result: {e}",
                        content={
                            "raw_response": result,
                            "cleaned_response": cleaned_result if 'cleaned_result' in locals() else result
                        },
                        extracted_text=text
                    )
                
            return OCRResponse(
                status="success",
                message="Text extracted and structured successfully",
                content=result,
                extracted_text=text
            )
            
        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            return OCRResponse(
                status="error",
                message=f"File not found: {str(e)}",
                content={},
                extracted_text=""
            )
        except Exception as e:
            logger.error(f"Unexpected error during file processing: {e}")
            return OCRResponse(
                status="error",
                message=f"Unexpected error during file processing: {str(e)}",
                content={},
                extracted_text=""
            )



