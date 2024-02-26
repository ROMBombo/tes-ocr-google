from fastapi import APIRouter, Response, status, Form, UploadFile, File, Header
from typing import List
from google.cloud import documentai_v1 as documentai
import vertexai
from vertexai.language_models import TextGenerationModel
import json
import re
import time
from pydantic import BaseModel
from datetime import datetime

router = APIRouter(
    prefix="/v1/google",
    tags=["Google"]
)

# config
project_id = 'development-ai-414204'
location = 'us'
processor_id = '32df38d65d2e2038' # processor doc ocr
processor_id_fp = '69f9e2b08b16b080' # processor form parser

def set_mime_type(file_format):
    if file_format == 'pdf':
        mime_type = 'application/pdf'
    elif file_format == 'gif':
        mime_type = 'image/gif'
    elif file_format in ['tiff', 'tif']:
        mime_type = 'image/tiff'
    elif file_format in ['jpg', 'jpeg']:
        mime_type = 'image/jpeg'
    elif file_format == 'png':
        mime_type = 'image/png'
    elif file_format == 'bmp':
        mime_type = 'image/bmp'
    else:
        mime_type = 'image/webp'
    return mime_type

class RequestVertexAILLM(BaseModel):
    custom_prompt: str
    # output: str

@router.post('/vertexai-llm')
async def vertexai_llm(request: RequestVertexAILLM,response: Response, x_key: str = Header()):
    if x_key != "x-dev-crm":
        response.status_code = status.HTTP_401_UNAUTHORIZED
        return {
            "code": response.status_code,
            "status": "UNAUTHORIZED",
            "message": "x-key is not valid"
        }
    
    vertexai.init(project=project_id, location="us-central1")
    parameters = {
        "candidate_count": 1,
        "max_output_tokens": 1024,
        "temperature": 1,
        "top_k": 40
    }
    model = TextGenerationModel.from_pretrained("text-unicorn@001")
    try:
        result = model.predict(request.custom_prompt, **parameters)
        result = result.text
        
        json_string_cleaned = result.strip('`')

        json_string_cleaned = re.sub(r'^.*?\{', '{', json_string_cleaned, flags=re.DOTALL)

        json_string_cleaned = json_string_cleaned.replace('\r', '').replace('\n', '')
        try:
            result = json.loads(json_string_cleaned)
        except:
            result = result

        return {
            "code": 200,
            "status": "OK",
            "response": result
        }
    except Exception as e:
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return {
            "code": response.status_code,
            "status": "INTERNAL_SERVER_ERROR",
            "message": str(e)
        }

@router.post('/ocr-document/bulk')
async def ocr_bulk(response: Response, files: List[UploadFile] = File(), dokumen: str = Form(), custom_prompt: str = Form(), x_key: str = Header()):
    start_time_total = time.time()
    if x_key != 'x-dev-crm':
        response.status_code = status.HTTP_401_UNAUTHORIZED
        return {
            "code": response.status_code,
            "status": "UNAUTHORIZED",
            "message": "x-key is not valid"
        }
    
    for file in files:
        file_format = file.filename.split(".")[-1]
        file_format = file_format.lower()

        if file_format not in ['pdf', 'gif', 'tiff', 'tif', 'jpg', 'jpeg', 'png', 'bmp', 'webp']:
            response.status_code = status.HTTP_400_BAD_REQUEST
            return {
                "code": response.status_code,
                "status": "BAD_REQUEST",
                "message": "The file format is not supported"
            }
    
    if dokumen not in ['ktp', 'ijazah', 'transkrip']:
        response.status_code = status.HTTP_404_NOT_FOUND
        return {
            "code": response.status_code,
            "status": "NOT_FOUND",
            "message": "dokumen must be `ktp`, `ijazah`, or `transkrip`"
        }

    else:
        documentai_client = documentai.DocumentProcessorServiceClient()
            
        resource_name = documentai_client.processor_path(project_id, location, processor_id)

        results = []

        for file in files:
            start_time = time.time()

            note = ""

            content = await file.read()
            
            file_format = file.filename.split(".")[-1]
            file_format = file_format.lower()

            mime_type = set_mime_type(file_format)

            raw_document = documentai.RawDocument(content=content, mime_type=mime_type)

            try:
                request = documentai.ProcessRequest(name=resource_name, raw_document=raw_document)

                data = documentai_client.process_document(request=request)
                sukses = 1
            except Exception as e:
                sukses = 0
                note = "Failed to process the file with Document AI. Message: " + str(e)
                prompt = ''
                result = ''
                text_llm_result = ''

            if sukses != 0:
                vertexai.init(project=project_id, location="us-central1")
                parameters = {
                    "candidate_count": 1,
                    "max_output_tokens": 1024,
                    "temperature": 1,
                    "top_k": 40
                }
                model = TextGenerationModel.from_pretrained("text-unicorn@001")

                prompt = data.document.text + "\n" + custom_prompt

                try:
                    result = model.predict(prompt,**parameters)
                except Exception as e:
                    sukses = 0
                    note = "Failed to process the file with LLM. Message: " + str(e)
                    result = ''
                    text_llm_result = ''

                if sukses != 0:
                    text_llm_result = result.text

                    json_string_cleaned = text_llm_result.strip('`')

                    json_string_cleaned = re.sub(r'^.*?\{', '{', json_string_cleaned, flags=re.DOTALL)

                    json_string_cleaned = json_string_cleaned.replace('\r', '').replace('\n', '')
                    try:
                        result = json.loads(json_string_cleaned)
                    except:
                        result = result.text
            
            end_time = time.time()
            response_time = end_time - start_time

            if dokumen == 'ktp':
                if 'nik' in custom_prompt.lower():
                    if isinstance(result, dict):
                        if 'nik' in result.keys():
                            if len(result['nik']) != 16:
                                sukses = 0
                                note = "NIK must be 16 characters"
                        elif 'NIK' in result.keys():
                            if len(result['NIK']) != 16:
                                sukses = 0
                                note = "NIK must be 16 characters"

            results.append(
                {
                    "result": result,
                    "status": sukses,
                    "note": note,
                    "response_time": str(round(response_time, 2)) + " seconds"
                }
            )

        end_time_total = time.time()
        total_response_time = end_time_total - start_time_total
        
        return {
            "code": status.HTTP_200_OK,
            "status": "OK",
            "data": results,
            "total_response_time": str(round(total_response_time, 2)) + " seconds"
        }
