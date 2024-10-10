"""
python custom_integration.py
"""
# *********************************************************************************************************************

from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))
print(sys.path)

# standard imports
import datetime
import os

# 3rd party imports
# import aiohttp # TODO: what does this do and why?  Is it just out tests.
import boto3 # TODO: again this issue that all generated so no auto completino in VS Code
import requests

# custom imports
import hf_integration.lexv2_converters

MAX_PAGES=100 # have some sort of maximum loops

class lexv2_apis:
    """This class implements a wrapper around boto3 apis for Lex v2"""
    
    def __init__(self, 
        aws_credentials_file: str = "~/.aws/credentials", 
        region_name: str = "us-east-1" # individual method calls can override this overrides configuration file
        # doesn't seem to actually work for things like list bots
        # aws_config_file: str = "~/.aws/config" assume all in credentials
        ) -> None:
        """Authorization
        Uses standard boto3 authentication with assumed standard files
        i.e assumes you have done aws configure on the machine you expect it to run on
        assumption would be some part of the process of the docker starting would have to replicate this
        TODO: better replaced with secrets and env variables

        Connects to 
        https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/lexv2-models.html
        https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/lexv2-runtime.html

        """
        
        if not os.path.isfile(aws_credentials_file):
            raise RuntimeError("AWS Credential file not instantiated")
        
        self.region_name = region_name
        self.modelclient = boto3.client('lexv2-models',region_name=self.region_name)
        self.iamclient = boto3.client('iam')
        self.runtimeclient = boto3.client('lexv2-runtime',region_name=self.region_name)
        
    def get_iam_user(self) -> dict:
        """Returns who is making the boto3 calls"""
        return self.iamclient.get_user()
                
    def create_bot(self, 
                   bot_name: str,
                   description: str = "Blah",
                   aws_role: str = "role/aws-service-role/lexv2.amazonaws.com/AWSServiceRoleForLexV2Bots", 
                   # This is standard - will read the rest of the ARN from the user calling it
                   child_directed: bool = False,
                   # TODO: would need some sort of passthrough from HF for this ideally at the moment saying no.
                   idle_session_TTL_seconds: int = 3600 # default to an hour here
                   # botTags
                   # testBotAliasTags
                   # botType    (Bot/BotNetwork)
                   # botMembers (Only if BotNetork)
                   ) -> dict:
        """Create Bot as a basic skeleton
        https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/lexv2-models/client/create_bot.html
        Requires: AmazonLexFullAccess permission level for the calling user
        Also requires: bedrock:ListFoundationModels, bedrock:InvokeModel if doing anything generative in the GUI
        """
        
        # Going to get the user to work out the account that we are working with
        # The role and bot must be under the same project that the user is in
        user_dict = self.get_iam_user()
        user_arn = user_dict["Arn"]
        assert isinstance(user_arn,str)
        account_info = user_arn.split(":user/")[0]      
        role_arn = f'{account_info}:role/{aws_role}'
        
        # create the bot
        response_dict = self.modelclient.create_bot(
            botName=bot_name,
            description=description,
            roleArn=role_arn,
            dataPrivacy={
                'childDirected': child_directed
            },
            idleSessionTTLInSeconds=idle_session_TTL_seconds,
            botType='Bot' # currently supporting bot not bot network.
        )

        return response_dict

    def delete_bot(self, 
                   bot_id: str, # get from the URL https://us-east-1.console.aws.amazon.com/lexv2/home?region=us-east-1#bot/XRZZ68GFZO
                   force: bool = False, # skipResourceInUseCheck 
                   ) -> dict:
        """Delete Bot
        force: True|False default True - skips resource in use check and just deletes it.
        https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/lexv2-models/client/delete_bot.html
        """
        response_dict = self.modelclient.delete_bot(
            botId=bot_id,
            skipResourceInUseCheck=force
        )
        
        return response_dict  

    def list_bots(self,
        sort_by: dict = {'attribute': 'BotName','order': 'Ascending'},
        max_results: int = 10) -> list:
        """ Provides a botSummaries list - deals with paging from AWS
        doesn't implement Amazon side filters
        https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/lexv2-models/client/list_bots.html"""
         
        response_dict = self.modelclient.list_bots(sortBy=sort_by,maxResults=max_results)
            
        bot_summaries = []
        bot_summaries.extend(response_dict["botSummaries"])
        
        i = 1
        while "nextToken" in response_dict:
            if i > MAX_PAGES:
                raise RuntimeError("Too many pages of Bots to retrieve")
            response_dict = self.modelclient.list_bots(sortBy=sort_by,maxResults=max_results,nextToken=response_dict["nextToken"])
            bot_summaries.extend(response_dict["botSummaries"])

        for summary in bot_summaries:
            summary = _clean_dates_to_isostrings(summary)
        
        return bot_summaries
    
    def import_bot(self, 
                   zip_file_name: str, #file name including extension
                   tempdir: str # with the file in 
                   ) -> dict:
        """Handles creating the upload url from the zip and then starting and monitoring the import till it's done
        Supports bot import not bot locale input.
        """
        
        # get the upload url
        # https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/lexv2-models/client/create_upload_url.html
        url_response_dict = self.modelclient.create_upload_url()
        import_id = str(url_response_dict["importId"])
        upload_url = str(url_response_dict["uploadUrl"])
        print(import_id)
        print(upload_url)
        
        
        # Upload the file to the presigned link
        fq_zip_file_name = os.path.join(tempdir,zip_file_name)
        with open(fq_zip_file_name,mode='rb') as upload_file:
            upload_data = upload_file.read()
        requests_response = requests.request("PUT", upload_url, 
                                  data=upload_data, 
                                  headers={"Content-Type": "application/zip"}
        )
        
        # Start the import
        # https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/lexv2-models/client/start_import.html
        
        # need the bot information
        workdir = hf_integration.lexv2_converters._extract_zip_to_tempdirdir()
        hf_integration.lexv2_converters._check_manifest()
        
        

        return {"something":"anything"}

def _clean_dates_to_isostrings(some_dict: dict) -> dict:
    """Recursived checks all keys for types, if datetime turns to iso string"""
    for key in some_dict:
        some_value = some_dict[key]
        if isinstance(some_value,datetime.datetime):
            some_dict[key] = some_value.isoformat()
        elif isinstance(some_value,dict):
            _clean_dates_to_isostrings(some_dict=some_value)
    return some_dict


