import gradio as gr
import requests
import json
import os
import logging
from databricks.sdk import WorkspaceClient

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from databricks.sdk.service import apps

def get_token_and_endpoint():
    """Get Databricks token and API endpoint with robust error handling"""
    token = os.getenv('DATABRICKS_TOKEN')
    endpoint = os.getenv('API_ENDPOINT', "https://adb-2411896520795414.14.azuredatabricks.net/serving-endpoints/Chatbot_poc/invocations")

    
    logger.info(f"Initial endpoint: {endpoint}")
    logger.info(f"Token from env vars: {'Available' if token else 'Not available'}")
    
    # If not available in env vars, try other methods
    if token is None:
        try:
            # Try databricks.sdk
            logger.info("Attempting to get token from WorkspaceClient...")
            w = WorkspaceClient()
            token = w.current_user.token.token_value
            logger.info("Successfully got token from WorkspaceClient")
        except Exception as e:
            logger.warning(f"Failed to get token from WorkspaceClient: {str(e)}")
            
            try:
                # Try dbutils in notebook context
                logger.info("Attempting to get token from dbutils...")
                import dbutils.secrets
                token = dbutils.secrets.get(scope="poc_scope", key="accessForDatabricksAPI")
                logger.info("Successfully got token from dbutils")
            except Exception as e2:
                logger.warning(f"Failed to get token from dbutils: {str(e2)}")
                
                # Final fallback - hardcoded token for testing
                logger.warning("Using fallback hardcoded token (for testing only)")
                # Replace with a valid token for testing
                token = None
    
    return token, endpoint

# def get_token_and_endpoint():
#     """Get Databricks token and API endpoint with robust error handling"""
#     token = None
    
#     # Try accessing the app resource (using correct method)
#     try:
#         from databricks.sdk.core import ApiClient
#         from databricks.sdk.service.provisioning import AppDeployment
        
#         # For Databricks Apps, use the resource API
#         deployment = AppDeployment()
#         resource = deployment.get_resource("secret", resource_type="ScopeSecret")
#         token = resource.value
#         logger.info("Successfully got token from App resources")
#     except Exception as e:
#         logger.warning(f"Failed to get token from App resources: {str(e)}")
    
#     # If not found, try environment variable
#     if not token:
#         token = os.getenv('DATABRICKS_TOKEN')
#         logger.info(f"Token from env vars: {'Available' if token else 'Not available'}")
    
#     # If still not found, try WorkspaceClient (with correct attribute path)
#     if not token:
#         try:
#             from databricks.sdk import WorkspaceClient
#             w = WorkspaceClient()
#             # The correct attribute path is different than what you had
#             # This varies by SDK version, so try both approaches
#             try:
#                 token = w.config.token
#                 logger.info("Successfully got token from WorkspaceClient.config.token")
#             except:
#                 token = w.api_client.token
#                 logger.info("Successfully got token from WorkspaceClient.api_client.token")
#         except Exception as e:
#             logger.warning(f"Failed to get token from WorkspaceClient: {str(e)}")
    
#     # Final option - use a token stored in app settings/environment
#     if not token:
#         # For Databricks App deployment, you should set this in your app settings
#         token = os.getenv('APP_TOKEN')
#         logger.info(f"Token from APP_TOKEN env var: {'Available' if token else 'Not available'}")
    
#     # Set endpoint
#     endpoint = os.getenv('API_ENDPOINT', "https://adb-2411896520795414.14.azuredatabricks.net/serving-endpoints/Chatbot_poc/invocations")
#     logger.info(f"Using endpoint: {endpoint}")
    
#     # If no token found, raise exception or return None
#     if not token:
#         logger.error("No valid token available")
    
#     return token, endpoint

def respond(message, history):
    """Handler for chat messages with comprehensive error handling"""
    logger.info(f"Received message: {message}")
    
    if not message or len(message.strip()) == 0:
        return "Please enter a question."
    
    # Get credentials
    token, endpoint = get_token_and_endpoint()
    
    if not token:
        logger.error("No valid token available")
        return "ERROR: Could not authenticate with the Databricks API. Please check your configuration."
    
    # Prepare request
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {token}'
    }
    
    payload = {
        "dataframe_records": [{"query": message}]
    }
    
    logger.info(f"Sending request to endpoint: {endpoint}")
    
    # Make API call with robust error handling
    try:
        response = requests.post(
            endpoint, 
            json=payload, 
            headers=headers, 
            timeout=30
        )
        
        logger.info(f"Response status code: {response.status_code}")
        
        # Check for non-200 response
        if response.status_code != 200:
            error_msg = f"ERROR: API returned status code {response.status_code}"
            
            # Try to get more info from response
            try:
                response_text = response.text[:500]
                logger.error(f"Error response: {response_text}")
                error_msg += f"\nDetails: {response_text}"
            except:
                logger.error("Could not read error response text")
            
            return error_msg
        
        # Try to parse JSON response
        try:
            response_json = response.json()
            logger.info("Successfully parsed JSON response")
            
            # Check expected structure
            if "predictions" in response_json:
                return response_json["predictions"]
            else:
                logger.error(f"Unexpected response format: {json.dumps(response_json)[:200]}")
                return "ERROR: Unexpected response format from API. Missing 'predictions' field."
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {str(e)}")
            logger.error(f"Response content: {response.text[:200]}")
            return f"ERROR: Could not parse API response as JSON. Details: {str(e)}"
            
    except requests.exceptions.Timeout:
        logger.error("Request timed out")
        return "ERROR: Request to API timed out. Please try again later."
    except requests.exceptions.ConnectionError:
        logger.error(f"Connection error to {endpoint}")
        return "ERROR: Could not connect to the API endpoint. Please check your network connection."
    except Exception as e:
        logger.error(f"Unexpected error: {type(e).__name__} - {str(e)}")
        return f"ERROR: {type(e).__name__} - {str(e)}"

# Create the Gradio interface
demo = gr.ChatInterface(
    respond,
    chatbot=gr.Chatbot(show_label=False, container=False, show_copy_button=True, bubble_full_width=True),
    textbox=gr.Textbox(placeholder="Ask me a question about your appliance",
                      container=False, scale=7),
    title="Appliance Chatbot Demo",
    description="This chatbot is a demo example to interact with an appliance chatbot",
    examples=[["What do I do if my range burner won't light?"],],
    cache_examples=False,
    retry_btn=None,
    undo_btn=None,
    clear_btn="Clear",
)

# Test the API connection on startup
def test_api_connection():
    logger.info("===== TESTING API CONNECTION =====")
    token, endpoint = get_token_and_endpoint()
    
    if not token:
        logger.error("No token available for API test")
        return
    
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {token}'
    }
    
    payload = {
        "dataframe_records": [{"query": "Test question"}]
    }
    
    try:
        logger.info(f"Sending test request to {endpoint}")
        response = requests.post(endpoint, json=payload, headers=headers, timeout=30)
        logger.info(f"Test response status: {response.status_code}")
        logger.info(f"Test response preview: {response.text[:100]}...")
    except Exception as e:
        logger.error(f"Test API call failed: {str(e)}")

# Run the API test
test_api_connection()

# Launch the app
if __name__ == "__main__":
    logger.info("Starting Gradio app on port 8000")
    demo.launch(server_name="0.0.0.0", server_port=8000)