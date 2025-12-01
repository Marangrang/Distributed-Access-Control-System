"""Helper functions for AWS Secrets Manager integration."""
import boto3
from botocore.exceptions import ClientError
import json
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def get_secret(secret_name: str, region_name: str = "us-east-1") -> Dict[str, Any]:
    """
    Retrieve secret from AWS Secrets Manager.
    
    Args:
        secret_name: Name of the secret in AWS Secrets Manager
        region_name: AWS region (default: us-east-1)
        
    Returns:
        Dict containing the secret values
        
    Raises:
        ClientError: If secret retrieval fails
    """
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name='us-east-1'
    )

    try:
        get_secret_value_response = client.get_secret_value(SecretId=secret_name)
        
        if 'SecretString' in get_secret_value_response:
            secret = get_secret_value_response['SecretString']
            return json.loads(secret)
        else:
            # Handle binary secrets
            import base64
            decoded_binary_secret = base64.b64decode(
                get_secret_value_response['SecretBinary']
            )
            return json.loads(decoded_binary_secret)
            
    except ClientError as e:
        error_code = e.response['Error']['Code']
        
        if error_code == 'ResourceNotFoundException':
            logger.error(f"Secret '{secret_name}' was not found")
        elif error_code == 'InvalidRequestException':
            logger.error(f"Invalid request: {e}")
        elif error_code == 'InvalidParameterException':
            logger.error(f"Invalid parameters: {e}")
        else:
            logger.error(f"Error retrieving secret: {e}")
            
        raise


def put_secret(secret_name: str, secret_value: Dict[str, Any], 
               description: str = "", region_name: str = "us-east-1") -> Dict[str, Any]:
    """
    Store secret in AWS Secrets Manager.
    
    Args:
        secret_name: Name for the secret
        secret_value: Dict containing secret values
        description: Optional description
        region_name: AWS region
        
    Returns:
        Dict with AWS response
        
    Raises:
        ClientError: If secret creation fails
    """
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name='us-east-1'
    )

    try:
        secret_string = json.dumps(secret_value)
        
        response = client.create_secret(
            Name=secret_name,
            SecretString=secret_string,
            Description=description or f"Secret for {secret_name}"
        )
        
        logger.info(f"✓ Secret '{secret_name}' created successfully")
        return response
        
    except client.exceptions.ResourceExistsException:
        logger.warning(f"Secret '{secret_name}' already exists")
        # Update existing secret
        response = client.update_secret(
            SecretId=secret_name,
            SecretString=secret_string
        )
        logger.info(f"✓ Secret '{secret_name}' updated successfully")
        return response
        
    except ClientError as e:
        logger.error(f"Error creating/updating secret: {e}")
        raise