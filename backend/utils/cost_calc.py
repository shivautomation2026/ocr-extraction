from google import genai
from google.oauth2 import service_account
from backend.core.config import settings

def main():

    credentials = service_account.Credentials.from_service_account_file(
    settings.GOOGLE_APPLICATION_CREDENTIALS,
    scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )   

    client = genai.Client(
        vertexai=True,
        credentials=credentials,
        project=settings.GOOGLE_CLOUD_PROJECT.get_secret_value(),
        location=settings.GOOGLE_CLOUD_LOCATION,
    )

    

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="2+2",
        # config=genai.types.GenerateContentConfig(
        #     temperature=0.1
        # )
    )

    usage = response.usage_metadata
    prompt_tokens = usage.prompt_token_count
    output_tokens = usage.candidates_token_count
    thought_tokens = usage.thoughts_token_count
    total_output_tokens = output_tokens + thought_tokens
    total_tokens = usage.total_token_count



    print("Prompt tokens:", prompt_tokens)
    print("Output tokens:", output_tokens)
    print("Thought tokens:", thought_tokens)
    print("Total output tokens:", total_output_tokens)
    print("Total tokens:", total_tokens)


if __name__ == "__main__":
    main()
